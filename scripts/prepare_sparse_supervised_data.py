"""Freeze disjoint COCO subsets for a small supervised sparse-routing pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_sparse_esmoe import assert_disjoint


def select_splits(names, excluded, counts, seed):
    """Assign each eligible COCO ID to exactly one deterministic split."""
    eligible = sorted(set(names) - set(excluded))
    random.Random(seed).shuffle(eligible)
    if min(counts) < 1 or sum(counts) > len(eligible):
        raise ValueError("positive split sizes must fit the eligible pool")
    result, offset = {}, 0
    for split, count in zip(("train", "dev", "test"), counts):
        result[split] = eligible[offset : offset + count]
        offset += count
    return result


def select_trainval_splits(train_names, val_names, excluded, counts, seed):
    """Keep official training images out of the two validation subsets."""
    train_pool = sorted(set(train_names) - set(excluded))
    val_pool = sorted(set(val_names) - set(excluded))
    if set(train_pool) & set(val_pool):
        raise ValueError("official train/val pools overlap")
    if min(counts) < 1 or counts[0] > len(train_pool) or sum(counts[1:]) > len(val_pool):
        raise ValueError("split sizes exceed official source pools")
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    return {
        "train": train_pool[: counts[0]],
        "dev": val_pool[: counts[1]],
        "test": val_pool[counts[1] : counts[1] + counts[2]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs=3, type=int, default=(64, 32, 64))
    parser.add_argument("--train-source", choices=("val2017", "train2017"), default="val2017")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("use a fresh output directory")
    prior = json.loads(args.exclude_report.read_text())
    excluded_rows = prior["calibration"] + prior["evaluation"]
    excluded = {Path(row["image"]).name for row in excluded_rows}
    with zipfile.ZipFile(args.archive) as archive:
        ids = [Path(line.strip()).name for line in archive.read("coco/val2017.txt").decode().splitlines()]
        if args.train_source == "train2017":
            train_ids = [Path(line.strip()).name for line in archive.read("coco/train2017.txt").decode().splitlines()]
            splits = select_trainval_splits(train_ids, ids, excluded, args.counts, args.seed)
        else:
            splits = select_splits(ids, excluded, args.counts, args.seed)
        sources = {"train": args.train_source, "dev": "val2017", "test": "val2017"}
        output.mkdir(parents=True)
        (output / "preparation.py").write_bytes(Path(__file__).read_bytes())
        archive_names = set(archive.namelist())
        for split, names in splits.items():
            source = sources[split]
            (output / "images" / source).mkdir(parents=True, exist_ok=True)
            (output / "labels" / source).mkdir(parents=True, exist_ok=True)
            for name in names:
                label = f"coco/labels/{source}/{Path(name).stem}.txt"
                content = archive.read(label) if label in archive_names else b""
                (output / "labels" / source / Path(name).with_suffix(".txt")).write_bytes(content)

    def download(item):
        name, source = item
        url = f"https://s3.amazonaws.com/images.cocodataset.org/{source}/{name}"
        path = output / "images" / source / name
        error = None
        for _ in range(3):
            try:
                urllib.request.urlretrieve(url, path)
                if cv2.imread(str(path)) is None:
                    raise ValueError(f"invalid image: {name}")
                return url
            except (OSError, ValueError) as exc:
                error = exc
        raise RuntimeError(f"download failed: {name}") from error

    items = [(name, sources[split]) for split, split_names in splits.items() for name in split_names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        urls = dict(zip([name for name, _ in items], pool.map(download, items)))
    identities = {}
    for split, names in splits.items():
        rows = []
        for name in names:
            image = output / "images" / sources[split] / name
            label = output / "labels" / sources[split] / Path(name).with_suffix(".txt")
            rows.append(
                {
                    "image": str(image),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "label": str(label),
                    "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                    "url": urls[name],
                    "source": sources[split],
                }
            )
        assert_disjoint(rows, excluded_rows)
        for previous in identities.values():
            assert_disjoint(rows, previous)
        identities[split] = rows
        (output / f"{split}.txt").write_text("\n".join(row["image"] for row in rows) + "\n")
    classes = yaml.safe_load((ROOT / "ultralytics/cfg/datasets/coco.yaml").read_text())["names"]
    for split in ("dev", "test"):
        (output / f"{split}.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(output),
                    "train": str(output / "train.txt"),
                    "val": str(output / f"{split}.txt"),
                    "names": classes,
                }
            )
        )
    manifest = {
        "seed": args.seed,
        "sources": sources,
        "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": identities,
        "label_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "label_source": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
        "excluded_report_sha256": hashlib.sha256(args.exclude_report.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({key: len(value) for key, value in identities.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
