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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs=3, type=int, default=(64, 32, 64))
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
        splits = select_splits(ids, excluded, args.counts, args.seed)
        output.mkdir(parents=True)
        (output / "preparation.py").write_bytes(Path(__file__).read_bytes())
        (output / "images/val2017").mkdir(parents=True)
        (output / "labels/val2017").mkdir(parents=True)
        for name in [name for split_names in splits.values() for name in split_names]:
            label = f"coco/labels/val2017/{Path(name).stem}.txt"
            # The published split includes background images with no label file.
            content = archive.read(label) if label in archive.namelist() else b""
            (output / "labels/val2017" / Path(name).with_suffix(".txt")).write_bytes(content)

    def download(name):
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}"
        path = output / "images/val2017" / name
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

    names = [name for split_names in splits.values() for name in split_names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        urls = dict(zip(names, pool.map(download, names)))
    identities = {}
    for split, names in splits.items():
        rows = []
        for name in names:
            image = output / "images/val2017" / name
            label = output / "labels/val2017" / Path(name).with_suffix(".txt")
            rows.append(
                {
                    "image": str(image),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "label": str(label),
                    "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                    "url": urls[name],
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
