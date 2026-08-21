import xml.etree.ElementTree as ET
from pathlib import Path

VOC_ROOT = Path("/Users/gatilin/MyWork/datasets/voc/VOC0712")
VOC2007 = VOC_ROOT / "VOCdevkit" / "VOC2007"

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def voc2yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    labels = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in VOC_CLASSES:
            continue
        cls_id = VOC_CLASSES.index(cls)

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)

        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    return labels


def convert_split(split_name):
    imgset_file = VOC2007 / "ImageSets" / "Main" / f"{split_name}.txt"
    if not imgset_file.exists():
        return 0

    img_ids = imgset_file.read_text().strip().split("\n")
    labels_dir = VOC2007 / "labels" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id in img_ids:
        img_id = img_id.strip()
        if not img_id:
            continue
        xml_path = VOC2007 / "Annotations" / f"{img_id}.xml"
        if not xml_path.exists():
            continue

        labels = voc2yolo(xml_path)
        if labels:
            (labels_dir / f"{img_id}.txt").write_text("\n".join(labels))
            count += 1

    return count


def make_list_file(split_name):
    imgset_file = VOC2007 / "ImageSets" / "Main" / f"{split_name}.txt"
    if not imgset_file.exists():
        return 0
    img_ids = imgset_file.read_text().strip().split("\n")
    img_paths = []
    for img_id in img_ids:
        img_id = img_id.strip()
        if not img_id:
            continue
        img_path = VOC2007 / "JPEGImages" / f"{img_id}.jpg"
        if img_path.exists():
            img_paths.append(str(img_path))

    list_file = VOC2007 / f"{split_name}.txt"
    list_file.write_text("\n".join(img_paths))
    return len(img_paths)


def main(ctx):
    for split in ["train", "val", "trainval", "test"]:
        n = convert_split(split)
        print(f"{split}: {n} images converted to YOLO labels")

    for split in ["train", "val", "trainval", "test"]:
        n = make_list_file(split)
        print(f"{split}.txt: {n} image paths")

    yaml_content = f"path: {VOC2007}\ntrain: trainval.txt\nval: test.txt\nnc: 20\nnames: {VOC_CLASSES}\n"
    yaml_path = VOC2007 / "voc0712.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Created: {yaml_path}")

    n_jpg = len(list((VOC2007 / "JPEGImages").glob("*.jpg")))
    print(f"VOC2007 total: {n_jpg} images")
    return {"status": "ok", "images": n_jpg, "yaml": str(yaml_path)}
