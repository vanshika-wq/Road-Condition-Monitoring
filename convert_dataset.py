
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(r"C:\Users\juhis\OneDrive\Desktop\capstone\RDD2022\India\India")
TRAIN = ROOT / "train"
TEST = ROOT / "test"

xml_files = list((TRAIN / "annotations" / "xmls").glob("*.xml"))
print("XML files found:", len(xml_files))

if not xml_files:
    print("No XML files found. Check the extracted dataset folder.")
    raise SystemExit(1)

classes = sorted({
    obj.findtext("name")
    for xml in xml_files
    for obj in ET.parse(xml).getroot().findall("object")
    if obj.findtext("name")
})

if not classes:
    print("XML files found, but no object class names were detected.")
    raise SystemExit(1)

labels = ROOT / "labels" / "train"
labels.mkdir(parents=True, exist_ok=True)

for xml in xml_files:
    root = ET.parse(xml).getroot()
    size = root.find("size")

    if size is None:
        continue

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    rows = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")

        if name not in classes or box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        x = ((xmin + xmax) / 2) / width
        y = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        if w > 0 and h > 0:
            rows.append(
                f"{classes.index(name)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

    (labels / f"{xml.stem}.txt").write_text(
        "\n".join(rows), encoding="utf-8"
    )

config = {
    "path": str(ROOT),
    "train": "train/images",
    "val": "train/images",
    "test": "test/images",
    "names": {i: name for i, name in enumerate(classes)},
}

(ROOT / "data.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8"
)

print("Conversion complete.")
print("Classes:", classes)
print("Labels created:", len(list(labels.glob("*.txt"))))
print("Config:", ROOT / "data.yaml")
