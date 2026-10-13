
from pathlib import Path
import random
import shutil
import yaml

ROOT = Path(r"C:\Users\juhis\OneDrive\Desktop\capstone\RDD2022\India\India")
SOURCE_IMAGES = ROOT / "train" / "images"
SOURCE_LABELS = ROOT / "labels" / "train"

random.seed(42)

images = [
    p for p in SOURCE_IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

if not images:
    raise RuntimeError(f"No images found in {SOURCE_IMAGES}")

valid = [p for p in images if (SOURCE_LABELS / f"{p.stem}.txt").exists()]
if len(valid) < 2:
    raise RuntimeError("Not enough matching images and labels.")

random.shuffle(valid)
split = max(1, int(len(valid) * 0.15))
val_images = valid[:split]
train_images = valid[split:]

for split_name, files in [("train", train_images), ("val", val_images)]:
    img_out = ROOT / "yolo" / "images" / split_name
    lbl_out = ROOT / "yolo" / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in files:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(SOURCE_LABELS / f"{img.stem}.txt",
                     lbl_out / f"{img.stem}.txt")

old_yaml = ROOT / "data.yaml"
config = yaml.safe_load(old_yaml.read_text(encoding="utf-8"))

config["path"] = str(ROOT / "yolo")
config["train"] = "images/train"
config["val"] = "images/val"
config.pop("test", None)

(ROOT / "yolo" / "data.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8"
)

print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Dataset config:", ROOT / "yolo" / "data.yaml")
print("Original dataset preserved.")
