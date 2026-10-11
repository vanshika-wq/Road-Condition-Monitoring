
from ultralytics import YOLO
from collections import Counter
import os
import time

print("=" * 48)
print("       ROAD DAMAGE DETECTION SYSTEM")
print("=" * 48)

model = YOLO("baseline/best.pt")

image_path = input(
    "Enter the path of the road image to analyze: "
).strip().strip('"')

start_time = time.time()

result = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=True,
    project="runs",
    name="live_demo",
    exist_ok=True
)[0]

inference_time = time.time() - start_time

print("\n" + "=" * 48)
print("             DETECTION REPORT")
print("=" * 48)

print("Image analyzed:", os.path.basename(result.path))
print("Total detections:", len(result.boxes))

counts = Counter(
    result.names[int(box.cls[0])]
    for box in result.boxes
)

print("\nClass-wise Summary:")
if counts:
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
else:
    print("No road damage detected.")

print("\nDetection Details:")
for box in result.boxes:
    class_name = result.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"- {class_name}: {confidence:.2%} confidence")

print(f"\nProcessing time: {inference_time:.2f} seconds")

saved_image = os.path.abspath(
    os.path.join(
        str(result.save_dir),
        os.path.basename(result.path)
    )
)

print("\nPrediction image saved at:")
print(saved_image)
print("=" * 48)

os.startfile(saved_image)
