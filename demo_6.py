from ultralytics import YOLO
import cv2, glob, os

model = YOLO(r"baseline\best_9class.pt")
files = glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.png", recursive=True)
shown = 0

for f in files:
    if any(x in f.lower() for x in ["runs", ".git", "site-packages"]):
        continue
    result = model.predict(f, imgsz=640, conf=0.10, verbose=False)[0]
    if len(result.boxes) > 0:
        print(f, "| Detections:", len(result.boxes))
        cv2.imshow("Road Damage Detection - " + os.path.basename(f), result.plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        shown += 1
        if shown == 6:
            break

print(f"Total images with detections: {shown}")
if shown == 0:
    print("Kisi image mein detection nahi mila.")
