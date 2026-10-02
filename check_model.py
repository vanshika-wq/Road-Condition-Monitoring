from ultralytics import YOLO

model = YOLO("baseline/best.pt")
print(model.names)