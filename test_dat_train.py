from ultralytics import YOLO

model = YOLO("yolov8-dat.yaml")

model.train(
    data="coco8.yaml",   # tiny built-in sample dataset, auto-downloads (~1 MB)
    epochs=1,
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0,
    plots=False,
)