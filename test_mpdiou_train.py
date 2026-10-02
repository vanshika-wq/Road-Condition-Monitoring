from ultralytics import YOLO

model = YOLO("yolov8-dat-slimneck.yaml")

model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0,
    plots=False,
)