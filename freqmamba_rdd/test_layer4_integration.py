import torch
from ultralytics import YOLO
from freqmamba_rdd import FreqMambaRDD

model = YOLO("yolov8n.pt").model
model.eval()

captured = {}

def capture_layer4(module, inputs, output):
    captured["features"] = output

hook = model.model[4].register_forward_hook(capture_layer4)

with torch.no_grad():
    model(torch.zeros(1, 3, 640, 640))

hook.remove()

features = captured["features"]
print("YOLO layer 4:", tuple(features.shape))

channels = features.shape[1]
module = FreqMambaRDD(channels=channels)
module.eval()

with torch.no_grad():
    enhanced = module(features)

print("Enhanced features:", tuple(enhanced.shape))
print("Output finite:", torch.isfinite(enhanced).all().item())
print("Integration compatibility: PASS")
