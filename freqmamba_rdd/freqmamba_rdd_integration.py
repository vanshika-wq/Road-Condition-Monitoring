
import torch
import torch.nn as nn
from ultralytics import YOLO
from .freqmamba_rdd import FreqMambaRDD

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("yolo11n.pt").model.to(device).eval()

captured = {}

def capture_layer4(module, inputs, output):
    captured["features"] = output

hook = yolo.model[4].register_forward_hook(capture_layer4)

with torch.no_grad():
    yolo(torch.zeros(1, 3, 640, 640, device=device))

hook.remove()

features = captured["features"].detach()
print("YOLO11n Layer 4:", tuple(features.shape))

adapter_in = nn.Conv2d(128, 32, kernel_size=1).to(device)
adapter_out = nn.Conv2d(32, 128, kernel_size=1).to(device)
freqmamba = FreqMambaRDD(channels=32).to(device).eval()

try:
    reduced = adapter_in(features)
    enhanced = freqmamba(reduced)
    restored = adapter_out(enhanced)

    print("Reduced features:", tuple(reduced.shape))
    print("Enhanced features:", tuple(enhanced.shape))
    print("Restored features:", tuple(restored.shape))
    print("Output finite:", torch.isfinite(restored).all().item())

    assert restored.shape == features.shape
    assert torch.isfinite(restored).all()

    print("FreqMambaRDD integration test: PASS")

except Exception as e:
    print("Integration test failed:")
    print(type(e).__name__ + ":", e)
