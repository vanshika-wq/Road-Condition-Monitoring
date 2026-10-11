
import torch
from ultralytics import YOLO

torch.manual_seed(42)

model = YOLO("yolo11n.pt").model
model.eval()

captured = {}

def capture_layer4(module, inputs, output):
    captured["features"] = output

hook = model.model[4].register_forward_hook(capture_layer4)

x = torch.zeros(1, 3, 640, 640)

with torch.no_grad():
    model(x)

hook.remove()

features = captured["features"]

print("YOLO11n Layer 4:", tuple(features.shape))
print("Output finite:", torch.isfinite(features).all().item())

features = features.detach().requires_grad_(True)
loss = features.mean()
loss.backward()

print("Gradient exists:", features.grad is not None)
print(
    "Gradient finite:",
    features.grad is not None
    and torch.isfinite(features.grad).all().item()
)

assert features.shape == (1, 128, 80, 80)
assert torch.isfinite(features).all()
assert features.grad is not None
assert torch.isfinite(features.grad).all()

print("YOLO11n Layer 4 test: PASS")
