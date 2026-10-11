import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt").model
model.eval()

captured = {}
target_layers = {4, 6, 9, 12, 15, 18, 21}

def make_hook(index):
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            captured[index] = tuple(output.shape)
        elif isinstance(output, (list, tuple)):
            captured[index] = [
                tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__
                for item in output
            ]
    return hook

hooks = []
for index, layer in enumerate(model.model):
    if index in target_layers:
        hooks.append(layer.register_forward_hook(make_hook(index)))

with torch.no_grad():
    model(torch.zeros(1, 3, 640, 640))

for index in sorted(captured):
    print(f"Layer {index}: {captured[index]}")

for hook in hooks:
    hook.remove()
