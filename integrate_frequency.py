
import torch
import torch.nn as nn
from ultralytics import YOLO

from freqmamba_rdd.frequency_branch import FrequencyBranch


WEIGHTS = r"baseline\best_9class.pt"
DEVICE = "cpu"


class Stage1Frequency(nn.Module):
    def __init__(self, channels=128, reduced_channels=32):
        super().__init__()

        self.reduce = nn.Conv2d(channels, reduced_channels, 1)
        self.frequency = FrequencyBranch(reduced_channels)
        self.restore = nn.Conv2d(reduced_channels, channels, 1)

        nn.init.normal_(self.restore.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.restore.bias)

    def forward(self, x):
        enhanced = self.frequency(self.reduce(x))
        return x + self.restore(enhanced)


def main():
    torch.manual_seed(42)

    yolo = YOLO(WEIGHTS)
    model = yolo.model.to(DEVICE).eval()

    captured = {}

    def inspect_layer(module, inputs, output):
        if isinstance(output, torch.Tensor):
            captured["shape"] = tuple(output.shape)

    hook = model.model[4].register_forward_hook(inspect_layer)

    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640, device=DEVICE))

    hook.remove()

    if "shape" not in captured:
        raise RuntimeError("Layer 4 output was not captured.")

    shape = captured["shape"]
    channels = shape[1]

    print("Original Layer 4:", shape)

    model.freq_stage1 = Stage1Frequency(channels).to(DEVICE).eval()

    def apply_frequency(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Layer 4 output is not a tensor.")
        return model.freq_stage1(output)

    hook = model.model[4].register_forward_hook(apply_frequency)

    try:
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 640, 640, device=DEVICE))

        print("Full detector forward pass: PASS")
        print("Frequency branch input:", shape)
        print("Frequency branch output:", captured["shape"])
        print("Detection output produced:", result is not None)

    finally:
        hook.remove()


if __name__ == "__main__":
    main()
