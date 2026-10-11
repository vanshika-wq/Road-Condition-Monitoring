
import torch
import torch.nn as nn
from ultralytics import YOLO

from freqmamba_rdd.freqmamba_rdd import FreqMambaRDD

WEIGHTS = r"baseline\best_9class.pt"
DEVICE = "cpu"


class FreqMambaAdapter(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=32):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, hidden_channels, 1)
        self.freqmamba = FreqMambaRDD(hidden_channels)
        self.restore = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x):
        reduced = self.reduce(x)
        enhanced = self.freqmamba(reduced)
        return x + self.restore(enhanced)


def main():
    torch.manual_seed(42)

    detector = YOLO(WEIGHTS).model.to(DEVICE).eval()
    print("Classes:", len(detector.names))

    captured = {}
    adapter = None

    def integrate(module, inputs, output):
        nonlocal adapter

        if not isinstance(output, torch.Tensor):
            raise TypeError("Layer 4 output is not a tensor")

        if adapter is None:
            adapter = FreqMambaAdapter(
                in_channels=output.shape[1]
            ).to(output.device).eval()

        captured["input_shape"] = tuple(output.shape)
        result = adapter(output)
        captured["output_shape"] = tuple(result.shape)

        return result

    hook = detector.model[4].register_forward_hook(integrate)

    try:
        with torch.no_grad():
            result = detector(
                torch.zeros(1, 3, 640, 640, device=DEVICE)
            )

        print("Layer 4 input:", captured["input_shape"])
        print("Enhanced output:", captured["output_shape"])
        print("Detector forward pass: PASS")
        print("Detection output produced:", result is not None)

    finally:
        hook.remove()


if __name__ == "__main__":
    main()
