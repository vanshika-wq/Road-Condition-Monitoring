
import torch
import torch.nn as nn
from ultralytics import YOLO

from freqmamba_rdd.freqmamba_rdd import FreqMambaRDD

WEIGHTS = r"baseline\best_9class.pt"
DEVICE = "cpu"


class FreqMambaAdapter(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=32):
        super().__init__()

        self.reduce = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1
        )
        self.freqmamba = FreqMambaRDD(hidden_channels)
        self.restore = nn.Conv2d(
            hidden_channels, in_channels, kernel_size=1
        )

    def forward(self, x):
        enhanced = self.reduce(x)
        enhanced = self.freqmamba(enhanced)
        enhanced = self.restore(enhanced)

        return x + enhanced


class IntegratedBlock(nn.Module):
    def __init__(self, original_block, adapter):
        super().__init__()

        self.original_block = original_block
        self.adapter = adapter

        self.i = original_block.i
        self.f = original_block.f
        self.type = original_block.type
        self.np = original_block.np

    def forward(self, x):
        x = self.original_block(x)
        x = self.adapter(x)
        return x


def main():
    torch.manual_seed(42)

    print("Loading 9-class YOLO model...")
    yolo = YOLO(WEIGHTS)
    detector = yolo.model.to(DEVICE)

    original_block = detector.model[4]

    adapter = FreqMambaAdapter(
        in_channels=128,
        hidden_channels=32
    ).to(DEVICE)

    detector.model[4] = IntegratedBlock(
        original_block,
        adapter
    )

    detector.eval()

    print("Classes:", len(detector.names))
    print(
        "Adapter registered:",
        isinstance(detector.model[4].adapter, FreqMambaAdapter)
    )

    adapter_parameters = sum(
        p.numel()
        for p in detector.model[4].adapter.parameters()
    )
    print("Adapter parameters:", adapter_parameters)

    x = torch.zeros(
        1, 3, 640, 640, device=DEVICE
    )

    try:
        with torch.no_grad():
            result = detector(x)

        print("Forward pass: PASS")
        print("Detection output produced:", result is not None)

    except Exception as error:
        print("Forward pass: FAIL")
        print(type(error).__name__ + ":", error)
        raise


if __name__ == "__main__":
    main()
