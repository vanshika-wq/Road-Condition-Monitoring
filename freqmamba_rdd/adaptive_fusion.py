import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.spatial_proj = nn.Conv2d(channels, channels, 1)
        self.frequency_proj = nn.Conv2d(channels, channels, 1)
        self.state_proj = nn.Conv2d(channels, channels, 1)

        self.weight_generator = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 1)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, spatial, frequency, state):
        size = spatial.shape[-2:]

        if frequency.shape[-2:] != size:
            frequency = F.interpolate(
                frequency, size=size, mode="bilinear", align_corners=False
            )
        if state.shape[-2:] != size:
            state = F.interpolate(
                state, size=size, mode="bilinear", align_corners=False
            )

        spatial = self.spatial_proj(spatial)
        frequency = self.frequency_proj(frequency)
        state = self.state_proj(state)

        combined = torch.cat([spatial, frequency, state], dim=1)
        weights = torch.softmax(self.weight_generator(combined), dim=1)

        fused = (
            spatial * weights[:, 0:1]
            + frequency * weights[:, 1:2]
            + state * weights[:, 2:3]
        )

        return spatial + self.fusion(fused)


if __name__ == "__main__":
    torch.manual_seed(42)

    spatial = torch.randn(2, 32, 32, 32, requires_grad=True)
    frequency = torch.randn(2, 32, 32, 32, requires_grad=True)
    state = torch.randn(2, 32, 32, 32, requires_grad=True)

    model = AdaptiveFusion(32)
    output = model(spatial, frequency, state)
    output.mean().backward()

    print("Spatial shape  :", tuple(spatial.shape))
    print("Frequency shape:", tuple(frequency.shape))
    print("State shape    :", tuple(state.shape))
    print("Output shape   :", tuple(output.shape))
    print("Output finite  :", torch.isfinite(output).all().item())
    print("Input gradients:", all(
        x.grad is not None and torch.isfinite(x.grad).all().item()
        for x in [spatial, frequency, state]
    ))
    print("Parameter gradients:", all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    ))
