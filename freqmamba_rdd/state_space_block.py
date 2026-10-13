import torch
import torch.nn as nn


class StateSpaceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.LayerNorm(channels)
        self.input_proj = nn.Linear(channels, channels * 2)
        self.state_proj = nn.Linear(channels, channels)
        self.geometry_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Linear(channels, channels)

        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

    def forward(self, x, geometry=None):
        residual = x
        b, c, h, w = x.shape

        if geometry is None:
            geometry = torch.zeros_like(x)
        elif geometry.shape[-2:] != (h, w):
            geometry = torch.nn.functional.interpolate(
                geometry, size=(h, w), mode="bilinear", align_corners=False
            )

        geometry = self.geometry_proj(geometry)
        x = x + geometry

        sequence = x.flatten(2).transpose(1, 2)
        sequence = self.norm(sequence)

        values, gate = self.input_proj(sequence).chunk(2, dim=-1)
        values = self.state_proj(values)
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        values = values * torch.sigmoid(gate)
        values = self.output_proj(values)

        output = values.transpose(1, 2).reshape(b, c, h, w)
        return output + residual


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    geometry = torch.randn(2, 32, 16, 16, requires_grad=True)
    model = StateSpaceBlock(32)

    y = model(x, geometry)
    y.mean().backward()

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Output finite:", torch.isfinite(y).all().item())
    print("Input gradient finite:", x.grad is not None and torch.isfinite(x.grad).all().item())
    print("Geometry gradient finite:", geometry.grad is not None and torch.isfinite(geometry.grad).all().item())
    print("Parameter gradients:", all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))
