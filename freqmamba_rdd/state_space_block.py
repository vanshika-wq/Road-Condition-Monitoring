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
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

    def forward(self, x, geometry_embedding=None):
        residual = x
        b, c, h, w = x.shape

        sequence = x.flatten(2).transpose(1, 2)
        sequence = self.norm(sequence)

        values, gate = self.input_proj(sequence).chunk(2, dim=-1)
        values = self.state_proj(values)

        if geometry_embedding is not None:
            geometry = self.geometry_proj(geometry_embedding)
            geometry = geometry.flatten(2).transpose(1, 2)
            gate = gate + geometry

        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        values = values * torch.sigmoid(gate)
        values = self.output_proj(values)

        output = values.transpose(1, 2).reshape(b, c, h, w)

        return output + residual


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    embedding = torch.randn(2, 32, 64, 64)

    model = StateSpaceBlock(channels=32)
    y = model(x, embedding)
    y.mean().backward()

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Output finite:", torch.isfinite(y).all().item())
    print("Input gradients finite:", x.grad is not None and torch.isfinite(x.grad).all().item())
    print(
        "Parameter gradients finite:",
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        )
    )