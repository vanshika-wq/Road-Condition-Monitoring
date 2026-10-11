import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.local_features = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.gradient_features = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.geometry_embedding = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x, return_embedding=False):
        local = self.local_features(x)

        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))

        gradients = torch.cat([dx.abs(), dy.abs()], dim=1)
        gradient_gate = self.gradient_features(gradients)

        geometry_input = torch.cat(
            [gradients[:, :x.shape[1]], gradients[:, x.shape[1]:]], dim=1
        )
        embedding = self.geometry_embedding(geometry_input)
        edges = embedding * gradient_gate

        output = x + self.fusion(torch.cat([local, edges], dim=1))

        if return_embedding:
            return output, embedding
        return output


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    model = GeometryBranch(channels=32)
    y, embedding = model(x, return_embedding=True)
    y.mean().backward()

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Geometry embedding shape:", tuple(embedding.shape))
    print("Output finite:", torch.isfinite(y).all().item())
    print("Input gradient finite:", x.grad is not None and torch.isfinite(x.grad).all().item())
    print("Parameters receive gradients:", all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))
