import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        haar = torch.tensor([
            [[1, 1], [1, 1]],
            [[-1, -1], [1, 1]],
            [[-1, 1], [-1, 1]],
            [[1, -1], [-1, 1]]
        ], dtype=torch.float32) * 0.5
        self.register_buffer("filters", haar.unsqueeze(1))

    def forward(self, x):
        channels = x.shape[1]
        filters = self.filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
        coeffs = F.conv2d(x, filters, stride=2, groups=channels)
        b, _, h, w = coeffs.shape
        coeffs = coeffs.reshape(b, channels, 4, h, w)
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]


class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = HaarDWT()
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 1),
            nn.Sigmoid()
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        low = self.low_proj(ll)

        details = torch.cat([lh, hl, hh], dim=1)
        gate = self.detail_gate(details)
        high = self.high_proj(details * gate)

        size = x.shape[-2:]
        low = F.interpolate(low, size=size, mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=size, mode="bilinear", align_corners=False)

        enhanced = self.fusion(torch.cat([low, high], dim=1))
        return x + enhanced


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    model = FrequencyBranch(channels=32)
    y = model(x)
    loss = y.mean()
    loss.backward()

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Output finite:", torch.isfinite(y).all().item())
    print("Input gradient finite:", x.grad is not None and torch.isfinite(x.grad).all().item())
    print("Parameters receive gradients:", all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))


