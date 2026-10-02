import torch
import torch.nn as nn


class DeformableAttention(nn.Module):
    """Simplified deformable attention: learns where to sample instead of
    looking at the whole feature map uniformly."""

    def __init__(self, dim, n_heads=4, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads

        self.offset_proj = nn.Conv2d(dim, n_heads * n_points * 2, kernel_size=3, padding=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        value = self.value_proj(x)
        offsets = self.offset_proj(x)  # predicted sampling offsets

        # Build a base sampling grid (normalized to [-1, 1])
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([xs, ys], dim=-1)  # H, W, 2

        out = torch.zeros_like(value)
        offsets = offsets.view(B, self.n_heads, self.n_points, 2, H, W)

        for h in range(self.n_heads):
            head_val = value[:, h * self.head_dim:(h + 1) * self.head_dim]
            acc = torch.zeros_like(head_val)
            for p in range(self.n_points):
                off = offsets[:, h, p].permute(0, 2, 3, 1) * 0.1  # scale offsets
                grid = (base_grid.unsqueeze(0) + off).clamp(-1, 1)
                sampled = nn.functional.grid_sample(
                    head_val, grid, align_corners=True, mode="bilinear"
                )
                acc = acc + sampled
            out[:, h * self.head_dim:(h + 1) * self.head_dim] = acc / self.n_points

        return self.output_proj(out)


class BottleneckDA(nn.Module):
    """Conv -> Deformable Attention -> Conv, with a residual connection.
    Same shape as Fig. 2 of the paper."""

    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.attn = DeformableAttention(c2)
        self.cv2 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y = self.attn(y)
        y = self.cv2(y)
        return x + y if self.add else y


class C2f_DAttn(nn.Module):
    """Same idea as Ultralytics' C2f, but its bottleneck blocks use
    Deformable Attention instead of plain convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, kernel_size=1)
        self.m = nn.ModuleList(
            BottleneckDA(self.c, self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    block = BottleneckDA(64, 64)
    dummy = torch.randn(1, 64, 20, 20)  # batch=1, channels=64, 20x20 feature map
    out = block(dummy)
    print("input shape :", dummy.shape)
    print("output shape:", out.shape)

    c2f = C2f_DAttn(64, 64, n=2)
    dummy2 = torch.randn(1, 64, 20, 20)
    out2 = c2f(dummy2)
    print("C2f_DAttn input :", dummy2.shape)
    print("C2f_DAttn output:", out2.shape)