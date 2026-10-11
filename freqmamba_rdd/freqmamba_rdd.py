import torch
import torch.nn as nn

from .frequency_branch import FrequencyBranch
from .geometry_branch import GeometryBranch
from .state_space_block import StateSpaceBlock
from .adaptive_fusion import AdaptiveFusion

class FreqMambaRDD(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.frequency_branch = FrequencyBranch(channels)
        self.geometry_branch = GeometryBranch(channels)
        self.state_space_block = StateSpaceBlock(channels)
        self.fusion = AdaptiveFusion(channels)

    def forward(self, x):
        frequency_features = self.frequency_branch(x)

        geometry_features, geometry_embedding = self.geometry_branch(
            x, return_embedding=True
        )

        state_features = self.state_space_block(
            geometry_features, geometry_embedding
        )

        output = self.fusion(
            x,
            frequency_features,
            state_features
        )

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    model = FreqMambaRDD(channels=32)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Output finite:", torch.isfinite(output).all().item())
    print("Input gradient finite:", x.grad is not None and torch.isfinite(x.grad).all().item())
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Parameter gradients:", all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.requires_grad
    ))
