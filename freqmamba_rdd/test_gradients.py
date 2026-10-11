import torch
from freqmamba_rdd import FreqMambaRDD

model = FreqMambaRDD(channels=32)
x = torch.randn(2, 32, 64, 64, requires_grad=True)

output = model(x)
loss = output.mean()
loss.backward()

print("Output shape:", output.shape)
print("Input gradient exists:", x.grad is not None)
print("Input gradient finite:", torch.isfinite(x.grad).all().item())
print(
    "Parameters with gradients:",
    sum(p.grad is not None for p in model.parameters()),
    "/",
    sum(1 for p in model.parameters())
)
