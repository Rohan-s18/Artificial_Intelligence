import torch

# defining the device
mps_device = torch.device("mps")

# Create a Tensor directly on the mps device
x = torch.ones(5, device=mps_device)

x = torch.ones(5, device="mps")

# Any operation happens on the GPU
y = x * 2

print(x.device)
print(y.device)
