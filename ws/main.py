import torch
import math

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(-x)

print(y)