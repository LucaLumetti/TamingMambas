import torch
from mamba_ssm import Mamba, Mamba2

batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")

# Initialize the Mamba model
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

# Initialize the Mamba2 model
model2 = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    headdim=32,  # Additional parameter for Mamba2
    ngroups=1,   # Number of groups for group normalization
    sequence_parallel=False, # Whether to use sequence parallelism
).to("cuda")

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Print the number of parameters for each model
print(f"Mamba model parameters: {count_parameters(model)}")
print(f"Mamba2 model parameters: {count_parameters(model2)}")

# Measure inference time for Mamba model
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
y = model(x)
end_event.record()

# Wait for all CUDA operations to finish
torch.cuda.synchronize()

mamba_time = start_event.elapsed_time(end_event) # Time in milliseconds

print(f"\nMamba model time: {mamba_time} ms")
print(y.shape)
assert y.shape == x.shape

# Measure inference time for Mamba2 model
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
y = model2(x)
end_event.record()

# Wait for all CUDA operations to finish
torch.cuda.synchronize()

mamba2_time = start_event.elapsed_time(end_event) # Time in milliseconds

print(f"\nMamba2 model time: {mamba2_time} ms")
print(y.shape)
assert y.shape == x.shape
