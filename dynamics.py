import torch
import matplotlib.pyplot as plt

from model import PatchSequenceHORN

torch.manual_seed(1)
torch.set_grad_enabled(False)

# Parameters
patch_size = 5
stride = 1  # Overlapping patches
feature_dim = 16  # CNN output per patch
num_hidden = 32
num_output = 10

# Hyperparameters
h = 1.0
alpha = 0.04
omega_base = 2. * torch.pi / 28.
gamma_base = 0.01

# Heterogeneous oscillation parameters
num_patches = ((28 - patch_size) // stride + 1) ** 2  # Calculate number of timesteps
print(f"Patch size: {patch_size}, Stride: {stride}")
print(f"Number of timesteps (patches): {num_patches}")

omega_min = 0.5 * omega_base
omega_max = 2.0 * omega_base
omega = torch.rand(num_hidden) * (omega_max - omega_min) + omega_min

gamma_min = 0.5 * gamma_base
gamma_max = 2.0 * gamma_base
gamma = torch.rand(num_hidden) * (gamma_max - gamma_min) + gamma_min

# Construct model
model = PatchSequenceHORN(
    patch_size=patch_size,
    stride=stride,
    feature_dim=feature_dim,
    num_hidden=num_hidden,
    num_output=num_output,
    h=h,
    alpha=alpha,
    omega=omega,
    gamma=gamma
)
model.eval()

# Set weights (access internal HORN)
model.horn.i2h.weight[:, :] = torch.randn(num_hidden, feature_dim)
model.horn.i2h.bias[:] = 0
model.horn.h2h.weight[:, :] = torch.randn(num_hidden, num_hidden) * 0.1
model.horn.h2h.bias[:] = 0

# Show weight matrix
plt.figure(figsize=(10, 8))
plt.matshow(model.horn.h2h.weight.detach(), fignum=0)
plt.colorbar()
plt.title('W_hh (recurrent weights)')

# Create a test image (simple MNIST-like digit)
# For demo, create a synthetic image (circle or random)
test_image = torch.zeros(1, 1, 28, 28)
# Draw a simple pattern
for i in range(28):
    for j in range(28):
        if (i-14)**2 + (j-14)**2 < 50:  # Circle
            test_image[0, 0, i, j] = 1.0

# Or load a real MNIST sample (uncomment if you have torchvision)
# import torchvision
# mnist = torchvision.datasets.MNIST(root='data', train=True, download=True)
# test_image = mnist[0][0].unsqueeze(0)  # (1, 28, 28)

# Run model dynamics
random_init = 1.0
out = model.forward(test_image, random_init=random_init, record=True)

# Get recorded states
x_t = out['rec_x_t']  # (batch, timesteps, num_hidden)
num_timesteps = out['num_timesteps']

print(f"Output shape: {out['output'].shape}")
print(f"Recorded amplitudes shape: {x_t.shape}")

# Plot amplitude dynamics
plt.figure(figsize=(12, 6))
timesteps = range(num_timesteps)
for i in range(min(16, num_hidden)):  # Plot first 16 oscillators
    plt.plot(timesteps, x_t[0, :, i], alpha=0.7)

plt.xlabel('timestep (patch position)')
plt.ylabel('amplitude (x_t)')
plt.title(f'Patch Sequence HORN: {num_hidden} oscillators, {num_timesteps} patches')
plt.tight_layout()
plt.show()

# Also visualize the patch sequence to understand what the model sees
plt.figure(figsize=(12, 4))
# Show first few patches
for t in range(min(8, num_timesteps)):
    # Reconstruct patch position
    patch_h = t // ((28 - patch_size) // stride + 1)
    patch_w = t % ((28 - patch_size) // stride + 1)
    
    plt.subplot(2, 4, t+1)
    # Extract patch from original image
    patch = test_image[0, 0, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]
    plt.imshow(patch, cmap='gray')
    plt.title(f'Patch t={t}')
    plt.axis('off')

plt.suptitle('First 8 patches in the sequence (scanning left to right, top to bottom)')
plt.tight_layout()
plt.show()