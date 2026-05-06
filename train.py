import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

from model import PatchSequenceHORN

parser = argparse.ArgumentParser(description='Patch Sequence HORN training')
parser.add_argument('--patch-size', type=int, default=5, help='size of each patch')
parser.add_argument('--stride', type=int, default=1, help='stride for sliding window')
parser.add_argument('--feature-dim', type=int, default=16, help='CNN feature dimension per patch')
parser.add_argument('--num-hidden', type=int, default=32, help='number of HORN units')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega-base', type=float, default=0.224, help='base frequency (2*pi/28)')
parser.add_argument('--gamma-base', type=float, default=0.01, help='base damping coefficient')
parser.add_argument('--heterogeneous', action='store_true', default=True, help='use heterogeneous oscillator parameters')

args = parser.parse_args()
print(args)

# Fix seed
torch.manual_seed(args.seed)

# Calculate number of timesteps from image size (28x28)
image_size = 28
num_timesteps = ((image_size - args.patch_size) // args.stride + 1) ** 2
print(f"Each image will be converted to {num_timesteps} patches (timesteps)")

# Oscillator parameters
omega_base = args.omega_base
gamma_base = args.gamma_base

if args.heterogeneous:
    omega_min = 0.5 * omega_base
    omega_max = 2.0 * omega_base
    omega = torch.rand(args.num_hidden) * (omega_max - omega_min) + omega_min
    
    gamma_min = 0.5 * gamma_base
    gamma_max = 2.0 * gamma_base
    gamma = torch.rand(args.num_hidden) * (gamma_max - gamma_min) + gamma_min
else:
    omega = torch.ones(args.num_hidden) * omega_base
    gamma = torch.ones(args.num_hidden) * gamma_base

# Load MNIST
train_set = torchvision.datasets.MNIST(root='data', train=True, 
                                       transform=torchvision.transforms.ToTensor(), 
                                       download=True)
test_set = torchvision.datasets.MNIST(root='data', train=False, 
                                      transform=torchvision.transforms.ToTensor(), 
                                      download=True)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Model
model = PatchSequenceHORN(
    patch_size=args.patch_size,
    stride=args.stride,
    feature_dim=args.feature_dim,
    num_hidden=args.num_hidden,
    num_output=10,
    h=args.h,
    alpha=args.alpha,
    omega=omega,
    gamma=gamma
)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# Output directory
if not os.path.exists('out'):
    os.makedirs('out')

log_file = open('out/log.txt', 'a')


def evaluate_model(data_loader, epoch=None, batch_idx=None):
    """Evaluate model on test/validation set"""
    model.eval()
    correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            # images: (batch, 1, 28, 28)
            output = model(images, record=False)
            loss = criterion(output['output'], labels)
            
            total_loss += loss.item()
            pred = output['output'].argmax(1)
            correct += (pred == labels).sum().item()
    
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy, total_loss / len(data_loader)


def plot_dynamics(images, labels, epoch, batch_idx):
    """Plot oscillator dynamics for first batch sample"""
    model.eval()
    with torch.no_grad():
        output = model(images[:1], record=True)
    
    x_t = output['rec_x_t'][0]  # (timesteps, num_hidden)
    num_timesteps = output['num_timesteps']
    
    plt.figure(figsize=(12, 6))
    for i in range(min(16, args.num_hidden)):
        plt.plot(range(num_timesteps), x_t[:, i], alpha=0.7)
    
    plt.xlabel('timestep (patch position)')
    plt.ylabel('amplitude (x_t)')
    plt.title(f'Patch Sequence HORN Dynamics - Epoch {epoch}, Batch {batch_idx}\nLabel: {labels[0].item()}')
    plt.tight_layout()
    plt.savefig(f'out/dynamics_epoch{epoch:02d}_batch{batch_idx:04d}.png')
    plt.close()


# Training loop
best_accuracy = 0.0
print("\nStarting training...")
print(f"Each image = {num_timesteps} patches (timesteps)")
print(f"Hidden units: {args.num_hidden}")
print(f"Feature dimension per patch: {args.feature_dim}")
print("-" * 50)

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in pbar:
        # images: (batch, 1, 28, 28)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output['output'], labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Plot dynamics every 200 batches
        if batch_idx % 200 == 0 and batch_idx > 0:
            plot_dynamics(images, labels, epoch, batch_idx)
    
    # Evaluate
    train_acc, train_loss = evaluate_model(train_loader)
    test_acc, test_loss = evaluate_model(test_loader)
    
    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Log
    log_file.write(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%\n")
    log_file.flush()
    
    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), 'out/best_model.pt')
        print(f"  *** New best model! Saved to out/best_model.pt ***")
    
    # Save checkpoint
    torch.save(model.state_dict(), f'out/epoch{epoch:02d}.pt')

print(f"\nTraining complete! Best test accuracy: {best_accuracy:.2f}%")
log_file.write(f"Best test accuracy: {best_accuracy:.2f}%\n")
log_file.close()