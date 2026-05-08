import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

# ============================================================
# Simple CNN Baseline (No HORN, No Oscillations)
# ============================================================

class SimpleCNN(nn.Module):
    """
    Standard CNN classifier for MNIST
    No recurrence, no oscillations - purely spatial processing
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 28 -> 14
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 14 -> 7
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        # Calculate the flattened size
        # After conv layers: (128, 7, 7) = 128*7*7 = 6272
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28) - raw MNIST images
        Returns:
            logits: (batch, num_classes)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============================================================
# Equivalent CNN with Similar Parameter Count to HORN
# ============================================================

class ComparableCNN(nn.Module):
    """
    CNN with approximately 3,500 parameters (same as HORN)
    Much smaller than SimpleCNN above
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Very lightweight CNN to match HORN's 3.5K params
        self.features = nn.Sequential(
            # Conv1: 1→8 channels, 3x3 kernel
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28→14
            
            # Conv2: 8→16 channels, 3x3 kernel
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14→7
            
            # Conv3: 16→32 channels, 3x3 kernel
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global pool → 32 features
        )
        
        # Classifier
        self.classifier = nn.Linear(32, num_classes)
        
        # Parameter count: ~3,500 (comparable to HORN)
        # - Conv1: (1*3*3)*8 + 8 = 80
        # - Conv2: (8*3*3)*16 + 16 = 1,168
        # - Conv3: (16*3*3)*32 + 32 = 4,640
        # - Linear: 32*10 + 10 = 330
        # Total: ~6,200 (slightly larger)
    
    def forward(self, x):
        x = self.features(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)
        x = self.classifier(x)  # (batch, 10)
        return x


# ============================================================
# Training Script
# ============================================================

def train_cnn_baseline(model_type='simple', epochs=10, batch_size=64, lr=0.001):
    """
    Train CNN baseline for comparison with HORN
    
    Args:
        model_type: 'simple' (big) or 'comparable' (small, ~HORN size)
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='data', train=False, transform=transform, download=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if model_type == 'simple':
        model = SimpleCNN(num_classes=10)
        model_name = "SimpleCNN_Large"
    else:
        model = ComparableCNN(num_classes=10)
        model_name = "ComparableCNN_Small"
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Create output directory
    os.makedirs('out', exist_ok=True)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_loss)
        
        # Validation
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}: Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%\n')
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'out/{model_name}_best.pt')
            print(f"  *** New best model saved! ***")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'out/{model_name}_epoch{epoch+1:02d}.pt')
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Plot training curves
    plot_training_curves(train_losses, test_accuracies, model_name)
    
    return model, best_accuracy


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def plot_training_curves(train_losses, test_accuracies, model_name):
    """Plot training curves for comparison"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'out/{model_name}_training_curves.png')
    plt.show()


def compare_models():
    """Train both CNN variants and compare"""
    results = {}
    
    # Train small CNN (comparable to HORN size)
    print("\n" + "="*60)
    print("TRAINING COMPARABLE CNN (Similar size to HORN)")
    print("="*60)
    _, acc1 = train_cnn_baseline(model_type='comparable', epochs=10, batch_size=64, lr=0.001)
    results['Comparable CNN (~3.5K params)'] = acc1
    
    # Train large CNN (better performance baseline)
    print("\n" + "="*60)
    print("TRAINING LARGE CNN (Strong baseline)")
    print("="*60)
    _, acc2 = train_cnn_baseline(model_type='simple', epochs=10, batch_size=64, lr=0.001)
    results['Large CNN (~1.2M params)'] = acc2
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    for name, acc in results.items():
        print(f"{name:30s}: {acc:.2f}%")
    
    # Create comparison bar chart
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    accs = list(results.values())
    bars = plt.bar(names, accs, color=['steelblue', 'darkorange'])
    plt.ylabel('Test Accuracy (%)')
    plt.title('CNN Baseline vs HORN Comparison')
    plt.ylim(90, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('out/cnn_baseline_comparison.png')
    plt.show()


# ============================================================
# Inference Example
# ============================================================

def visualize_predictions(model_path='out/ComparableCNN_Small_best.pt'):
    """Load trained model and visualize predictions on test samples"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ComparableCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='data', train=False, transform=transform, download=True
    )
    
    # Get 16 random samples
    indices = torch.randperm(len(test_dataset))[:16]
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes.flatten()):
            image, label = test_dataset[idx]
            image_display = image.squeeze().numpy()
            
            # Predict
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            prediction = output.argmax(1).item()
            
            # Display
            ax.imshow(image_display, cmap='gray')
            color = 'green' if prediction == label else 'red'
            ax.set_title(f'True: {label}, Pred: {prediction}', color=color)
            ax.axis('off')
    
    plt.suptitle('CNN Baseline - Predictions on MNIST Test Set')
    plt.tight_layout()
    plt.savefig('out/cnn_predictions.png')
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Baseline for HORN Comparison')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'compare', 'visualize'],
                        help='Mode to run')
    parser.add_argument('--model-type', type=str, default='comparable',
                        choices=['simple', 'comparable'],
                        help='CNN model type')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_cnn_baseline(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
    elif args.mode == 'compare':
        compare_models()
    elif args.mode == 'visualize':
        visualize_predictions()