import math
import torch
import torch.nn as nn

#horn
class HORN(nn.Module):
    """
    Harmonic Oscillator Recurrent Network (HORN)
    """
    
    def __init__(self, num_input, num_nodes, num_output, h, alpha, omega, gamma):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output

        self.h = h
        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma

        self.omega_factor = self.omega * self.omega
        self.gamma_factor = 2.0 * self.gamma
        self.gain_rec = 1. / math.sqrt(self.num_nodes)

        self.i2h = nn.Linear(num_input, num_nodes)
        self.h2h = nn.Linear(num_nodes, num_nodes)
        self.h2o = nn.Linear(num_nodes, num_output)

    def dynamics_step(self, x_t, y_t, input_t):
        y_t = y_t + self.h * (
            self.alpha * torch.tanh(
                self.i2h(input_t)
                + self.gain_rec * self.h2h(y_t)
            )
            - self.omega_factor * x_t
            - self.gamma_factor * y_t
        )
        x_t = x_t + self.h * y_t
        return x_t, y_t

    def forward(self, batch, random_init=None, record=False):
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}

        if record:
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret['rec_x_t'] = rec_x_t
            ret['rec_y_t'] = rec_y_t

        if random_init is not None:
            x_0 = torch.randn(batch_size, self.num_nodes) * random_init
            y_0 = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            x_0 = torch.zeros(batch_size, self.num_nodes)
            y_0 = torch.zeros(batch_size, self.num_nodes)

        x_t = torch.autograd.Variable(x_0)
        y_t = torch.autograd.Variable(y_0)

        for t in range(num_timesteps):
            x_t, y_t = self.dynamics_step(x_t, y_t, batch[t])

            if record:
                rec_x_t[:, t, :] = x_t
                rec_y_t[:, t, :] = y_t

        output = self.h2o(x_t)
        ret['output'] = output
        return ret

#process in patches with cnn

class PatchCNN(nn.Module):
    """
    CNN that processes each image patch independently
    Extracts features from local spatial structure at each timestep
    """
    
    def __init__(self, patch_size=5, in_channels=1, feature_dim=16):
        super().__init__()
        
        self.patch_size = patch_size
        
        # CNN for a single patch
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Project to feature_dim
        self.projection = nn.Linear(16, feature_dim)
    
    def forward(self, patch):
        """
        Args:
            patch: (batch, 1, patch_size, patch_size)
        Returns:
            features: (batch, feature_dim)
        """
        x = self.patch_cnn(patch)
        x = self.projection(x)
        return x

#combining horn with cnn
class PatchSequenceHORN(nn.Module):
    """
    Convert image to sequence of overlapping patches,
    process each patch through CNN,
    HORN through the patch sequence
    """
    
    def __init__(self, patch_size, stride, feature_dim, num_hidden, num_output, 
                 h, alpha, omega, gamma):
        super().__init__()
        
        self.patch_size = patch_size
        self.stride = stride
        
        # CNN for patch processing
        self.patch_cnn = PatchCNN(patch_size=patch_size, in_channels=1, feature_dim=feature_dim)
        
        # HORN for sequence processing
        self.horn = HORN(feature_dim, num_hidden, num_output, h, alpha, omega, gamma)
    
    def forward(self, image, random_init=None, record=False):
        
        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
        
        # Extract overlapping patches
        # Using unfold to create sliding windows
        patches_h = image.unfold(2, self.patch_size, self.stride)  # (batch, 1, num_h, width, patch_size)
        patches_h = patches_h.unfold(3, self.patch_size, self.stride)  # (batch, 1, num_h, num_w, patch_size, patch_size)
        
        # Get dimensions
        num_patches_h = patches_h.shape[2]
        num_patches_w = patches_h.shape[3]
        num_timesteps = num_patches_h * num_patches_w
        
        # Reshape to sequence: (timesteps, batch, 1, patch_size, patch_size)
        patches = patches_h.permute(2, 3, 0, 1, 4, 5).contiguous()
        patches = patches.view(num_timesteps, batch_size, 1, self.patch_size, self.patch_size)
        
        # Process each patch through CNN to get features
        features = []
        for t in range(num_timesteps):
            feat = self.patch_cnn(patches[t])  # (batch, feature_dim)
            features.append(feat)
        
        # Stack into sequence: (timesteps, batch, feature_dim)
        features_seq = torch.stack(features, dim=0)
        
        # Pass to HORN
        output = self.horn(features_seq, random_init=random_init, record=record)
        
        # Also return patch sequence info for debugging
        output['num_patches_h'] = num_patches_h
        output['num_patches_w'] = num_patches_w
        output['num_timesteps'] = num_timesteps
        
        return output
