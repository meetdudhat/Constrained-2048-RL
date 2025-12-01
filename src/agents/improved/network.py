"""
Neural Network Architecture for AlphaZero 2048.

This module defines the deep learning model that guides the MCTS search.
It acts as the "intuition" for the agent, providing:
1.  Policy Head: Which move is likely the best (Up, Down, Left, Right)?
2.  Value Head: How likely is the current board to lead to a win?

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agents.improved.config import DEVICE

class Network(nn.Module):
    """
    Implements a Hybrid-Kernel Convolutional Neural Network.

    Architectural Decisions:
    - Input Representation: The board is not passed as raw integers (e.g., 2, 1024).
      Instead, it uses One-Hot Encoding (18 channels). This prevents the network
      from getting confused by the massive magnitude difference between a '2' tile
      and a '2048' tile, treating them instead as distinct categorical features.
    
    - Parallel Convolutions: Standard CNNs use uniform 3x3 kernels. However, 
      2048 is strictly grid-aligned.
        - 1x4 Kernels: Scan entire rows to understand horizontal merges.
        - 4x1 Kernels: Scan entire columns to understand vertical merges.
        - 2x2 Kernels: Detect immediate local merges (e.g., two '4's adjacent).
    
    Args:
        input_channels (int): Depth of the input tensor (18 for one-hot layers).
        num_actions (int): Output size for the policy head (4 directions).
        filters (int): Number of feature maps per convolutional branch.
    """
    def __init__(self, input_channels=18, num_actions=4, filters=128):
        super(Network, self).__init__()
        
        # --- Body: Parallel Feature Extraction ---
        
        # Branch 1: Horizontal Scanner
        # Uses a 1x4 kernel to process a full row at once. 
        # Crucial for anticipating the result of "Left" or "Right" moves.
        self.conv_h = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(1, 4), padding=0)
        self.bn_h = nn.BatchNorm2d(filters)
        
        # Branch 2: Vertical Scanner
        # Uses a 4x1 kernel to process a full column at once.
        # Crucial for anticipating the result of "Up" or "Down" moves.
        self.conv_v = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(4, 1), padding=0)
        self.bn_v = nn.BatchNorm2d(filters)
        
        # Branch 3: Local 2x2 Pattern Detector
        # Detects small clusters, such as "checkerboard" hazards or immediate merge setups.
        # Padding is manually defined to ensure the output aligns with the 4x4 grid structure.
        self.pad_l2 = nn.ZeroPad2d((0, 1, 0, 1)) 
        self.conv_l2 = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(2, 2), padding=0)
        self.bn_l2 = nn.BatchNorm2d(filters)
        
        # Branch 4: Contextual 3x3 Pattern Detector
        # Captures broader neighborhood information.
        self.conv_l3 = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(3, 3), padding='same')
        self.bn_l3 = nn.BatchNorm2d(filters)

        # --- Feature Aggregation ---
        # We flatten the outputs of all branches and concatenate them.
        # Calculation:
        # (1, 4) kernel on 4x4 input -> Output is 4x1 -> 4 spatial locations
        # (4, 1) kernel on 4x4 input -> Output is 1x4 -> 4 spatial locations
        # (2, 2) kernel on 4x4 input -> Output is 4x4 -> 16 spatial locations
        # (3, 3) kernel on 4x4 input -> Output is 4x4 -> 16 spatial locations
        self.flat_size = (4 * filters) + (4 * filters) + (16 * filters) + (16 * filters)

        self.fc_mix = nn.Linear(self.flat_size, 256)
        self.bn_mix = nn.BatchNorm1d(256)
        
        # --- Head 1: Policy (Actor) ---
        # Outputs a probability distribution over the 4 possible moves.
        self.fc_policy_hidden = nn.Linear(256, 128)
        self.bn_policy = nn.BatchNorm1d(128)
        self.fc_policy_output = nn.Linear(128, num_actions)
        
        # --- Head 2: Value (Critic) ---
        # Outputs a scalar (-1 to 1) predicting the final game outcome.
        self.fc_value_hidden = nn.Linear(256, 128)
        self.bn_value = nn.BatchNorm1d(128)
        self.fc_value_output = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass defining the computational graph.
        
        Args:
            x (torch.Tensor): Input batch of shape (Batch, 18, 4, 4).
            
        Returns:
            policy (torch.Tensor): Logits/Probabilities for actions.
            value (torch.Tensor): Scalar evaluation of the state.
        """
        # 1. Execute Parallel Branches
        # Each branch applies Convolution -> BatchNorm -> ReLU -> Flatten
        h = F.relu(self.bn_h(self.conv_h(x))).view(x.size(0), -1)
        v = F.relu(self.bn_v(self.conv_v(x))).view(x.size(0), -1)
        
        # Note: L2 branch requires pre-padding to maintain dimensions
        l2 = F.relu(self.bn_l2(self.conv_l2(self.pad_l2(x)))).view(x.size(0), -1)
        l3 = F.relu(self.bn_l3(self.conv_l3(x))).view(x.size(0), -1)
        
        # 2. Fuse Features
        concatenated = torch.cat((h, v, l2, l3), dim=1)
        mixed = F.relu(self.bn_mix(self.fc_mix(concatenated)))
        
        # 3. Policy Head
        p = F.relu(self.bn_policy(self.fc_policy_hidden(mixed)))
        p = self.fc_policy_output(p)
        # Softmax to get probabilities for MCTS priors
        policy = F.softmax(p, dim=1) 
        
        # 4. Value Head
        v = F.relu(self.bn_value(self.fc_value_hidden(mixed)))
        v = self.fc_value_output(v)
        value = torch.tanh(v) # Squash between -1 (Lose) and 1 (Win)
        
        return policy, value

    def predict(self, state_tensor):
        """
        Inference helper for the MCTS loop (bridges CPU/Numpy and GPU/Torch).
        
        The MCTS code runs on CPU using Numpy arrays. This method handles
        the overhead of moving that single state to the GPU for a prediction.
        
        Args:
            state_tensor (np.array): shape (18, 4, 4)
            
        Returns:
            (np.array, float): The policy distribution and value scalar.
        """
        # Prepare input: Add batch dimension (1, 18, 4, 4) and send to VRAM
        x = torch.from_numpy(state_tensor).unsqueeze(0).to(DEVICE)
        
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            policy, value = self.forward(x)
        
        # Return cleanly to CPU/Numpy for MCTS processing
        return policy.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()