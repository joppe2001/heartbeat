from torch import nn
import torch.nn.functional as F


# Simple 1D CNN as baseline
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 43, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 classes

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x