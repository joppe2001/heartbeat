import torch.nn as nn
import torch.nn.functional as F
import torch

class TunableCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        # First conv layer
        in_channels = 1
        self.conv_layers.append(
            nn.Conv1d(in_channels, config["channels_1"],
                      kernel_size=config["kernel_size_1"])
        )

        # Additional conv layers
        prev_channels = config["channels_1"]
        for i in range(1, config["num_conv_layers"]):
            channels = config[f"channels_{i + 1}"]
            kernel_size = config[f"kernel_size_{i + 1}"]
            self.conv_layers.append(
                nn.Conv1d(prev_channels, channels, kernel_size=kernel_size)
            )
            prev_channels = channels

        # Calculate size after convolutions
        self.calculate_conv_output_size(config)

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, config["fc_size"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.fc2 = nn.Linear(config["fc_size"], 5)  # 5 classes

    def calculate_conv_output_size(self, config):
        # Helper function to calculate output size
        x = torch.randn(1, 1, 187)  # Sample input
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool1d(x, 2)
        self.conv_output_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool1d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
