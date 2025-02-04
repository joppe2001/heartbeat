import torch.nn as nn


class EnhancedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, config["channels_1"], kernel_size=config["kernel_1"]),
            nn.BatchNorm1d(config["channels_1"], momentum=0.01),  # Added momentum
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Second conv block with residual
        self.conv2 = nn.Sequential(
            nn.Conv1d(config["channels_1"], config["channels_2"], kernel_size=config["kernel_2"]),
            nn.BatchNorm1d(config["channels_2"], momentum=0.01),  # Added momentum
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers with safety check for BatchNorm
        self.fc_features = nn.Sequential(
            nn.Linear(config["channels_2"], config["fc_size"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"])
        )

        # Separate BatchNorm layer for safe handling
        self.bn = nn.BatchNorm1d(config["fc_size"], momentum=0.01)

        # Final classification layer
        self.classifier = nn.Linear(config["fc_size"], 5)

    def forward(self, x):
        # If input is a single sample, add dummy batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Apply FC layers
        x = self.fc_features(x)

        # Only apply BatchNorm during training with more than 1 sample
        if self.training and x.size(0) > 1:
            x = self.bn(x)

        x = self.classifier(x)
        return x