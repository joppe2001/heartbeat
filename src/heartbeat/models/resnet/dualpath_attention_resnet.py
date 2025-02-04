import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm1d(out_channels)
        )

        # Skip connection with 1x1 conv if channels change
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride),
            nn.BatchNorm1d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.BatchNorm1d(channels // 8),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)
        return x * att


class DualPathResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(1, config["init_channels"], 7, padding=3),
            nn.BatchNorm1d(config["init_channels"]),
            nn.ReLU()
        )

        # Local path (small kernels)
        self.local_blocks = nn.ModuleList([
            ResidualBlock(config["init_channels"], config["channels_1"], kernel_size=3),
            ResidualBlock(config["channels_1"], config["channels_2"], kernel_size=3),
            ResidualBlock(config["channels_2"], config["channels_3"], kernel_size=3)
        ])

        # Global path (large kernels)
        self.global_blocks = nn.ModuleList([
            ResidualBlock(config["init_channels"], config["channels_1"], kernel_size=7),
            ResidualBlock(config["channels_1"], config["channels_2"], kernel_size=7),
            ResidualBlock(config["channels_2"], config["channels_3"], kernel_size=7)
        ])

        # Attention modules
        self.local_attention = TemporalAttention(config["channels_3"])
        self.global_attention = TemporalAttention(config["channels_3"])

        # Feature fusion
        combined_channels = config["channels_3"] * 2
        self.fusion = nn.Sequential(
            nn.Conv1d(combined_channels, config["fc_size"], 1),
            nn.BatchNorm1d(config["fc_size"]),
            nn.ReLU()
        )

        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["fc_size"], 5)

    def forward(self, x):
        # Initial shared features
        x = self.init_conv(x)

        # Local path
        local_features = x
        for block in self.local_blocks:
            local_features = block(local_features)
        local_features = self.local_attention(local_features)

        # Global path
        global_features = x
        for block in self.global_blocks:
            global_features = block(global_features)
        global_features = self.global_attention(global_features)

        # Combine paths
        combined = torch.cat([local_features, global_features], dim=1)
        fused = self.fusion(combined)

        # Classification
        pooled = self.pool(fused).squeeze(-1)
        features = self.dropout(pooled)
        output = self.fc(features)

        return output