import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection handling
        self.skip = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class ResNetECG(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        # ResNet blocks
        self.layer1 = ResBlock1D(32, 32)
        self.layer2 = ResBlock1D(32, 64)
        self.layer3 = ResBlock1D(64, 128)
        self.layer4 = ResBlock1D(128, 256)

        # Global average pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x