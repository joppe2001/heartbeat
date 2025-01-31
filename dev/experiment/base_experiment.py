import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
import torch
from torch import nn
import torch.nn.functional as F


# Data preparation
def prepare_data():
    df = pd.read_parquet('../data/heart_big_train.parq')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Calculate class weights for weighted loss
    class_counts = np.bincount(y.astype(int))
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.FloatTensor(class_weights)

    # Stratified split to maintain class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights


# Simple 1D CNN as baseline
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 43, 128)  # Adjust size based on your data
        self.fc2 = nn.Linear(128, 5)  # 5 classes

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)  # Shape: [batch, 1, sequence_length]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Add main execution and logging here