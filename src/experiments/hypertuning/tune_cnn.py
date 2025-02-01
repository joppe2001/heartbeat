import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys


# Add src to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from heartbeat.data.prep_data import prepare_data
from heartbeat.evaluation.metrics import evaluate_model


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


def train_cnn(config):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = prepare_data()

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        ),
        batch_size=int(config["batch_size"]),
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        ),
        batch_size=int(config["batch_size"])
    )

    # Model setup
    model = TunableCNN(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device)
    )

    # Training loop
    for epoch in range(10):  # Reduced epochs for faster tuning
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        weighted_avg_recall = (
                0.3 * val_metrics['recall_class_0'] +  # Normal class
                0.7 * sum(val_metrics[f'recall_class_{i}'] for i in range(1, 5)) / 4  # Abnormal classes
        )

        # Report to Ray Tune
        tune.report(
            avg_recall=weighted_avg_recall,
            **val_metrics
        )


def main():
    ray.init()

    # Define search space
    config = {
        "num_conv_layers": tune.choice([2, 3, 4]),
        "channels_1": tune.choice([16, 32, 64]),
        "channels_2": tune.choice([32, 64, 128]),
        "channels_3": tune.choice([64, 128, 256]),
        "channels_4": tune.choice([128, 256, 512]),
        "kernel_size_1": tune.choice([3, 5, 7]),
        "kernel_size_2": tune.choice([3, 5, 7]),
        "kernel_size_3": tune.choice([3, 5, 7]),
        "kernel_size_4": tune.choice([3, 5, 7]),
        "fc_size": tune.choice([64, 128, 256]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128])
    }

    # Scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=3,
        reduction_factor=2,
        metric="avg_recall",
        mode="max"
    )

    # Search algorithm
    search_alg = OptunaSearch(
        metric="avg_recall",
        mode="max"
    )

    # Run hyperparameter tuning
    analysis = tune.run(
        train_cnn,
        config=config,
        num_samples=30,  # Number of trials
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={
            "cpu": 4,
            "gpu": 0.5 if torch.cuda.is_available() else 0
        },
        progress_reporter=tune.CLIReporter(
            metric_columns=["avg_recall", "training_iteration"]
        )
    )

    # Get best trial
    best_trial = analysis.best_trial
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial avg_recall: {best_trial.last_result['avg_recall']}")


if __name__ == "__main__":
    main()