import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.nn as nn
from pathlib import Path
import sys
import mlflow
from heartbeat.data.prep_data import prepare_data
from heartbeat.evaluation.metrics import evaluate_model
from heartbeat.models.cnn.enhanced_cnn import EnhancedCNN

# Add src to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)


def train_cnn(config):
    # Start a nested MLflow run for this trial
    with mlflow.start_run(run_name=f"training_trial", nested=True) as run:
        # Log the trial's hyperparameters
        mlflow.log_params(config)

        # Device setup
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data preparation
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = prepare_data()

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train).unsqueeze(1),  # Add channel dimension
                torch.LongTensor(y_train)
            ),
            batch_size=int(config["batch_size"]),
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val).unsqueeze(1),  # Add channel dimension
                torch.LongTensor(y_val)
            ),
            batch_size=int(config["batch_size"])
        )

        # Model setup with enhanced CNN
        model_config = {
            "channels_1": config["channels_1"],
            "channels_2": config["channels_2"],
            "kernel_1": config["kernel_size_1"],
            "kernel_2": config["kernel_size_2"],
            "fc_size": config["fc_size"],
            "dropout": config["dropout_rate"]
        }

        model = EnhancedCNN(model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, device=device)
        )

        # Training loop
        for epoch in range(30):
            model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            val_metrics = evaluate_model(model, val_loader, device)
            mean_accuracy = (
                    0.3 * val_metrics['recall_class_0'] +  # Normal class
                    0.7 * sum(val_metrics[f'recall_class_{i}'] for i in range(1, 5)) / 4  # Abnormal classes
            )

            # Log metrics
            metrics = {
                "train_loss": avg_loss,
                "mean_accuracy": mean_accuracy,
            }

            # Add validation metrics
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

            # Log metrics to MLflow
            mlflow.log_metrics(metrics, step=epoch)

            # Report to Ray Tune
            tune.report(metrics)

        # Log the final model for this trial
        mlflow.pytorch.log_model(model, f"model_trial_training")

        return mean_accuracy


def main():
    # Initialize Ray
    ray.init()

    # Define search space for enhanced CNN
    config = {
        "channels_1": tune.choice([16, 32, 64]),
        "channels_2": tune.choice([64, 128, 256]),
        "kernel_size_1": tune.choice([3, 5, 7]),
        "kernel_size_2": tune.choice([3, 5, 7]),
        "fc_size": tune.choice([128, 256, 512]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128])
    }

    # Setup MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tune_enhanced_cnn_experiment")

    # Start the parent run for hyperparameter tuning
    with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
        # Log the search space configuration
        mlflow.log_params({
            "search_space_channels_1": "16, 32, 64",
            "search_space_channels_2": "64, 128, 256",
            "search_space_kernel_sizes": "3, 5, 7",
            "search_space_fc_size": "128, 256, 512",
            "search_space_dropout_rate": "0.1-0.5",
            "search_space_learning_rate": "1e-4-1e-2",
            "search_space_weight_decay": "1e-5-1e-3",
            "search_space_batch_size": "32, 64, 128",
            "num_samples": 30,
            "scheduler": "ASHA",
            "search_algorithm": "Optuna"
        })

        # Scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=10,
            grace_period=3,
            reduction_factor=2
        )

        # Search algorithm
        search_alg = OptunaSearch()

        # Run hyperparameter tuning
        analysis = tune.run(
            train_cnn,
            config=config,
            num_samples=30,
            scheduler=scheduler,
            search_alg=search_alg,
            metric="mean_accuracy",
            mode="max",
            resources_per_trial={
                "cpu": 4,
                "gpu": 0.5 if torch.cuda.is_available() else 0
            },
            progress_reporter=tune.CLIReporter(
                metric_columns=["mean_accuracy", "training_iteration"]
            )
        )

        # Get best trial
        best_trial = analysis.best_trial

        best_config_path = "best_enhanced_model_config.txt"
        with open(best_config_path, 'w') as f:
            f.write("Best Enhanced Model Configuration:\n")
            f.write("=" * 30 + "\n\n")
            for param, value in best_trial.config.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nBest Mean Accuracy: {best_trial.last_result['mean_accuracy']:.4f}")

        # Log best trial info in parent run
        mlflow.log_params({
            "best_trial_config": best_trial.config,
            "best_trial_id": best_trial.trial_id,
            "best_trial_accuracy": best_trial.last_result['mean_accuracy']
        })

        # Log the results dataframe
        results_df = analysis.results_df
        results_df.to_csv("tune_enhanced_results.csv")
        mlflow.log_artifact("tune_enhanced_results.csv")

        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial mean_accuracy: {best_trial.last_result['mean_accuracy']}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()