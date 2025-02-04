import torch
import torch.nn as nn
from pathlib import Path
import sys
import mlflow
from mlflow.models.signature import infer_signature
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from heartbeat.data.prep_data import prepare_data
from heartbeat.evaluation.metrics import evaluate_model
from heartbeat.models.cnn.enhanced_cnn import EnhancedCNN

# Add src to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)


def create_progress_bar():
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    console = Console()
    best_val_accuracy = 0

    # Get an input example for model signature
    example_input = next(iter(train_loader))[0][:1]  # Get first batch, first item
    example_output = model(example_input.to(device))

    # Create model signature
    input_signature = infer_signature(
        example_input.numpy(),
        example_output.cpu().detach().numpy()
    )

    with create_progress_bar() as progress:
        epoch_task = progress.add_task("[cyan]Training...", total=num_epochs)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            batch_task = progress.add_task(f"[green]Epoch {epoch + 1}", total=len(train_loader))

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress.update(batch_task, advance=1)

            avg_train_loss = total_loss / len(train_loader)

            # Validation phase
            val_metrics = evaluate_model(model, val_loader, device)
            mean_accuracy = (
                    0.3 * val_metrics['recall_class_0'] +  # Normal class
                    0.7 * sum(val_metrics[f'recall_class_{i}'] for i in range(1, 5)) / 4  # Abnormal classes
            )

            # Log metrics to MLflow
            metrics = {
                "train_loss": avg_train_loss,
                "mean_accuracy": mean_accuracy,
            }
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

            mlflow.log_metrics(metrics, step=epoch)

            # Display metrics in a nice table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            table.add_row("Training Loss", f"{avg_train_loss:.4f}")
            table.add_row("Mean Accuracy", f"{mean_accuracy:.4f}")
            for key, value in val_metrics.items():
                table.add_row(key, f"{value:.4f}")

            console.print(table)

            # Save best model with signature
            if mean_accuracy > best_val_accuracy:
                best_val_accuracy = mean_accuracy
                mlflow.pytorch.log_model(
                    model,
                    "best_model",
                    signature=input_signature,
                    input_example=example_input.numpy()
                )
                console.print("[bold green]Saved new best model![/bold green]")

            progress.update(epoch_task, advance=1)
            progress.remove_task(batch_task)


def main():
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("enhanced_cnn_full_training")

    # Best hyperparameters from tuning
    config = {
        "channels_1": 32,
        "channels_2": 256,
        "kernel_1": 5,
        "kernel_2": 7,
        "fc_size": 512,
        "dropout": 0.45248288505781337,
        "learning_rate": 0.00042853729091914684,
        "weight_decay": 6.500779072784835e-05,
        "batch_size": 32
    }

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console = Console()
    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")

    with mlflow.start_run(run_name="enhanced_cnn_full_training"):
        # Add detailed run description
        mlflow.set_tag("mlflow.note.content", """
        Full training run of Enhanced CNN with:
        - Batch Normalization
        - Residual Connections
        - Adaptive Pooling
        - 100 epochs
        - Best hyperparameters from tuning
        """)

        # Log hyperparameters
        mlflow.log_params(config)

        # Data preparation
        console.print("[bold yellow]Preparing data...[/bold yellow]")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = prepare_data()

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train).unsqueeze(1),
                torch.LongTensor(y_train)
            ),
            batch_size=config["batch_size"],
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val).unsqueeze(1),
                torch.LongTensor(y_val)
            ),
            batch_size=config["batch_size"]
        )

        # Model setup
        model = EnhancedCNN(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, device=device)
        )

        # Training
        console.print("[bold green]Starting training...[/bold green]")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=100  # Increased number of epochs for full training
        )

        console.print("[bold green]Training completed![/bold green]")


if __name__ == "__main__":
    main()