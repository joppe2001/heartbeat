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
from heartbeat.models.resnet.dualpath_attention_resnet import DualPathResNet
from datetime import datetime

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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, config):
    console = Console()
    best_val_accuracy = 0
    best_model_path = None  # Track the path of the current best model

    # Get an input example for model signature
    example_input = next(iter(train_loader))[0][:1]
    example_output = model(example_input.to(device))

    # Create save directory
    save_dir = Path(__file__).parent / "models" / "saved" / datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Added gradient clipping
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

            # Learning rate scheduling
            scheduler.step(avg_train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
            metrics = {
                "train_loss": avg_train_loss,
                "mean_accuracy": mean_accuracy,
                "learning_rate": current_lr
            }
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value

            mlflow.log_metrics(metrics, step=epoch)

            # Display metrics
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            table.add_row("Training Loss", f"{avg_train_loss:.4f}")
            table.add_row("Mean Accuracy", f"{mean_accuracy:.4f}")
            table.add_row("Learning Rate", f"{current_lr:.6f}")
            for key, value in val_metrics.items():
                table.add_row(key, f"{value:.4f}")

            console.print(table)

            # Save best model (both MLflow and locally)
            if mean_accuracy > best_val_accuracy:
                best_val_accuracy = mean_accuracy

                # Delete previous best model if it exists
                if best_model_path is not None and best_model_path.exists():
                    best_model_path.unlink()

                # Save to MLflow
                mlflow.pytorch.log_model(
                    model,
                    "best_model",
                    signature=input_signature,
                    input_example=example_input.numpy()
                )

                # Save locally with full config
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': mean_accuracy,
                    'config': config,  # Save full config
                    'val_metrics': val_metrics
                }

                best_model_path = save_dir / "best_model.pt"  # Fixed filename without accuracy
                torch.save(save_dict, best_model_path)

                # Log model path to MLflow
                mlflow.log_artifact(str(best_model_path))

                console.print(f"[bold green]Saved new best model! Accuracy: {mean_accuracy:.4f}[/bold green]")
                console.print(f"[bold blue]Model saved to: {best_model_path}[/bold blue]")

            progress.update(epoch_task, advance=1)
            progress.remove_task(batch_task)

def main():
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("dual_path_resnet_training")

    # Model configuration
    config = {
        "init_channels": 32,
        "channels_1": 64,
        "channels_2": 128,
        "channels_3": 256,
        "fc_size": 512,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 64
    }

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console = Console()
    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")

    with mlflow.start_run(run_name="dual_path_resnet_training"):
        mlflow.set_tag("mlflow.note.content", """
        Dual-Path ResNet with Attention:
        - Local path (3x3 kernels) for fine details
        - Global path (7x7 kernels) for broader patterns
        - Temporal attention mechanism
        - Progressive channel growth
        - Gradient clipping and LR scheduling
        """)

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
        model = DualPathResNet(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, device=device)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        # Training
        console.print("[bold green]Starting training...[/bold green]")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=100,
            config=config  # Pass config to train_model
        )

        console.print("[bold green]Training completed![/bold green]")

if __name__ == "__main__":
    main()