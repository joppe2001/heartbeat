import torch
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from heartbeat.data.prep_data import prepare_data
from heartbeat.models.resnet.dualpath_attention_resnet import DualPathResNet


def load_model(model_path):
    # Default config in case it's not in the checkpoint
    default_config = {
        "init_channels": 32,
        "channels_1": 64,
        "channels_2": 128,
        "channels_3": 256,
        "fc_size": 512,
        "dropout": 0.3
    }

    console = Console()

    try:
        # Load checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )

        # Get config from checkpoint or use default
        config = checkpoint.get('config', default_config)
        if config is None:
            console.print("[yellow]Warning: No config found in checkpoint, using default config[/yellow]")
            config = default_config

        # Initialize model with config
        model = DualPathResNet(config)

        # Load and rename state dict keys
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {}

        for k, v in state_dict.items():
            # Convert old key names to new key names
            new_key = k.replace('local_path', 'local_blocks')
            new_key = new_key.replace('global_path', 'global_blocks')
            new_state_dict[new_key] = v

        # Load the renamed state dict
        try:
            model.load_state_dict(new_state_dict, strict=False)
            console.print("[green]Model loaded successfully with key remapping[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Some keys couldn't be loaded: {str(e)}[/yellow]")

        return model, config

    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        raise


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def main():
    console = Console()

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")

    try:
        # Load the saved model
        model_path = Path("../../models/saved/")
        latest_dir = max(model_path.glob("*"), key=lambda x: x.stat().st_mtime)
        model_file = latest_dir / "best_model.pt"

        # Create test results directory
        test_results_dir = latest_dir / "model_test"
        test_results_dir.mkdir(exist_ok=True)

        console.print(f"[bold yellow]Loading model from: {model_file}[/bold yellow]")
        console.print(f"[bold yellow]Test results will be saved to: {test_results_dir}[/bold yellow]")

        # Load model and get config
        model, config = load_model(model_file)
        model.to(device)
        model.eval()

        # Print model configuration
        console.print("\n[bold cyan]Model Configuration:[/bold cyan]")
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Parameter", style="dim")
        config_table.add_column("Value")

        for key, value in config.items():
            config_table.add_row(str(key), str(value))

        console.print(config_table)

        # Load test data
        console.print("\n[bold yellow]Loading test data...[/bold yellow]")
        (_, _), (_, _), (X_test, y_test), _ = prepare_data()

        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test).unsqueeze(1),
                torch.LongTensor(y_test)
            ),
            batch_size=32
        )

        # Evaluate
        console.print("\n[bold green]Starting evaluation...[/bold green]")

        all_preds = []
        all_labels = []

        # Track predictions per class for analysis
        class_predictions = {i: {'correct': 0, 'total': 0} for i in range(5)}

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)

                # Track per-class accuracy
                for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                    class_predictions[t]['total'] += 1
                    if p == t:
                        class_predictions[t]['correct'] += 1

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        report = classification_report(all_labels, all_preds, digits=4)
        cm = confusion_matrix(all_labels, all_preds)

        # Save confusion matrix plot
        plot_path = test_results_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, plot_path)

        # Save per-class accuracy analysis
        class_accuracy_path = test_results_dir / "class_accuracy.txt"
        with open(class_accuracy_path, "w") as f:
            f.write("Per-Class Accuracy Analysis:\n")
            f.write("=" * 30 + "\n")
            for class_idx, stats in class_predictions.items():
                accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"Class {class_idx}:\n")
                f.write(f"  Total samples: {stats['total']}\n")
                f.write(f"  Correct predictions: {stats['correct']}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n\n")

        # Display results
        console.print("\n[bold cyan]Classification Report:[/bold cyan]")
        console.print(report)

        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1-Score")

        # Add rows from classification report
        class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        report_dict = classification_report(all_labels, all_preds, output_dict=True)

        for i, class_name in enumerate(class_names):
            metrics = report_dict[str(i)]
            table.add_row(
                class_name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}"
            )

        console.print(table)

        # Save detailed results to test directory
        with open(test_results_dir / "test_results.txt", "w") as f:
            f.write("Model Configuration:\n")
            f.write("=" * 20 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\nClassification Report:\n")
            f.write("=" * 20 + "\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write("=" * 20 + "\n")
            f.write(str(cm))

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': all_labels,
            'predicted_label': all_preds
        })
        predictions_df.to_csv(test_results_dir / "predictions.csv", index=False)

        console.print(f"\n[bold green]Results saved to: {test_results_dir}[/bold green]")
        console.print(f"[bold green]Confusion matrix plot saved to: {plot_path}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during evaluation: {str(e)}[/bold red]")
        console.print_exception()
        return


if __name__ == "__main__":
    main()