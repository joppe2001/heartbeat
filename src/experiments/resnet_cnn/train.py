import sys
from pathlib import Path
import torch
from torch import nn
import mlflow
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from heartbeat.data.prep_data import prepare_data
from heartbeat.evaluation.metrics import evaluate_model, log_experiment
from heartbeat.mlflow.server import MLflowServer, setup_tracking
from heartbeat.models.resnet.model import ResNetECG
from config import ResNetConfig


def create_dataloaders(X, y, batch_size):
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


def detailed_evaluation(model, loader, device, save_dir):
    """Detailed evaluation with confusion matrix and classification report"""
    model.eval()
    all_preds = []
    all_labels = []

    print("\nRunning detailed evaluation...")
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Evaluating'):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Detailed Classification Report
    report = classification_report(all_labels, all_preds)

    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

    return cm, report


def analyze_errors(model, loader, device):
    """Collect examples of misclassifications"""
    model.eval()
    errors = {i: [] for i in range(5)}  # For each class

    print("\nAnalyzing errors...")
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Analyzing'):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            # Find misclassifications
            misclass = preds != y
            if misclass.any():
                for idx in torch.where(misclass)[0]:
                    true_class = y[idx].item()
                    pred_class = preds[idx].item()
                    errors[true_class].append({
                        'signal': X[idx].cpu().numpy(),
                        'predicted': pred_class
                    })
    return errors


def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0

    # Create progress bar for batches
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{num_epochs} [Train]',
                leave=False, ncols=100)

    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


def validate(model, loader, device, epoch, num_epochs):
    model.eval()
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{num_epochs} [Valid]',
                leave=False, ncols=100)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return evaluate_model(model, loader, device)


def main():
    with MLflowServer() as server:
        try:
            # Print start time and setup info
            start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}\n")

            config = ResNetConfig()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Load and prepare data
            print("\nPreparing data...")
            (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = prepare_data()

            # Create dataloaders
            train_loader = create_dataloaders(X_train, y_train, config.batch_size)
            val_loader = create_dataloaders(X_val, y_val, config.batch_size)

            print(f"\nDataset sizes:")
            print(f"Training:   {len(X_train):,} samples")
            print(f"Validation: {len(X_val):,} samples")
            print(f"Test:       {len(X_test):,} samples\n")

            # Initialize model and training components
            model = ResNetECG().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            # MLflow logging
            print("Initializing MLflow tracking...")
            setup_tracking("resnet_cnn")

            with mlflow.start_run():
                best_val_recall = 0
                patience_counter = 0

                print("\nStarting training loop...")
                print(f"{'='*50}")

                # Create progress bar for epochs
                epoch_pbar = tqdm(range(config.num_epochs), desc='Training Progress',
                                 ncols=100, position=0)

                for epoch in epoch_pbar:
                    # Train
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device,
                        epoch + 1, config.num_epochs
                    )

                    # Validate
                    val_metrics = validate(
                        model, val_loader, device,
                        epoch + 1, config.num_epochs
                    )

                    # Calculate average recall
                    avg_recall = sum(val_metrics.values()) / len(val_metrics)

                    # Update epoch progress bar
                    epoch_pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'avg_recall': f'{avg_recall:.4f}'
                    })

                    # Logging
                    metrics = {"train_loss": train_loss, **val_metrics}
                    log_experiment(vars(config), metrics)

                    # Early stopping
                    if avg_recall > best_val_recall + config.min_delta:
                        best_val_recall = avg_recall
                        patience_counter = 0
                        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
                        torch.save(
                            model.state_dict(),
                            f"{config.model_save_dir}/best_model.pth"
                        )
                        print(f"\nNew best model saved! Average recall: {avg_recall:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= config.patience:
                            print("\nEarly stopping triggered!")
                            break

                # Print training summary
                time_elapsed = time.time() - start_time
                print(f"\n{'=' * 50}")
                print("Training completed!")
                print(
                    f"Total time: {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {time_elapsed % 60:.0f}s")
                print(f"Best average recall: {best_val_recall:.4f}")
                print(f"{'=' * 50}\n")

                # Create test dataloader
                test_loader = create_dataloaders(X_test, y_test, config.batch_size)

                # Run detailed evaluation
                save_dir = Path(config.reports_save_dir)
                cm, report = detailed_evaluation(model, test_loader, device, save_dir)
                print("\nClassification Report:")
                print(report)

                # Analyze errors
                errors = analyze_errors(model, test_loader, device)
                print("\nError Analysis:")
                for class_idx, class_errors in errors.items():
                    print(f"Class {class_idx}: {len(class_errors)} misclassifications")

                # Log additional metrics to MLflow
                mlflow.log_artifact(str(save_dir / 'confusion_matrix.png'))
                mlflow.log_text(report, "classification_report.txt")

        except Exception as e:
            print(f"Training failed: {e}")
            raise e


if __name__ == "__main__":
    main()
