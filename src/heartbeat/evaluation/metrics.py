import torch
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate per-class recall
    recalls = recall_score(all_labels, all_preds, average=None)

    return {
        f'recall_class_{i}': recall
        for i, recall in enumerate(recalls)
    }


def log_experiment(params, metrics, model_path=None):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if model_path:
        mlflow.log_artifact(model_path)


def detailed_evaluation(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Detailed Classification Report
    report = classification_report(all_labels, all_preds, output_dict=True)

    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        'confusion_matrix': cm,
        'classification_report': report
    }
