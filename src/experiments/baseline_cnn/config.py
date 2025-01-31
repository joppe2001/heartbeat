from dataclasses import dataclass


@dataclass
class Config:
    # Model parameters
    model_name: str = "baseline_cnn"
    input_channels: int = 1
    hidden_channels: int = 32
    num_classes: int = 5

    # Training parameters
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

    # Paths
    data_path: str = "../../../data/heart_big_train.parq"
    model_save_dir: str = "./models"
    reports_save_dir: str = "../../../reports"

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001