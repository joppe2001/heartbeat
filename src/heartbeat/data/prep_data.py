import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path


def prepare_data():
    # Get the path relative to this script's location
    current_file = Path(__file__)
    data_path = current_file.parent.parent.parent.parent / "data" / "heart_big_train.parq"

    # Convert to absolute path and read the data
    df = pd.read_parquet(str(data_path.resolve()))
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