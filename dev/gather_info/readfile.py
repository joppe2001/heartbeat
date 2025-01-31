import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_parquet('../../data/heart_big_train.parq')

# Basic information gathering
print("\n=== Basic Dataset Information ===")
print(df.info())
print("\n=== Sample of Data ===")
print(df.head())
print("\n=== Data Shape ===")
print(f"Dataset shape: {df.shape}")
print("\n=== Column Names ===")
print(df.columns.tolist())

# If there's a target variable
if 'target' in df.columns:  # adjust 'target' to actual label column name
    print("\n=== Class Distribution ===")
    print(df['target'].value_counts(normalize=True))

    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    df['target'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png')