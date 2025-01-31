import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(df):
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='target')
    plt.title('Arrhythmia Class Distribution')
    plt.savefig('temp/class_distribution.png')
    plt.close()

    # 2. Sample signals from each class
    plt.figure(figsize=(15, 10))
    for class_num in df['target'].unique():
        sample = df[df['target'] == class_num].iloc[0]
        signal = sample.drop('target')
        plt.plot(signal.values, label=f'Class {int(class_num)}', alpha=0.7)
    plt.title('Sample Signals from Each Class')
    plt.legend()
    plt.savefig('temp/sample_signals.png')
    plt.close()


df = pd.read_parquet('../../data/heart_big_train.parq')
create_visualizations(df)