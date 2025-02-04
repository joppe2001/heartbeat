import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def visualize_results(analysis, results_dir):
    """Create and save visualizations of the hyperparameter tuning results"""
    # 1. Learning Curves Plot
    plt.figure(figsize=(12, 6))
    for trial in analysis.trials:
        metrics_df = analysis.trial_dataframes[trial.trial_id]
        if not metrics_df.empty:
            # Plot both mean_accuracy and train_loss
            plt.plot(
                metrics_df.index,
                metrics_df['mean_accuracy'],
                alpha=0.3,
                linestyle='-',
                label=f'Accuracy Trial {trial.trial_id}'
            )
    plt.title('Learning Curves Across Trials (Arrhythmia Classification)')
    plt.xlabel('Training Iteration')
    plt.ylabel('Performance')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / 'learning_curves.png', bbox_inches='tight')
    plt.close()

    # 2. Parameter Importance Plot - Focus on numerical parameters
    numeric_params = {
        'learning_rate': [],
        'dropout_rate': [],
        'weight_decay': [],
        'batch_size': []
    }

    for trial in analysis.trials:
        for param, values in numeric_params.items():
            values.append(trial.config.get(param))

    numeric_df = pd.DataFrame(numeric_params)

    # Calculate correlation with performance
    performances = [
        trial.last_result.get('mean_accuracy', 0)
        for trial in analysis.trials
    ]
    numeric_df['performance'] = performances

    correlations = numeric_df.corr()['performance'].drop('performance')

    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar')
    plt.title('Parameter Impact on Model Performance')
    plt.xlabel('Parameter')
    plt.ylabel('Correlation with Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / 'parameter_importance.png')
    plt.close()

    # 3. Performance Distribution per Class
    plt.figure(figsize=(12, 6))
    metrics_to_plot = [
        'val_recall_class_0',
        'val_recall_class_1',
        'val_recall_class_2',
        'val_recall_class_3',
        'val_recall_class_4'
    ]

    class_names = ['Normal', 'S-type', 'V-type', 'F-type', 'Q-type']

    performance_data = []
    for trial in analysis.trials:
        if trial.last_result:
            for metric in metrics_to_plot:
                if metric in trial.last_result:
                    performance_data.append({
                        'Class': class_names[int(metric.split('_')[-1])],
                        'Recall': trial.last_result[metric]
                    })

    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        sns.boxplot(data=perf_df, x='Class', y='Recall')
        plt.title('Performance Distribution Across Arrhythmia Classes')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'class_performance_distribution.png')
    plt.close()

    # 4. Architecture Configuration Analysis
    plt.figure(figsize=(12, 6))
    arch_params = {
        'num_conv_layers': [],
        'channels_1': [],
        'fc_size': []
    }

    performances = []

    for trial in analysis.trials:
        if trial.last_result:
            for param in arch_params:
                arch_params[param].append(trial.config.get(param))
            performances.append(trial.last_result.get('mean_accuracy', 0))

    arch_df = pd.DataFrame(arch_params)
    arch_df['performance'] = performances

    # Create bubble plot
    plt.scatter(
        arch_df['num_conv_layers'],
        arch_df['channels_1'],
        s=arch_df['fc_size'] / 2,  # Size of bubbles
        c=arch_df['performance'],  # Color based on performance
        alpha=0.6,
        cmap='viridis'
    )

    plt.colorbar(label='Mean Accuracy')
    plt.title('Architecture Configuration Impact')
    plt.xlabel('Number of Conv Layers')
    plt.ylabel('First Layer Channels')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'architecture_analysis.png')
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(str(results_dir / 'learning_curves.png'))
    mlflow.log_artifact(str(results_dir / 'parameter_importance.png'))
    mlflow.log_artifact(str(results_dir / 'class_performance_distribution.png'))
    mlflow.log_artifact(str(results_dir / 'architecture_analysis.png'))
