# Arrhythmia Classification using Deep Learning

## By Joppe Montezinos

## Project Overview
A deep learning approach to classify five types of arrhythmia patterns in ECG data, with a focus on maintaining high recall for medical applications. The project evolved through multiple architectures, from a baseline CNN to a sophisticated Dual-Path ResNet with an attention mechanism.

## Key Results
- **Baseline CNN**: 93.10% average recall
- **Enhanced CNN**: 90.62% accuracy
- **Dual-Path ResNet**: Designed for improved stability

## Dataset
- **87,554 samples** with **187 features**
- **5 classes** with significant imbalance:
  - **Class 0 (Normal)**: 82.77%
  - **Class 4 (Type 4)**: 7.35%
  - **Class 2 (Type 2)**: 6.61%
  - **Class 1 (Type 1)**: 2.54%
  - **Class 3 (Type 3)**: 0.73%

## Model Architectures

### 1. Baseline CNN
- Simple 1D CNN with weighted loss
- 2 convolutional layers
- Max pooling and 2 fully connected (FC) layers
- Early stopping at epoch 20

### 2. Enhanced CNN
- 2 convolutional blocks (32→256 channels)
- Kernel sizes: 5, 7
- Batch normalization and adaptive pooling
- Dropout rate: 0.45

### 3. Dual-Path ResNet with Attention
- 3 residual blocks (32→64→128→256)
- **Dual path design**:
  - Local path: 3x3 kernels for finer details
  - Global path: 7x7 kernels for broader patterns
- Temporal attention mechanism

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/joppe2001/heartbeat.git
   cd heartbeat
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MLflow tracking:**
   ```bash
   mlflow server --host 127.0.0.1 --port 5001
   ```

## Project Structure
```
heartbeat/
├── data/
├── journal/
├── notebooks/
├── reports/
└── src/
    ├── experiments/
    │   ├── baseline_cnn/
    │   ├── hypertuning/
    │   ├── models/
    │   └── resnet_cnn/
    ├── heartbeat/
    │   ├── data/
    │   ├── evaluation/
    │   ├── mlflow/
    │   ├── models/
    │   │   ├── cnn/
    │   │   └── resnet/
    │   ├── train/
    │   └── visualizations/
    └── utils/
```

## Training

### Baseline CNN:
```txt
src/heartbeat/models/cnn/baseline.py
```

### Enhanced CNN:
```txt
src/heartbeat/models/cnn/enhanced_cnn.py
```

### Dual-Path ResNet:
```txt
src/heartbeat/models/resnet/dualpath_attention_resnet.py
```

## Monitoring
- **Training progress**: Rich progress bars with time estimation
- **Experiment tracking**: MLflow ([http://127.0.0.1:5001](http://127.0.0.1:5001))
- **Metrics**: Confusion matrices and classification reports

## Key Features
- Custom weighted loss function for class imbalance
- Extensive hyperparameter tuning using Ray
- Focus on recall for medical context
- Progressive model complexity evolution

## Results Visualization
- Confusion matrices saved in experiment artifacts
- Per-class performance metrics
- Training progress logs

