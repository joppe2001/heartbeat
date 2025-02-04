# Arrhythmia Classification using Deep Learning: An Iterative Approach
[GitHub Repository](https://github.com/joppe2001/heartbeat)

## 1. Executive Summary
This project focused on developing a deep learning model for classifying five types of arrhythmia patterns in ECG data. Through iterative experimentation, we evolved from a baseline CNN achieving 93.10% average recall to more sophisticated architectures including an Enhanced CNN and a Dual-Path ResNet with attention mechanism. Key challenges addressed included severe class imbalance (82.77% normal cases) and maintaining high recall for minority classes.

## 2. Top Architectures

### 2.1 Baseline CNN (93.10% avg recall)
- Simple 1D CNN with weighted loss function
- Early stopping at epoch 20
- Training time: ~9 minutes
- Dataset split: 61,287/13,133/13,134 (train/val/test)

### 2.2 Enhanced CNN (90.62% accuracy)
Best Configuration:
- 2 conv blocks (channels: 32→256)
- Kernel sizes: 5, 7
- BatchNorm after each conv layer
- Adaptive pooling
- Dropout rate: 0.45
- Learning rate: 4.29e-4
- Batch size: 32

### 2.3 Dual-Path ResNet with Attention
Architecture:
- 3 residual blocks (32→64→128→256 channels)
- Dual path with different kernel sizes:
  * Local path: small kernels (3) for detail
  * Global path: large kernels (7) for patterns
- Temporal attention layer
- Progressive channel growth

## 3. Hyperparameter Search Space

| Parameter        | Range/Values              | Best Value   |
|-----------------|---------------------------|--------------|
| Conv layers     | 2-4                      | 2           |
| Channel sizes   | [16,32,64]→[128,256,512] | [32, 256]   |
| Kernel sizes    | [3,5,7]                  | [5, 7]      |
| Learning rate   | 10^-4 to 10^-2           | 4.29e-4     |
| Batch size      | [32,64,128]              | 32          |
| Dropout rate    | 0.1-0.5                  | 0.45        |

## 4. Hypotheses and Results

### 4.1 Initial Hypotheses
1. Simple CNN limitations:
   - Expected struggles with class imbalance
   - Potential issues with long-range dependencies
   ➤ Result: Surprisingly effective (93.10% recall)

2. Class imbalance handling:
   - Weighted loss function would help minority classes
   ➤ Result: Confirmed, achieved high recall across classes (0.87-0.99)

### 4.2 Architecture Evolution Hypotheses
1. Enhanced CNN:
   - BatchNorm would improve training stability
   - Wider architecture would capture better features
   ➤ Result: Mixed - showed instability in training (0.3-0.7 accuracy swings)

2. Dual-Path Architecture:
   - Separate paths for local/global patterns
   - Attention mechanism for important segments
   ➤ Result: Pending final evaluation, designed to address Enhanced CNN's instability

## 5. Key Insights and Discoveries

### 5.1 Unexpected Findings
1. Baseline Performance:
   - CNN handled class imbalance better than expected
   - Weighted loss function proved highly effective
   - Quick convergence (20 epochs) suggested efficient architecture

2. Precision-Recall Trade-offs:
   - High recall (0.87-0.99) across classes
   - Precision varied significantly (0.31-0.99)
   - Class 4 showed exceptional performance (0.99 recall, 0.94 precision)

### 5.2 Critical Challenges
1. Minority Class Detection:
   - Type 1: Good recall (0.87) but low precision (0.52)
   - Type 3: High recall (0.90) but very low precision (0.31)
   - Need for better feature discrimination in minority classes

## 6. Conclusion
The project demonstrated the effectiveness of iterative model development and hypothesis testing. While the baseline CNN showed surprisingly strong performance, subsequent architectures revealed important insights about the trade-offs between model complexity and stability. The Dual-Path ResNet represents a promising direction for combining local and global pattern recognition while maintaining high recall for minority classes.