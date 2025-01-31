# Baseline CNN Experiment

## Purpose
Initial baseline using a simple 1D CNN to establish baseline performance for arrhythmia classification.

## Hypothesis
A simple CNN should be able to capture local patterns in the ECG signal that are indicative of different arrhythmia types.

## Architecture
- 2 convolutional layers
- ReLU activation
- Max pooling
- 2 fully connected layers

## Expected Challenges
- Class imbalance (addressed using weighted loss)
- Capturing long-range dependencies