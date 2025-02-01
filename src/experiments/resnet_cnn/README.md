# ResNet CNN Experiment

## Purpose
Improved CNN architecture using ResNet blocks for better gradient flow and deeper feature extraction.

## Architecture
- 4 ResNet blocks with skip connections
- Batch normalization layers
- Global average pooling
- Deeper network compared to baseline

## Expected Improvements
- Better feature hierarchy with deeper layers
- Improved gradient flow through skip connections
- Potentially better precision while maintaining recall

## Configuration
- Learning rate: 0.001
- Batch size: 32
- Early stopping patience: 5