# AIT 526 Lab 4
- Hamaad Zuberi

## 4.1 Change neural network parameters
### 4.1.1 Batch size: 16, 32, 64

Changing the batch_size (set in cell 3) would directly affect the number of iterations within the train and test loops for each epoch and the amount of memory required. As we reduce the batch_size, it also seems to be increasing the Accuracy (64.5% for 64, 70.6% for 32, 78.3% for 16) and Avg. Loss (1.117848 for 64, 0.789082 for 32, 0.621078 for 16).