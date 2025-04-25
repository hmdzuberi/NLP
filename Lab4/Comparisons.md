# AIT 526 Lab 4
- April 24, 2025

- Hamaad Zuberi
- G01413525



I couldn't figure out to present the answers so I've exported notebooks for each implementation of the different parameters asked (see file names). The `result_base.html` contains the output of the base result (64 batch size, 0.001 learning rate, 3 layers, 512 layer size, and using ReLU activation function).

## 4.1 Change neural network parameters
### 4.1.1 Batch size: 16, 32, 64

(see cell 3)

Changing the batch_size would directly affect the number of iterations within the train and test loops for each epoch and the amount of memory required. As we reduce the batch_size, it also seems to be increasing the Accuracy (64.5% for 64, 70.6% for 32, 78.3% for 16) and Avg. Loss (1.117848 for 64, 0.789082 for 32, 0.621078 for 16).

### 4.1.2 Learning rate: 0.01, 0.001, 0.0001

(see cell 5)

Higher learning rates (e.g., 0.01) usually lead to faster progress but can be unstable; lower rates (0.001, 0.0001) are more stable but slower. In limited epochs, 0.01 is expected to perform best. Results matched expectations—0.01 (81.2%) > 0.001 (64.5%) > 0.0001 (40.7%).

le=0.0001 Accuracy: 40.7%, Avg loss: 2.247688
le=0.001 Accuracy: 64.5%, Avg loss: 1.117848
le=0.01 Accuracy: 81.2%, Avg loss: 0.519866

## 4.2 Change neural network models
### 4.2.1 Number of layers: 1 layer, 2 layers, 3 layers

(see cell 4)

Although deeper models (2–3 layers) typically outperform shallower ones with proper training, the 1-layer model achieved the highest accuracy (72.5%). The deeper models performed worse (69.2% and 64.5%) likely due to undertraining—limited epochs and sub-optimal hyperparameters hindered their convergence.

3 layers: Accuracy: 64.5%, Avg loss: 1.117848
2 layers: Accuracy: 69.2%, Avg loss: 0.911143
1 layer: Accuracy: 72.5%, Avg loss: 0.856679

4.2.2 Size of the layer: 128, 256, 512

(see cell 4)

Larger layers (512) performed better than smaller ones (256, 128), with higher accuracy and lower loss. This matches expectation that larger layers capture more complex patterns leading to better performance but, all accuracies are very low (62-65%) and the improvements with size are marginal.

512: Accuracy: 64.5%, Avg loss: 1.117848
256: Accuracy: 62.9%, Avg loss: 1.169294
128: Accuracy: 60.6%, Avg loss: 1.314661

4.2.3 Activation functions: Sigmod, ReLU

(see cell 4)

ReLU usually trains faster and gives better accuracy than Sigmoid, especially in deeper networks, because it avoids the vanishing gradient issue. Sigmoid can still work, but often learns slower. The result with ReLU (64.5%) is expected, but Sigmoid only reached 14.3%—much worse than expected and close to random guessing (10%). This very low performance of Sigmoid likely comes from the vanishing gradient problem. Sigmoid’s gradients shrink for large input values, making it hard for early layers to learn. This was likely made worse by limited training (few epochs, low learning rate).

ReLU: Accuracy: 64.5%, Avg loss: 1.117848
Sigmoid: Accuracy: 14.3%, Avg loss: 2.290145


