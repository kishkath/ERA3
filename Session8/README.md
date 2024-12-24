**Session8: Advanced Convolutions & Augmentations**: The session describes about the various types of convolutions specifying their advantages & disadvantages. It also introduces a technique to reduce over-fitting issue known as Augmentating data using PyTorch library 
Albumentations. 

 Do More: https://albumentations.ai/
 

### Session 8 Assignment: 

🔏 Problem Statement: 
--------------------
Write a new network that has the architecture to C1C2C3C40 (No MaxPooling, but convolutions, where the last one has a stride of 2 instead) (NO restriction on using 1x1) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)

one of the layers must use Depthwise Separable Convolution
 
one of the layers must use Dilated Convolution

use GAP (compulsory):- add FC after GAP to target #of classes (optional)

use albumentation library and apply:

  horizontal flip
  
  shiftScaleRotate
  
  coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your 
  dataset), mask_fill_value = None)

-> Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

💡 Define Problem:
------------------
 Develop the neural network such that it follows the provided architecture maintaining less than 200K parameters to achieve validation-accuracy of 85%.

🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Advanced Convolutions & Augmentations
    │   ├── models
    │   │   ├── model.py: The Network Architecture designed to achieve 85% accuracy.
    │   ├── utility
    │   │   ├── dataset.py: Managing the data & retrieving it.
    │   │   ├── run.py:     Makes the model learn.
    │   │   ├── utils.py:   Contains the utilities required for the process.
    │   │   ├── visualize.py: Contains the code for visualizing.
    │   ├── cifar-10c.ipynb:  Execution of Network.
    └── README.md Details about the Process.


🔑 Model Architecture:
---------------------
 "C1-C2-C3-C4-output"
 * For every convolution block, there has to be 3 3x3 kernel convolutions with a stride of 2.
 * Total Parameter Count: 196,330
   
 * Calculation formulae:
   --------------------
    ## Formulas for Calculations

     ### 1. Output Feature Map Size (N_out):
     N_out = floor((N_in + 2 * Padding - Kernel Size) / Stride) + 1
     
     ### 2. Receptive Field (RF_out):
     RF_out = RF_in + (Kernel Size - 1) * J_in
     
     ### 3. Output Jump (J_out):
     J_out = J_in * Stride
     
     ### 4. Effective Kernel Size for Dilated Convolutions:
     Effective Kernel Size = Kernel Size + (Kernel Size - 1) * (Dilation - 1)

 
     ------------------------------------------------------------------------------------------------------------------------------
     | Block            | Layer                              | Kernel Size | Stride | Padding | Dilation | N-out | RF-out | J-out |
     |------------------|------------------------------------|-------------|--------|---------|----------|-------|--------|-------|
     | **Conv Block 1** | Conv2D (3 → 32)                    | 3           | 1      | 1       | 1        | 32    | 3      | 1     |
     |                  | Conv2D (32 → 64, dilated)          | 3           | 1      | 1       | 2        | 32    | 7      | 1     |
     | **Conv Block 2** | Conv2D (64 → 64)                   | 3           | 1      | 1       | 1        | 32    | 9      | 1     |
     |                  | Conv2D (64 → 128, stride 2)        | 3           | 2      | 1       | 1        | 16    | 13     | 2     |
     | **Conv Block 3** | Depthwise Conv2D (128 → 128)       | 3           | 1      | 0       | 1        | 14    | 17     | 2     |
     |                  | Pointwise Conv2D (128 → 64)        | 1           | 1      | 0       | 1        | 14    | 17     | 2     |
     |                  | Conv2D (64 → 64, stride 2)         | 3           | 2      | 0       | 1        | 6     | 25     | 4     |
     | **Conv Block 4** | Conv2D (64 → 32)                   | 3           | 1      | 0       | 1        | 4     | 33     | 4     |
     | **Global Pool**  | AdaptiveAvgPool2d (1 × 1)          | -           | -      | -       | -        | 1     | 49     | 4     |
     ------------------------------------------------------------------------------------------------------------------------------

🔋 Augmented Images: 
-------------------
 <img src="https://github.com/kishkath/ERA/assets/60026221/c9ea71fe-3cf9-47d7-9a2c-12339e4ebbf4" width = 720 height = 360>

💊 Network Results: 
-------------------
 Trained the network for 60 Epochs with SGD optimizer and CrossEntropyLoss fn.

 Neural Network has the receptive field of 49
 
 Achieved the desired accuracy at 47th Epoch.
 
     
    Epoch 47
    Train: Loss=0.7390 Batch_id=390 Accuracy=77.51: 100%|██████████| 391/391 [01:43<00:00,  3.77it/s]
    Test set: Average loss: 0.0034, Accuracy: 8557/10000 (85.57%)
    
    Epoch 48
    Train: Loss=0.5536 Batch_id=390 Accuracy=77.60: 100%|██████████| 391/391 [01:40<00:00,  3.89it/s]
    Test set: Average loss: 0.0036, Accuracy: 8475/10000 (84.75%)
    
    Epoch 49
    Train: Loss=0.7201 Batch_id=390 Accuracy=77.87: 100%|██████████| 391/391 [01:37<00:00,  4.01it/s]
    Test set: Average loss: 0.0035, Accuracy: 8493/10000 (84.93%)
    
    Epoch 50
    Train: Loss=0.4808 Batch_id=390 Accuracy=77.90: 100%|██████████| 391/391 [01:56<00:00,  3.36it/s]
    Test set: Average loss: 0.0034, Accuracy: 8530/10000 (85.30%)
    


