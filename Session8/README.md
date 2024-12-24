**Session9: Advanced Convolutions & Augmentations**: The session describes about the various types of convolutions specifying their advantages & disadvantages. It also introduces a technique to reduce over-fitting issue known as Augmentating data using PyTorch library 
Albumentations. 

 Do More: https://albumentations.ai/
 

### Session 9 Assignment: 

🔏 Problem Statement: 
--------------------
Write a new network thathas the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, 
where the last one has a stride of 2 instead) total RF must be more than 44

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
    │   │   ├── S9_model.py: The Network Architecture designed to achieve 85% accuracy.
    │   ├── utility
    │   │   ├── dataset.py: Managing the data & retrieving it.
    │   │   ├── run.py:     Makes the model learn.
    │   │   ├── utils.py:   Contains the utilities required for the process.
    │   │   ├── visualize.py: Contains the code for visualizing.
    │   ├── CIFAR10_V0.ipynb:  Execution of Network.
    └── README.md Details about the Process.


🔑 Model Architecture:
---------------------
 "C1-C2-C3-C4-output"
 * For every convolution block, there has to be 3 3x3 kernel convolutions with a stride of 2.
 * Total Parameter Count: 180,978

🔋 Augmented Images: 
-------------------
 <img src="https://github.com/kishkath/ERA/assets/60026221/c9ea71fe-3cf9-47d7-9a2c-12339e4ebbf4" width = 720 height = 360>

💊 Network Results: 
-------------------
 Trained the network for 72 Epochs with SGD optimizer and CrossEntropyLoss fn.
 
 Achieved the desired accuracy at 45th Epoch.
 
     Epoch 45
     Train: Loss=0.5043 Batch_id=390 Accuracy=77.08: 100%|██████████| 391/391 [00:17<00:00, 22.08it/s]
     Test set: Average loss: 0.0035, Accuracy: 8526/10000 (85.26%)
 
 <img src="https://github.com/kishkath/ERA/assets/60026221/2e0d4048-3233-4f65-9670-be0db37b4b15" width = 720 height = 360>

 * Mis-classified Images:
-------------------------
 <img src="https://github.com/kishkath/ERA/assets/60026221/a030e214-6184-4a86-91e1-3b2ddaa951f9" width = 720 height = 360>

