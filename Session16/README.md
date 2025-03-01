# UNet Segmentation Assignment

This assignment is to train your own UNet from scratch for image segmentation on the Oxford-IIIT Pet dataset. The goal is to train the network four times with different configurations:

1. **MP+Tr+BCE**:  
   - **MP**: Max Pooling for downsampling  
   - **Tr**: Transpose Convolution for upsampling  
   - **BCE**: Binary Cross-Entropy Loss

2. **MP+Tr+Dice Loss**:  
   - **MP**: Max Pooling for downsampling  
   - **Tr**: Transpose Convolution for upsampling  
   - **Dice Loss**

3. **StrConv+Tr+BCE**:  
   - **StrConv**: Strided Convolution for downsampling  
   - **Tr**: Transpose Convolution for upsampling  
   - **BCE Loss**

4. **StrConv+Ups+Dice Loss**:  
   - **StrConv**: Strided Convolution for downsampling  
   - **Ups**: Bilinear Upsampling for upsampling  
   - **Dice Loss**

The project is organized into several modules that handle dataset loading, model definition, loss functions, training/validation, and visualization.

---

## Folder Structure

```
project/
├── data/                            # Data directory (downloaded via torchvision)
│    └── oxford-iiit-pet/            # Oxford-IIIT Pet dataset folder
│         ├── images/
│         └── annotations/
│              └── trimaps/
├── dataset.py                       # Custom dataset with cleaning and preprocessing
├── model.py                         # Standard UNet (MP+Tr)
├── strided_unet_transpose.py        # UNet variant: Strided Convolution + Transpose Convolution
├── strided_unet_upsample.py         # UNet variant: Strided Convolution + Bilinear Upsampling
├── losses.py                        # Loss functions (BCE and Dice loss)
├── trainer.py                       # Training and validation functions
├── utils.py                         # Visualization utilities (e.g. plot_imgs, visualize_results)
├── unet_maxpool_transpose
     └── model.py  
├── unet_strided_upsampling
     └── model.py  
└── unet_strided_transpose/
     └── model.py                    # Main training & inference script
```

---

## Requirements

Install the following packages (see `requirements.txt`):
    
    ```plaintext
    torch>=1.8.0
    torchvision>=0.9.0
    albumentations>=1.0.0
    opencv-python>=4.5.3.56
    matplotlib>=3.3.4
    tqdm>=4.60.0
    Pillow>=8.0.0
    torchsummary>=1.5.1
    numpy>=1.19.5
    ```
    
    Install dependencies with:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ---

## Usage

The main script (`main.py`) lets you choose the model architecture, loss function, and other training parameters via command‑line arguments. Key arguments include:

    - `--data_root`: Path to the dataset root (e.g., `./data/oxford-iiit-pet`)
    - `--epochs`: Number of training epochs.
    - `--batch_size`: Batch size.
    - `--lr`: Learning rate.
    - `--step_size`: Step size for the learning rate scheduler.
    - `--gamma`: Gamma factor for the scheduler.
    - `--loss_type`: Loss function, either `bce` or `dice`.
    - `--model_type`: Model architecture, choose among:
      - `unet` (MP+Tr)
      - `strided_transpose` (StrConv+Tr)
      - `strided_upsample` (StrConv+Ups)
    - `--checkpoint_path`: File path to save the best model checkpoint.
    - `--infer_image`: (Optional) Path to an image (or directory) for inference.
    - `--plot_samples`: (Optional) Number of training samples to visualize.
    - `--plot_mode`: (Optional) Visualization mode: `both`, `image`, or `mask`.
    - `--plot_save_path`: (Optional) File path to save the visualization figure.

### Running on Kaggle/Colab

After placing all files in your Kaggle working directory (with proper folder structure), run the following one-liner in a Kaggle Notebook cell (adjust paths as needed):

```bash
from types import SimpleNamespace
from main import main

args = SimpleNamespace(
    data_root="./data/oxford-iiit-pet",         # Dataset will be downloaded here.
    epochs=10,                                 # Use a small number of epochs for testing.
    batch_size=32,
    lr=1e-4,
    model_type="strided_upsample",             # Choose from "unet", "strided_transpose", or "strided_upsample"
    step_size=8,
    gamma=0.1,
    bce_weight=0.5,
    plot_samples=12,                           # Number of training samples to visualize.
    plot_mode="both",                          # "both": display images and masks side by side.
    loss_type="dice",                          # Choose "bce" or "dice"
    checkpoint_path="/content/checkpoint.pth",
    infer_image="/content/dog.jpg"             # Provide a valid path for inference.
)

main(args)

```

Replace `"your_image.jpg"` with an actual filename if you want inference.

---

## Training & Inference

- The dataset is automatically downloaded via torchvision if not already present and then loaded by `CustomDataset` (which cleans the file lists and applies Albumentations transforms).
- The training and validation loops are defined in `trainer.py` and compute both average loss and pixel-wise accuracy.
- The model summary is displayed (if `torchsummary` is installed).
- After training, the best model checkpoint is saved.
- For inference, the script can process either a single image or all images in a directory, and results are visualized using functions in `utils.py`.

---

## Results

After training, report your results (e.g., average train/validation loss and accuracy) for each configuration:
1. **MP+Tr+BCE**: Standard UNet with BCE loss.
2. **MP+Tr+Dice Loss**: Standard UNet with Dice loss.
3. **StrConv+Tr+BCE**: Strided Convolution with Transpose Convolution using BCE loss.
4. **StrConv+Ups+Dice Loss**: Strided Convolution with Upsampling using Dice loss.

Include screenshots or saved figures of the predicted masks compared to the input images.

---

## Conclusion

This project implements four different training configurations for UNet-based segmentation. The code is modularized into dataset, model, loss, trainer, and utility modules. You can easily switch between model types and loss functions via command-line arguments. The assignment demonstrates training and inference for a custom segmentation task on the Oxford-IIIT Pet dataset.

---

## License

This project is provided for educational purposes.
