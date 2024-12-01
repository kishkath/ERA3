# MNIST CNN Model with Parameter and Accuracy Constraints

![Model Test](https://github.com/kishkath/ERA3/actions/workflows/model_test.yml/badge.svg)

This project implements a Convolutional Neural Network (CNN) for the MNIST dataset with specific constraints:
- Less than 25,000 parameters
- Achieves >95% training accuracy in 2 epochs
- Includes automated testing via GitHub Actions

## Project Structure

```
.
├── model.py          # CNN architecture definition
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training and testing functions
├── inference.py      # Model inference utilities
├── main.py          # Main training script
├── test_model.py     # Unit tests for model constraints
├── run_all.py        # Complete workflow script
├── requirements.txt  # Project dependencies
└── .github/
    └── workflows/
        └── model_test.yml  # GitHub Actions workflow
```

## Model Architecture

The CNN architecture is designed to be lightweight yet effective:
- Input Layer: 28x28x1 (MNIST image)
- Conv1: 8 filters, 3x3 kernel
- Conv2: 16 filters, 3x3 kernel
- Conv3: 24 filters, 3x3 kernel with MaxPooling
- Fully Connected: 64 neurons
- Output: 10 classes (digits 0-9)

Total parameters: <25,000

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, simply run:
```bash
python main.py
```

The script will:
1. Initialize the CNN model
2. Load and preprocess the MNIST dataset
3. Train for two epochs
4. Validate the model meets constraints:
   - Parameter count < 25,000
   - Training accuracy > 95% after 2 epochs
5. Run inference on test set

## Using the Model for Inference

The `inference.py` module provides two main functions:

1. Single image prediction:
```python
from inference import predict_single_image
prediction, probability = predict_single_image(model, "path/to/image.jpg", device)
```

2. Batch inference:
```python
from inference import batch_inference
predictions, accuracy = batch_inference(model, test_loader, device)
```

## GitHub Actions Workflow

This project includes automated testing via GitHub Actions, which triggers on every push and pull request.

### Test Suite

The test suite (`test_model.py`) verifies three main constraints:
1. **Parameter Count**: Ensures model has less than 25,000 parameters
2. **Training Accuracy**: Verifies >95% accuracy achievement in two epochs
3. **Model Structure**: Validates model output shape and probability distribution

### Workflow Process

1. **Trigger**: The workflow runs on:
   - Every push to any branch
   - Every pull request to any branch

2. **Environment Setup**:
   - Uses Ubuntu latest
   - Sets up Python 3.8
   - Installs required dependencies

3. **Testing Steps**:
   - Runs pytest with coverage reporting
   - Executes all tests in test_model.py
   - Uploads coverage report to Codecov

### Running Tests Locally

To run the tests locally:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
python -m pytest test_model.py -v --cov=./

# Run specific test
python -m pytest test_model.py -v -k test_parameter_count
```

### Understanding Test Results

- ✅ Green check: All tests passed
  - Model has <25,000 parameters
  - Achieved >95% accuracy in 2 epochs
  - Model structure is valid
- ❌ Red X: Tests failed
  - Check logs for specific test failures
  - Review error messages for constraint violations

## Troubleshooting

Common issues and solutions:

1. **CUDA/CPU Compatibility**:
   - Code automatically detects and uses GPU if available
   - Falls back to CPU if no GPU is present

2. **Memory Issues**:
   - Batch size can be adjusted in dataset.py
   - Default batch size: 256

3. **Training Accuracy <95%**:
   - Check learning rate in optimizer
   - Verify data preprocessing
   - Ensure model architecture is correct

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run the tests locally
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 