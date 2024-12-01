import unittest
import torch
import torch.nn.functional as F
from model import Net, get_summary
from dataset import get_data_loaders
from dataset_augmented import get_augmented_data_loaders, get_mean_std
from train import train
import numpy as np

class TestModelConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = Net().to(cls.device)
        cls.train_loader, cls.test_loader = get_data_loaders(batch_size=128)
        cls.aug_train_loader, cls.aug_test_loader = get_augmented_data_loaders(batch_size=128)
        cls.optimizer = torch.optim.Adam(cls.model.parameters())

    def test_parameter_count(self):
        """Test if model has less than 25000 parameters"""
        total_params = get_summary(self.model)
        self.assertLess(total_params, 25000, 
                       f"Model has {total_params} parameters, exceeding limit of 25000")

    def test_training_accuracy(self):
        """Test if model achieves >95% accuracy in two epochs"""
        accuracies = []
        for epoch in range(1, 3):  # Run for 2 epochs
            accuracy = train(self.model, self.device, self.train_loader, 
                           self.optimizer, epoch=epoch)
            accuracies.append(accuracy)
        
        final_accuracy = accuracies[-1]  # Get the last epoch's accuracy
        self.assertGreater(final_accuracy, 0.95,
                          f"Model achieved only {final_accuracy*100:.2f}% accuracy, "
                          f"below required 95%")

    def test_model_structure(self):
        """Test basic model structure and forward pass"""
        batch_size = 10
        x = torch.randn(batch_size, 1, 28, 28).to(self.device)
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 10),
                        "Model output shape is incorrect")
        
        # Check if output is a valid probability distribution
        self.assertTrue(torch.allclose(torch.exp(output).sum(dim=1), 
                                     torch.ones(batch_size).to(self.device), 
                                     atol=1e-6),
                       "Model output is not a valid probability distribution")

    # New Test 1: Test Input Shape Requirements
    def test_input_shape(self):
        """Test if model handles correct and incorrect input shapes"""
        # Test correct shape
        correct_shape = torch.randn(1, 1, 28, 28).to(self.device)
        try:
            _ = self.model(correct_shape)
            shape_ok = True
        except:
            shape_ok = False
        self.assertTrue(shape_ok, "Model failed to process correct input shape")

        # Test incorrect shapes
        wrong_shapes = [
            torch.randn(1, 2, 28, 28),  # Wrong channels
            torch.randn(1, 1, 32, 32),  # Wrong height/width
            torch.randn(1, 1, 28)       # Missing dimension
        ]
        
        for idx, wrong_shape in enumerate(wrong_shapes):
            try:
                wrong_shape = wrong_shape.to(self.device)
                _ = self.model(wrong_shape)
                self.fail(f"Model should not accept wrong shape {idx}")
            except:
                pass

    # New Test 2: Test Model Components
    def test_model_components(self):
        """Test if model has all required components"""
        # Check for required layers
        model_layers = [module for name, module in self.model.named_modules()]
        
        # Test for convolutional layers
        conv_layers = [layer for layer in model_layers if isinstance(layer, torch.nn.Conv2d)]
        self.assertGreaterEqual(len(conv_layers), 3, "Model should have at least 3 Conv2d layers")
        
        # Test for batch normalization layers
        bn_layers = [layer for layer in model_layers if isinstance(layer, torch.nn.BatchNorm2d)]
        self.assertGreaterEqual(len(bn_layers), 3, "Model should have batch normalization layers")
        
        # Test for dropout
        dropout_layers = [layer for layer in model_layers if isinstance(layer, torch.nn.Dropout2d)]
        self.assertGreaterEqual(len(dropout_layers), 1, "Model should have dropout layer")

    # New Test 3: Test Forward Pass Values
    def test_forward_pass_values(self):
        """Test if forward pass produces valid values"""
        self.model.eval()
        x = torch.zeros(1, 1, 28, 28).to(self.device)  # Test with zero input
        output = self.model(x)
        
        # Test if output contains NaN
        self.assertFalse(torch.isnan(output).any(), 
                        "Model output contains NaN values")
        
        # Test if output contains Inf
        self.assertFalse(torch.isinf(output).any(), 
                        "Model output contains Inf values")
        
        # Test if output values are within reasonable range
        self.assertTrue((output <= 0).all(), 
                       "Log softmax output should be <= 0")
        
        # Test if output responds to different inputs
        x_ones = torch.ones(1, 1, 28, 28).to(self.device)
        output_ones = self.model(x_ones)
        self.assertFalse(torch.allclose(output, output_ones), 
                        "Model outputs identical values for different inputs")

if __name__ == '__main__':
    unittest.main() 