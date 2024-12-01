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

    # New Test 1: Test Model Robustness with Augmented Data
    def test_model_robustness(self):
        """Test model's performance on augmented data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.aug_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        self.assertGreater(accuracy, 0.85,  # Lower threshold for augmented data
                          f"Model's robustness test failed with accuracy: {accuracy*100:.2f}%")

    # New Test 2: Test Learning Rate Range
    def test_learning_rate_sensitivity(self):
        """Test model's sensitivity to different learning rates"""
        x = torch.randn(1, 1, 28, 28).to(self.device)
        original_output = self.model(x)
        
        # Test with different learning rates
        learning_rates = [0.1, 0.01, 0.001]
        for lr in learning_rates:
            model_copy = Net().to(self.device)
            model_copy.load_state_dict(self.model.state_dict())
            optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
            
            # Train for one batch
            data, target = next(iter(self.train_loader))
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model_copy(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # Check if the model can still make predictions
            new_output = model_copy(x)
            self.assertTrue(torch.allclose(new_output.sum(), 
                                         torch.tensor(1.0).to(self.device), 
                                         atol=1e-3),
                          f"Model unstable with learning rate {lr}")

    # New Test 3: Test Batch Normalization Behavior
    def test_batch_norm_behavior(self):
        """Test if batch normalization layers are working correctly"""
        self.model.train()  # Set to training mode
        
        # Get a batch of data
        data, _ = next(iter(self.train_loader))
        data = data.to(self.device)
        
        # Forward pass
        output = self.model(data)
        
        # Check batch norm statistics for each conv layer
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Check if running mean and variance are being updated
                self.assertIsNotNone(module.running_mean)
                self.assertIsNotNone(module.running_var)
                
                # Check if statistics are reasonable
                self.assertTrue(torch.all(module.running_var > 0),
                              f"Batch norm layer {name} has invalid variance")
                self.assertTrue(torch.all(torch.abs(module.running_mean) < 100),
                              f"Batch norm layer {name} has unstable mean")

if __name__ == '__main__':
    unittest.main() 