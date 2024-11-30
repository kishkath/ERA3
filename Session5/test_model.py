import unittest
import torch
from model import Net, get_summary
from dataset import get_data_loaders
from train import train

class TestModelConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = Net().to(cls.device)
        cls.train_loader, _ = get_data_loaders(batch_size=128)
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

if __name__ == '__main__':
    unittest.main() 