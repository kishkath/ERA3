import torch
import torch.optim as optim
from model import Net, get_summary
from dataset import get_data_loaders
from train import train, test

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = Net().to(device)
    
    # Print model summary and parameter count
    total_params = get_summary(model)
    print(f'Total parameters: {total_params}')
    
    # Check if model meets parameter constraint
    if total_params >= 25000:
        raise ValueError(f'Model has {total_params} parameters, which exceeds the limit of 25,000')
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train for two epochs
    print("Training for 2 epochs...")
    final_accuracy = 0
    for epoch in range(1, 3):
        print(f"\nEpoch {epoch}:")
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        final_accuracy = train_accuracy  # Store the last epoch's accuracy

    # Test the model
    test_accuracy = test(model, device, test_loader)
    print(f'Test accuracy: {test_accuracy*100:.2f}%')

    # Check if model meets accuracy constraint
    if final_accuracy < 0.95:
        raise ValueError(f'Model achieved {final_accuracy*100:.2f}% accuracy, '
                        f'which is below the required 95%')
    
    print(f'Training completed successfully!')
    print(f'Final training accuracy: {final_accuracy*100:.2f}%')

if __name__ == '__main__':
    main() 