import torch
import torch.optim as optim
from model import Net, get_summary
from dataset_augmented import get_augmented_data_loaders
from train import train, test
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory for augmented samples
    os.makedirs('augmented_samples', exist_ok=True)
    
    # Initialize model
    model = Net().to(device)
    
    # Print model summary and parameter count
    total_params = get_summary(model)
    print(f'Total parameters: {total_params}')
    
    # Check if model meets parameter constraint
    if total_params >= 25000:
        raise ValueError(f'Model has {total_params} parameters, which exceeds the limit of 25,000')
    
    # Get data loaders with augmentation
    print("\nPreparing data loaders and saving augmented samples...")
    train_loader, test_loader = get_augmented_data_loaders(batch_size=128, save_samples=True)
    print("✓ Data preparation completed. Check 'augmented_samples' directory for sample images.")
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train for two epochs
    print("\nTraining for 2 epochs...")
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