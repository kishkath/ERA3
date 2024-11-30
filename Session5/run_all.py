import os
import torch
import torch.optim as optim
from torchvision import datasets
from PIL import Image
# import matplotlib.pyplot as plt

# Import all project modules
from model import Net, get_summary
from dataset import get_data_loaders
from train import train, test
from inference import predict_single_image, batch_inference

def setup_environment():
    """Setup CUDA and working directories"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    return device

def download_and_prepare_data():
    """Download MNIST dataset and prepare data loaders"""
    print("\n1. Downloading and preparing MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=128)
    print("✓ Data preparation completed")
    
    # Save a few examples for later inference
    examples = next(iter(test_loader))
    example_data, _ = examples
    # for i in range(5):
    #     img = example_data[i].squeeze().numpy()
    #     plt.imsave(f'results/example_{i}.png', img, cmap='gray')
    
    return train_loader, test_loader

def initialize_model(device):
    """Initialize and analyze the model"""
    print("\n2. Initializing model...")
    model = Net().to(device)
    
    # Get model summary and parameter count
    print("\nModel Summary:")
    total_params = get_summary(model)
    print(f"\nTotal parameters: {total_params:,}")
    
    if total_params >= 25000:
        raise ValueError(f"Model has {total_params:,} parameters, exceeding the limit of 25,000")
    
    print("✓ Model initialization completed")
    return model

def train_model(model, device, train_loader, test_loader):
    """Train the model and validate results"""
    print("\n3. Training model...")
    optimizer = optim.Adam(model.parameters())
    
    # Train for two epochs
    final_accuracy = 0
    for epoch in range(1, 3):
        print(f"\nEpoch {epoch}:")
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        final_accuracy = train_accuracy  # Store the last epoch's accuracy
    
    # Validate accuracy requirement
    if final_accuracy < 0.95:
        raise ValueError(f"Model achieved only {final_accuracy*100:.2f}% accuracy, "
                        f"below the required 95%")
    
    print(f"✓ Training completed with {final_accuracy*100:.2f}% accuracy")
    
    # Test the model
    print("\n4. Testing model...")
    test_accuracy = test(model, device, test_loader)
    print(f"✓ Testing completed with {test_accuracy*100:.2f}% accuracy")
    
    return final_accuracy, test_accuracy

def run_inference_examples(model, device):
    """Run inference on saved example images"""
    print("\n5. Running inference examples...")
    
    # Single image inference
    print("\nSingle image predictions:")
    for i in range(5):
        image_path = f'results/example_{i}.png'
        prediction, probability = predict_single_image(model, image_path, device)
        print(f"Example {i}: Predicted digit {prediction} with {probability*100:.2f}% confidence")

def save_results(model, train_accuracy, test_accuracy):
    """Save model and results"""
    print("\n6. Saving results...")
    
    # Save model
    torch.save(model.state_dict(), 'results/model.pth')
    
    # Save accuracies
    with open('results/results.txt', 'w') as f:
        f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    
    print("✓ Results saved to 'results' directory")

def main():
    """Main function to run all processes"""
    print("Starting complete MNIST CNN workflow...")
    
    # Setup
    device = setup_environment()
    
    try:
        # Data preparation
        train_loader, test_loader = download_and_prepare_data()
        
        # Model initialization
        model = initialize_model(device)
        
        # Training
        train_accuracy, test_accuracy = train_model(model, device, train_loader, test_loader)
        
        # Inference examples
        run_inference_examples(model, device)
        
        # Save results
        save_results(model, train_accuracy, test_accuracy)
        
        print("\n✅ All processes completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main() 