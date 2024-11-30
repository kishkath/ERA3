import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

def predict_single_image(model, image_path, device):
    """
    Make prediction for a single image
    Args:
        model: trained PyTorch model
        image_path: path to the image file
        device: device to run inference on
    Returns:
        predicted class and probability
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        
    return pred.item(), prob[0][pred].item()

def batch_inference(model, test_loader, device):
    """
    Run inference on a batch of images
    Args:
        model: trained PyTorch model
        test_loader: DataLoader containing test data
        device: device to run inference on
    Returns:
        predictions and accuracy
    """
    model.eval()
    predictions = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.squeeze().cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return predictions, accuracy 