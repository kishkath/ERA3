from flask import Flask, render_template, jsonify, request, Response
from models.model import create_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

@app.route('/train', methods=['POST'])
def train():
    config = request.json
    model_id = config['model_id']
    start_channels = int(config['start_channels'])
    epochs = int(config['epochs'])
    optimizer_name = config['optimizer']
    batch_size = int(config['batch_size'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(start_channels).to(device)
    
    # Configure optimizer based on selection
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters())
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loaders(batch_size=batch_size)
    
    def generate_training_updates():
        for epoch in range(epochs):  # Use configured epochs
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % (len(train_loader) // 5) == 0:
                    accuracy = 100. * correct / total
                    update = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': running_loss / (batch_idx + 1),
                        'accuracy': accuracy
                    }
                    yield f"data: {json.dumps(update)}\n\n"
    
    return Response(generate_training_updates(), mimetype='text/event-stream')

@app.route('/predict', methods=['POST'])
def predict():
    config = request.json
    model_id = config['model_id']
    num_predictions = config['num_predictions']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(start_channels=16).to(device)  # Default starting channels
    
    try:
        # Load the trained model
        model.load_state_dict(torch.load(f'model_{model_id}.pth'))
        model.eval()
        
        # Get test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=num_predictions, shuffle=True)
        
        # Get one batch of images
        images, targets = next(iter(test_loader))
        images, targets = images.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Convert images to base64
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        image_list = []
        for img in images.cpu():
            img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_list.append(img_str)
        
        return jsonify({
            'predictions': predicted.cpu().tolist(),
            'actual': targets.cpu().tolist(),
            'images': image_list
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 