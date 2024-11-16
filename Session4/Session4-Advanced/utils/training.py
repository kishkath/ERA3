import torch
import torch.nn as nn
import torch.optim as optim
from models.model import create_model
from utils.data_loader import get_dataset
import json
from flask import Response
import traceback
import os

def train_model(model_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create model and move to device
        model = create_model(model_params['num_layers']).to(device)
        
        # Get dataset
        train_loader, test_loader = get_dataset(
            dataset_name=model_params['dataset'],
            batch_size=model_params['batch_size']
        )
        
        if train_loader is None or test_loader is None:
            return Response("Error loading dataset", status=500)
        
        # Setup optimizer
        if model_params['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(model.parameters())
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        def generate_training_updates():
            try:
                for epoch in range(model_params['epochs']):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for batch_idx, (data, target) in enumerate(train_loader):
                        try:
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
                        except Exception as e:
                            print(f"Batch error: {str(e)}")
                            continue
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data, target in test_loader:
                            try:
                                data, target = data.to(device), target.to(device)
                                output = model(data)
                                val_loss += criterion(output, target).item()
                                _, predicted = output.max(1)
                                total += target.size(0)
                                correct += predicted.eq(target).sum().item()
                            except Exception as e:
                                print(f"Validation batch error: {str(e)}")
                                continue
                    
                    val_accuracy = 100. * correct / total
                    update = {
                        'epoch': epoch,
                        'val_loss': val_loss / len(test_loader),
                        'val_accuracy': val_accuracy
                    }
                    yield f"data: {json.dumps(update)}\n\n"
                
                # Save model after training
                model_path = f"model_{model_params['model_id']}.pth"
                torch.save(model.state_dict(), model_path)
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                
            except Exception as e:
                print(f"Training error: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate_training_updates(), mimetype='text/event-stream')
    
    except Exception as e:
        print(f"Setup error: {str(e)}")
        traceback.print_exc()
        return Response(f"data: {json.dumps({'error': str(e)})}\n\n", 
                       mimetype='text/event-stream')

def generate_predictions(model_id, num_predictions, dataset):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = f"model_{model_id}.pth"
        
        if not os.path.exists(model_path):
            return {'error': 'Model not found'}
        
        # Load the model
        model = create_model(4)  # Default to 4 layers for prediction
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        # Get test dataset
        _, test_loader = get_dataset(dataset_name=dataset, batch_size=num_predictions)
        
        if test_loader is None:
            return {'error': 'Could not load test dataset'}
        
        # Get first batch for predictions
        data, targets = next(iter(test_loader))
        data, targets = data.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            _, predicted = outputs.max(1)
        
        # Convert images to base64 strings
        import base64
        from io import BytesIO
        from PIL import Image
        
        images = []
        for img in data.cpu():
            # Normalize image to 0-255 range
            img = img.squeeze().numpy() * 255
            img = img.astype('uint8')
            # Convert to PIL Image
            pil_img = Image.fromarray(img)
            # Save to base64 string
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            images.append(img_str)
        
        return {
            'predictions': predicted.cpu().tolist(),
            'actual': targets.cpu().tolist(),
            'images': images
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return {'error': str(e)} 