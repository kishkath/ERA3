import torch
import numpy as np
from model import FashionMNISTConvNet
from dataset import FashionMNISTDataset
import base64
import io
from PIL import Image
from torch.utils.data import DataLoader

class InferenceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FashionMNISTConvNet().to(self.device)
        self.model.load_state_dict(torch.load('fashion_mnist_cnn.pth'))
        self.model.eval()
        self.dataset = FashionMNISTDataset()

    def get_random_predictions(self, n_samples=15):
        # Create a new DataLoader with batch_size = n_samples to ensure we get exactly what we need
        _, test_dataset = self.dataset.get_raw_datasets()
        
        # Create a specific loader for inference
        test_loader = DataLoader(
            test_dataset,
            batch_size=n_samples,
            shuffle=True,  # Shuffle to get random samples
            drop_last=True  # Ensure we get exactly n_samples
        )
        
        # Get one batch of samples
        images, labels = next(iter(test_loader))
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            predictions = outputs.argmax(dim=1)
        
        # Convert images to base64 for sending to frontend
        image_data = []
        for img in images:
            img = (img * 0.5 + 0.5).numpy()  # Denormalize
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img[0], mode='L')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            image_data.append(img_str)
        
        # Convert numeric labels to class names
        pred_labels = [self.dataset.get_class_label(p.item()) for p in predictions]
        actual_labels = [self.dataset.get_class_label(l.item()) for l in labels]
        
        return {
            'images': image_data,
            'predictions': pred_labels,
            'actual_labels': actual_labels
        } 