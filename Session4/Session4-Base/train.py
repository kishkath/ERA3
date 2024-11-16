import torch
import torch.nn as nn
import torch.optim as optim
from model import FashionMNISTConvNet
from dataset import FashionMNISTDataset
from typing import Callable
from tqdm import tqdm
import logging

class TrainingManager:
    def __init__(self, callback: Callable):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FashionMNISTConvNet().to(self.device)
        self.callback = callback
        self.is_training = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train(self, epochs=10):
        self.is_training = True
        
        # Initialize dataset and optimizer
        dataset = FashionMNISTDataset()
        train_loader, _ = dataset.get_data_loaders()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        total_batches = len(train_loader)
        log_interval = total_batches // 6  # Log 6 times per epoch
        
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Total epochs: {epochs}, Total batches per epoch: {total_batches}")
        self.callback({'status': 'log', 'message': f"Training started on {self.device}"})

        for epoch in range(epochs):
            if not self.is_training:
                break
                
            self.model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            batch_count = 0
            
            # Create progress bar for each epoch
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (data, target) in enumerate(pbar):
                if not self.is_training:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                running_loss += current_loss
                epoch_loss += current_loss
                batch_count += 1
                
                # Calculate progress within epoch (0 to 1)
                progress = batch_idx / total_batches
                
                # Send real-time updates for every batch
                self.callback({
                    'status': 'progress',
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': current_loss,
                    'total_epochs': epochs,
                    'total_batches': total_batches,
                    'epoch_progress': progress
                })
                
                # Log every 1/6th of total batches
                if batch_idx % log_interval == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    log_msg = (
                        f'Epoch: {epoch+1}/{epochs} ({(progress*100):0.1f}%), '
                        f'Batch: {batch_idx+1}/{total_batches}, '
                        f'Loss: {current_loss:.4f}, '
                        f'Avg Loss: {avg_loss:.4f}'
                    )
                    self.logger.info(log_msg)
                    self.callback({'status': 'log', 'message': log_msg})
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{running_loss/(batch_idx+1):.4f}'
                })
            
            # Calculate and send epoch-level statistics
            avg_epoch_loss = epoch_loss / batch_count
            epoch_end_msg = f"Epoch {epoch+1}/{epochs} completed. Average loss: {avg_epoch_loss:.4f}"
            self.logger.info(epoch_end_msg)
            self.callback({
                'status': 'epoch_end',
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'total_epochs': epochs
            })

        # Save model if training completed normally
        if self.is_training:
            model_path = 'fashion_mnist_cnn.pth'
            torch.save(self.model.state_dict(), model_path)
            complete_msg = f"Training completed. Model saved to {model_path}"
            self.logger.info(complete_msg)
            self.callback({'status': 'log', 'message': complete_msg})
            self.callback({'status': 'completed'})
        
        self.is_training = False

    def stop_training(self):
        """Stop the training process gracefully"""
        self.is_training = False
        self.logger.info("Training stopped by user")
        self.callback({'status': 'stopped'})