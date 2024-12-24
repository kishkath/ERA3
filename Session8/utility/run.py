import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

class Performance:

    def __init__(self, device, model, data,optimizer,criterion,l1_reg=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.test_loader = data[0], data[1]
        self.l1_reg = l1_reg

    def GetCorrectPredCount(self,pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss += loss.item()
            l1=0
            if self.l1_reg[0]==True:
                lambda_l1 = self.l1_reg[1]
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + lambda_l1*l1

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        train_acc.append(100 * correct / processed)
        train_losses.append(train_loss / len(self.train_loader))

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss

                correct += self.GetCorrectPredCount(output, target)

        test_loss /= len(self.test_loader.dataset)
        test_acc.append(100. * correct / len(self.test_loader.dataset))
        test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

def scores():
    return train_losses,train_acc,test_losses,test_acc
