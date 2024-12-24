import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Plots:
    def __init__(self,mis_classify_details=None ,num_images=None, loaded_data=None):
        self.num_images = num_images
        self.loaded_data = loaded_data
        self.mis_classify = mis_classify_details


    def plot_images(self):
        batch_data, batch_label = next(iter(self.loaded_data))
        fig = plt.figure()
        if self.num_images % 2 != 0:
            self.num_images -= 1
        self.num_rows = self.num_images // 4

        fig = plt.figure(figsize=(15, 7))
        counter = 0
        for i in range(self.num_images):
            sub = fig.add_subplot(self.num_rows, 4, i + 1)
            im2display = (np.squeeze(batch_data[i].permute(2, 1, 0)))
            sub.imshow(im2display.cpu().numpy())
            sub.set_title(batch_label[i].item())
            sub.axis('off')

        plt.tight_layout()
        plt.axis('off')
        plt.show()



    def mis_classified(self):
        model, testloader, device, images_needed = self.mis_classify
        storing_images = []
        storing_predicted_labels = []
        storing_target_labels = []
        if images_needed == None:
            images_needed = random.choice([10, 20])
        with torch.no_grad():
            model.eval()
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                for idx in range(len(pred)):
                    if pred[idx] != target[idx]:
                        storing_images.append(data[idx])
                        storing_predicted_labels.append(pred[idx])
                        storing_target_labels.append(target[idx])

        fig = plt.figure(figsize=(20, 14))

        if images_needed % 2 != 0:
            images_needed -= 1  # It becomes even so plotting would be good.

        num_rows = 0
        plots_per_row = 0
        if images_needed <= 10:
            num_rows = 5
            plots_per_row = images_needed // num_rows
        elif 22 > images_needed > 10:
            num_rows = 4
            plots_per_row = images_needed // num_rows
        elif images_needed > 20:
            num_rows = 10
            plots_per_row = images_needed // num_rows

        for i in range(images_needed):
            sub = fig.add_subplot(num_rows, plots_per_row, i + 1)
            im2display = (np.squeeze(storing_images[i].permute(2, 1, 0)))
            sub.imshow(im2display.cpu().numpy())
            sub.set_title(
                f"Predicted as: {classes[storing_predicted_labels[i]]} \n But, Actual is: {classes[storing_target_labels[i]]}")
        plt.tight_layout()
        plt.show()


def plot_metrics(metrics):
    if metrics is None:
        print("Please provide the metric values, unable to view them!")
        sys.exit(0)
    else:
        train_losses, train_acc, test_losses, test_acc = metrics
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")

