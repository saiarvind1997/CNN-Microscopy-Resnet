
"""
Code for Multiclass classification of Microscopy images using a CNN model trained by k-fold technique.

## The goal is to classify different images in the dataset finetuning for the best model architecture, hyper parameters and training procedure.

"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Define the path to your SEM dataset
data_dir = 'data_tmp'

# Define the transformation pipeline without normalization
transform   = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the SEM dataset and apply the transformation pipeline
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Define the number of folds for k-fold cross-validation
num_folds = 5
# Define the k-fold cross-validation iterator
kf        = KFold(n_splits=num_folds, shuffle=True)

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 32 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)  # Apply softmax function to output layer
        return x

# Iterate over the folds
for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')

    # Define the training and validation datasets for this fold
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    # Compute the mean and standard deviation values for each channel separately on the training set
    channel_means = np.zeros(3)
    channel_stds = np.zeros(3)
    for inputs, _ in train_dataset:
        channel_means += np.mean(inputs.numpy(), axis=(1,2))
        channel_stds += np.std(inputs.numpy(), axis=(1,2))
    num_train_samples = len(train_dataset)
    channel_means /= num_train_samples
    channel_stds /= num_train_samples

    # Print the mean and standard deviation values for each channel for this fold
    print('Channel means:', channel_means)
    print('Channel standard deviations:', channel_stds)

    # Define the transformation pipeline with normalization using the computed mean and std values
    normalize_transform = transforms.Normalize(mean=channel_means.tolist(), std=channel_stds.tolist())

    # Define the transformation pipeline with normalization for both training and validation sets
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize_transform
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize_transform
    ])

    # Define the dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the CNN model and the loss function
    model = Net()
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the CNN model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {} - Training loss: {:.6f}'.format(epoch+1, running_loss/len(train_loader)))

    # Evaluate the CNN model on the validation set and compute the F1 score
# Evaluate your model on validation set and calculate f1 score
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for inputs, labels in valid_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Collect true and predicted labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # Calculate f1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    print('F1 score:', f1)








