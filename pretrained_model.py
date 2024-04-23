"""
Code for Multiclass classification of Microscopy images using a pre-trained Resnet model.

## The goal is to classify different images in the dataset finetuning.

"""




import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import os

# Define the path to your SEM dataset
data_dir = 'data_tmp'

# Define the transformation pipeline without normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the SEM dataset and apply the transformation pipeline
dataset   = datasets.ImageFolder(data_dir, transform=transform)
# Define the number of folds for k-fold cross-validation
num_folds = 2
# Define the k-fold cross-validation iterator
kf        = KFold(n_splits=num_folds, shuffle=True)

# Define the ResNet model
# Loading the Pretrined Resnet model for finetuning for the task of image classification.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.softmax(x, dim=1)  # Apply softmax function to output layer
        return x

# Finetuning the Resnet by k-fold training technique.
# Iterate over the folds
for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold+1}')

    # Define the training and validation datasets for this fold
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    # Compute the mean and standard deviation values for each channel separately on the training set
    channel_means = np.zeros(3)
    channel_stds  = np.zeros(3)
    for inputs, _ in train_dataset:
        channel_means += np.mean(inputs.numpy(), axis=(1,2))
        channel_stds  += np.std(inputs.numpy(), axis=(1,2))
    num_train_samples  = len(train_dataset)
    channel_means     /= num_train_samples
    channel_stds      /= num_train_samples

    # Print the mean and standard deviation values for each channel for this fold
    print('Channel means:', channel_means)
    print('Channel standard deviations:', channel_stds)

    # Define the transformation pipeline with normalization using the computed mean and std values
    normalize_transform = transforms.Normalize(mean=channel_means.tolist(), std=channel_stds.tolist())

    # Define the transformation pipeline with normalization for both training and validation sets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])

    # Define the dataloaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the ResNet model and the loss function
    model     = Net()

    # Define the loss function as cross entropy loss for Multi class 
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #####################################################################
    ### Load epochs
    if os.path.isfile('resnet_fold.pt'):
        # Load the saved model
        checkpoint = torch.load('resnet_fold.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Loaded checkpoint '{}' (epoch {})".format('resnet_fold.pt', start_epoch))
    else:
        start_epoch = 0
        print("No checkpoint found at '{}'".format('resnet_fold.pt'))
        #########################################################################################
    # Finetuning the ResNet model
    num_epochs = 20
    epoch      = 0
    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {} - Training loss: {:.6f}'.format(epoch+1, running_loss/len(train_loader)))
        epoch+=1
    

#####################################################################
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

    # torch.save(model.state_dict(), f'resnet_fold{fold+1}.pt')
    # Save the model state at the end of each epoch
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }
    torch.save(checkpoint, f'resnet_fold.pt')

  
  
