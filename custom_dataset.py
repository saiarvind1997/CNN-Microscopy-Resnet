
"""
Code for creating a custom dataset manually for the task of Binary Classification.

L7 in data_tmp corresponds to Biological Images
L9 in data_tmp corresponds to Fibres Data

## The goal is to classify create the dataset
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root,classes, transform=None, train=True):
        self.root = root

        self.transform  = transform
        self.train      = train
        self.images     = []
        self.labels     = []
        
        # Define classes and labels
        
        classes = ['Biological', 'Fibres']
        labels  = {'Biological': 0, 'Fibres': 1} 
        
        # Load images and labels
        
        if self.train:
            for class_name in classes:    
                
                for label_name in os.listdir("./data_tmp/" + class_name) :
                    label = labels[class_name]
                    label_prefix = label_name.split('_')[0]
                    if label_prefix == 'L7':
                        prefix_label = 0
                    elif label_prefix == 'L9':
                        prefix_label = 1
                    else:
                        continue
                    for file_name in os.listdir("./data_tmp/" +  class_name + "/" + label_name):
                        
                        image_path = "./data/" + class_name + "/" +  label_name + "/" + file_name
                        self.images.append(image_path)
                        self.labels.append((label, prefix_label))
        else:
            for class_name in classes:
                for label_name in os.listdir(os.path.join("./data", class_name)):
                    label = labels[class_name]
                    label_prefix = label_name.split('_')[0]
                    if label_prefix == 'L7':
                        prefix_label = 0
                    elif label_prefix == 'L9':
                        prefix_label = 1
                    else:
                        continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image and apply transforms
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

