#!/usr/bin/env python3

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:24:53 2024

@author: saiful
"""
from torch import nn 
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
num_epochs =300
# Dataset Directory
normal_dir = "/data/saiful/pilot projects/brain_tumor_dataset/no/"
tumor_dir = "/data/saiful/pilot projects/brain_tumor_dataset/yes/"

class BrainMRIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def load_images(normal_dir, tumor_dir):
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.jpg')]
    tumor_images = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith('.jpg')]
    images = normal_images + tumor_images
    labels = [0] * len(normal_images) + [1] * len(tumor_images)  # 0 for normal, 1 for tumor
    return images, labels

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images and labels
images, labels = load_images(normal_dir, tumor_dir)

# Split into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create Dataset instances
train_dataset = BrainMRIDataset(train_images, train_labels, transformations)
test_dataset = BrainMRIDataset(test_images, test_labels, transformations)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        # Freeze all layers in the feature extractor during training
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Modify the classifier
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, 2)  # Binary classification

        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Network Initialization
model = ClassificationModel()
model.train()

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        total_correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        total_loss = 0
        total_correct = 0
        true_labels = []
        predictions = []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                true_labels.extend(labels.tolist())
                predictions.extend(predicted.tolist())

        valid_loss = total_loss / len(test_loader)
        valid_accuracy = total_correct / len(test_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy*100:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy*100:.2f}%")

    # Convert lists to numpy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Calculate metrics
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)

    # Print metrics
    print(f'== On test data ==')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Classification report
    report = classification_report(true_labels, predictions, digits=4)
    print('Classification report:\n', report)

# Remember to include the necessary imports at the beginning of your script.

train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs)
