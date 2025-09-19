# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import cv2
from torchvision.datasets import ImageFolder

# %%
print(os.getcwd())
file_path = "../data/archive/"
dataframe = pd.read_csv(file_path+"Test.csv")
dataframe.head()

# %%
dataframe.shape

# %%
image = cv2.imread(r"C:\Users\rahul\OneDrive\Desktop\Rahul\Study\Projects\traffic-sign-recognition\data\archive\Train\0\00000_00000_00025.png")
image.shape
plt.imshow(image)

# %% [markdown]
# Using Imagefolder for labeling

# %%
# applying transformations
transform_pipeline = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# train dataset and loader
train_dataset = ImageFolder(file_path+"Train",
                            transform=transform_pipeline)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
type(train_dataset)
type(train_loader)

# %% [markdown]
# Dataset for test

# %%
class TrafficSignDataset(Dataset):
    def __init__(self, file_path, csv_path, transform=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.csv_data = pd.read_csv(csv_path)

    def __len__(self):
        return len(os.listdir(self.file_path+"Test"))
    
    def __getitem__(self, idx):
        img_name = os.path.basename(self.csv_data.iloc[idx, -1])
        img_path = os.path.join(self.file_path, img_name)
        label = int(self.csv_data.iloc[idx, -2])
        if self.transform:
            image = self.transform(image)
        return image, label

test_dataset = TrafficSignDataset(file_path+"Test",file_path+"Test.csv", transform=transform_pipeline)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# %% [markdown]
# CNN model

# %%
class CNNModel(nn.Module):
    def __init__(self, num_classes=43):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# %% [markdown]
# Hyperparameters

# %%
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# %% [markdown]
# Model instance

# %%
model = CNNModel(num_classes=43)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# %% [markdown]
# Training loop

# %%
# training loop
train = False

if train:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
else:
    model_path = "../models/traffic_sign_cnn.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

# %%
model_path = "../models/traffic_sign_cnn.pth"
torch.save(model.state_dict(), model_path)

# %% [markdown]
# Model evaluation

# %%
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")



