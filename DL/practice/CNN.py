import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root = "dataset/PetImages/",
    transform = transform
)
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_dataset, test_dataset = random_split(
    dataset,[train_size,test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle  = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print(len(dataset))
print(dataset.classes)