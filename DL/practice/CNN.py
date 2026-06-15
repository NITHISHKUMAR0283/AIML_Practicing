import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="dataset/PetImages/",
    transform=transform
)
total_count = len(dataset)
train_size = int(0.7 * total_count)
val_size = int(0.15 * total_count)
test_size = total_count - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32*32*32,128)
        self.fc2 = nn.Linear(128,2)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
model = CatDogCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30  
best_val_loss = float('inf')
patience = 3  
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss= 0
    with torch.no_grad():
      for images,labels in val_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        val_loss += loss.item()
      
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        torch.save(model.state_dict(), 'best_model.pth')
    else:
      patience_counter+=1
      if patience_counter>=patience:
            print(f" Early stopping triggered! The ideal epoch count was around Epoch {epoch + 1 - patience}.")
            break

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")