import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,x):
        x = x.reshape(-1,784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))        
        x = self.fc3(x)
        return x


model = NeuralNet()



transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform = transform,
    download = True
)
test_dataset = datasets.MNIST(
    root = "./data",
    train=False,
    transform = transform,
    download = True
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)




criterian = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
num_epoch = 3
for epoch in range(num_epoch):
    for image,label in train_loader:
        output = model(image)
        optimizer.zero_grad()
        loss = criterian(output,label)
        loss.backward()
        optimizer.step()
        
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle  = True
)
correct = 0
total = 0
with torch.no_grad():
    for image , label in test_loader:
        output = model(image)
        _,prediction = torch.max(output,1)
        total += label.size(0)
        correct += (prediction==label).sum().item()
accuracy = 100*correct/total
print(accuracy)