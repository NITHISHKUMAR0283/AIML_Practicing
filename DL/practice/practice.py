# import torch 
# from torchvision import datasets
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import torch.nn as nn

# class NeuralNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784,128)
#         self.fc2 = nn.Linear(128,64)
#         self.fc3 = nn.Linear(64,10)
#     def forward(self,x):
#         x = x.reshape(-1,784)
#         x = self.fc1(x)
#         x = self.fc2(x)        
#         x = self.fc3(x)
#         return x


# model = NeuralNet()



# transform = transforms.ToTensor()

# train_dataset = datasets.MNIST(
#     root="./data",
#     train=True,
#     transform = transform,
#     download = True
# )
# test_dataset = datasets.MNIST(
#     root = "./data",
#     train=False,
#     transform = transform,
#     download = True
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size = 32,
#     shuffle = True
# )

# image , label = next(iter(train_loader))

# output = model(image)
# print(output.shape)
# print(output[0])



