import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

num_epochs = 51

Kernal1 = 8
Kernal2 = 6
Kernal3 = 4
Q = 3

class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        points = torch.tensor(self.annotations.iloc[index, 1:].tolist(), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, points


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image to range [-1, 1]
])

dataset = MyDataset('', '',
                    transform=transform)

dataloader = DataLoader(dataset, batch_size=85, shuffle=True)


# Define CNN model
class MyCNNModel(nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.out = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(256 * 5 * 5, 256),
            nn.Sigmoid(),
            nn.Linear(256, 50),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.out(x)
        return x


model = MyCNNModel().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                      len(dataloader), loss.item()))
        if epoch % 5 == 0:
            torch.save(model, 'cnn_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), 'CNN_test.pth')
#
# testdata = MyDataset('5e4testG.csv',
#                      'C:/Users/Administrator/Desktop/pythonProject/pr1/testcnn/',
#                      transform=transform)
# test_loader = DataLoader(testdata, batch_size=128, shuffle=False)
# switch to evaluation mode
# model.eval()
#
# # initialize variables for computing test loss and accuracy
# test_loss = 0.0
# num_correct = 0
# num_total = 0

# for i, (images, targets) in enumerate(test_loader):
#     # move images and targets to device
#     images = images.to(device)
#     targets = targets.to(device)
#
#     # forward pass
#     outputs = model(images)
#
#     # compute test loss
#     loss = criterion(outputs, targets)
#     test_loss += loss.item() * images.size(0)

    # compute accuracy
    # num_correct += torch.sum(torch.abs(outputs - targets) <= 0.1)
    # num_total += images.size(0)
    # if i % 10 == 0:
    #     print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1,
    #                                                               len(dataloader), loss.item()))

    # compute test loss and accuracy
# test_loss /= len(test_loader.dataset)
#     # test_acc = num_correct.double() / num_total
#     # acc = accuracy(model, test_x, test_y, 0.15)
#     # print("Accuracy on test data = %0.2f%%" % acc)
# print('Test Loss: {:.6f}'.format(test_loss))


# Save the model

