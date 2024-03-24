import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch import nn
import numpy as np


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_path = 'CNN1e0tt13.pth'
model_path = 'nek130.pt'

# model = MyCNNModel()
# model.load_state_dict(torch.load(model_path))
print('Starting load model')
# Cnn = MyCNNModel()
# Cnn.load_state_dict(torch.load(model_path))
Cnn = torch.load(model_path)
Cnn = Cnn.to(device)
print('Loaded')
Cnn = Cnn.eval()

img_path = 'testfull1'


def process_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(img).unsqueeze(0)
    return img


# Define the path to the CSV file containing the real values
real_values_data = '100testset.csv'

predictions = []
file_names = []
real_values = []

for filename in os.listdir(img_path):
    file_path = os.path.join(img_path, filename)
    image_tensor = process_image(file_path).to(device)
    with torch.no_grad():
        output = Cnn(image_tensor).squeeze(0).cpu().detach().numpy()
        predictions.append(output)
        file_names.append(filename)
        real_values_array = pd.read_csv(real_values_data, header=None)
        real_values.append(real_values_array.values)




selected_predictions = []
selected_file_names = []

for i, (prediction, real_value) in enumerate(zip(predictions, real_values)):
    if (prediction >= real_value*0.8).all():
        selected_predictions.append(prediction)
        selected_file_names.append(file_names[i])



# Plot selected predictions
wl = np.arange(1500, 1551, 1)
for i, prediction in enumerate(selected_predictions):
    plt.plot(wl, prediction, '--', label=f'Selected Prediction for {selected_file_names[i]}')
    print(selected_file_names[i])
    for j, real in enumerate(real_values):
        if selected_file_names[i] == file_names[j]:
            plt.plot(wl, real[i], label=f'Real Value for {file_names[j]}')
            print(file_names[j])


# plt.xlabel('Wavelength nm')
# plt.ylabel('Transmission')
# plt.title('Selected Predictions vs Real Values')
# plt.legend()
plt.show()



