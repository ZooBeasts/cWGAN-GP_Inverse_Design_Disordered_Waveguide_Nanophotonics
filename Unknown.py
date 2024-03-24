import torch
from torch import nn
import cv2
from Dataloader import MMIUnseenDataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load the saved trained Generator info
#model_path = r'C:/Users/Administrator/Desktop/pythonProject/pr1/logs/WGAN77/netG230.pt'
model_path = r'E:/50Kdatasave5/netG400.pt'

# Load the dataset
dataset = MMIUnseenDataset(
                     z_dim=250,
                     points_path='C:/Users/Administrator/Desktop/pythonProject/pr1/unseen1.csv',
                     )

# Output the results path & load the data into Generator
results_folder = r'C:\Users\Administrator\Desktop\pythonProject\pr1\ddd1'
gen = torch.load(model_path)
gen = gen.to(device)
gen = gen.eval()

# Generate the image array from given dataset
def predict(net: nn.Module, points):
    return net(points).squeeze(0).squeeze(0).cpu().detach().numpy()

# Generate the desired number of results and save to path
# 0 means to data 1st row
# 40000 means last row in dataset
stop_p = 1000
i = 0
for p in dataset:
    if i >= stop_p:
        break
    data = p.to(device, dtype=torch.float).unsqueeze(0)
    img_out = predict(gen, data)
    img = (img_out + 1) / 2
    img = np.round(255 * img)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    cv2.imwrite(results_folder + '\\' + 'map200_' + str(i+1) + '.png', img)
    i += 1





