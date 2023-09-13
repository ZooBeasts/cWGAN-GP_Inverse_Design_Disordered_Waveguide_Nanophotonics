from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from os.path import join
import torch
import numpy as np
from sklearn import preprocessing

Scaler = preprocessing.MinMaxScaler()

# our spectra data has 200 points, where i start at 1 so index is 201
dataindex =

class MMIUnseenDataset(Dataset):
    """
    MMIUnseenDataset is dataset for Unseenprediction.py
    where load the unseen data
    """

    def __init__(self, z_dim,points_path):
        self.data = pd.read_csv(points_path,header=None).to_numpy()
        self.z_dim = z_dim

    def __getitem__(self,index):
        item = self.data[index]
        # print(item)
        # print(item.shape)
        # points = item[0:dataindex-1].astype(np.float64)
        points = torch.from_numpy(item.astype(np.float64))
        points = torch.hstack([points, torch.randn(self.z_dim - len(points))])
        points = points.reshape([self.z_dim, 1, 1])
        # print(points.shape)
        return points



class MMIDataset(Dataset):
    """
    MMIDataset is for main to train
    and point refers to original 200 data points
    while points21 refers original 200 data points but for gradient penalty
    """

    def __init__(self, img_size, z_dim, points_path, img_folder):
        self.data = pd.read_csv(points_path, header=0, index_col=None).to_numpy()
        # self.data = pd.read_csv(points_path, header=0).to_numpy()
        self.img_folder = img_folder
        self.img_size = img_size
        self.z_dim = z_dim

    def __getitem__(self, index):
        item = self.data[index]
        img = cv2.imread(self.img_folder + '\\{}.png'.format(item[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))[:, :, np.newaxis]
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        points21 = item[1:dataindex].astype(np.float64).reshape(-1, 1)
        # points21 = item[1:dataindex].astype(np.float64)
        points21 = Scaler.fit_transform(points21)
        points21 = torch.from_numpy(points21).flatten(0)
        # points21 = torch.from_numpy(points21)

        points = item[1:dataindex].astype(np.float64).reshape(-1,1)
        # points = item[1:dataindex].astype(np.float64)
        points = Scaler.fit_transform(points)
        points = torch.from_numpy(points).flatten(0)
        # points = torch.from_numpy(points)
        assert len(points) <= self.z_dim
        points = torch.hstack([points, torch.randn(self.z_dim - len(points))])
        points = points.reshape([self.z_dim, 1, 1])
        # the shape of points should be [Z_DIM, CHANNELS_IMG, FEATURES_GEN]

        return points, img, points21

    def __len__(self):
        return len(self.data)

# remember to write down the file_path, image_path
def get_loader(
        img_size,
        batch_size,
        z_dim,
        points_path='',
        img_folder='',
        shuffle=True,
):
    return DataLoader(MMIDataset(img_size, z_dim, points_path, img_folder),
                      batch_size=batch_size, shuffle=shuffle)

#text onehot plz ignore
# class Condition(nn.Module):
#     def __init__(self, alpha: float):
#         super().__init__()
#
#         # From one-hot encoding to features: 21 => 784
#         self.fc = nn.Sequential(
#             nn.Linear(21, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(alpha))
#
#     def forward(self, labels: torch.Tensor):
#         # One-hot encode labels
#         x = F.one_hot(labels, num_classes=21)
#
#         # From Long to Float
#         x = x.float()
#
#         # To feature vectors
#         return self.fc(x)
#
#
# # Reshape helper
# class Reshape(nn.Module):
#     def __init__(self, *shape):
#         super().__init__()
#
#         self.shape = shape
#
#     def forward(self, x):
#         return x.reshape(-1, *self.shape)
