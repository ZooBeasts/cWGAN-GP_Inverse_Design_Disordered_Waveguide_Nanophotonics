from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from os.path import join
import torch
import numpy as np
from sklearn import preprocessing

Scaler = preprocessing.MinMaxScaler()

dataindex = 201

class MMIUnseenDataset(Dataset):

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
        # points = Scaler.fit_transform(points)
        points = torch.from_numpy(points).flatten(0)
        # points = torch.from_numpy(points)
        assert len(points) <= self.z_dim
        points = torch.hstack([points, torch.randn(self.z_dim - len(points))])
        points = points.reshape([self.z_dim, 1, 1])
        # the shape of points should be [Z_DIM, CHANNELS_IMG, FEATURES_GEN]

        return points, img, points21

    def __len__(self):
        return len(self.data)


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

if __name__ == "__main__":
    pass
