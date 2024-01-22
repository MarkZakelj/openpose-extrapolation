import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
from einops import rearrange


class OpenPoseDataset(Dataset):
    def __init__(self, data_dir):
        self.poses = torch.load(os.path.join(data_dir, 'poses.pt'))
        self.poses_missing = torch.load(os.path.join(data_dir, 'poses_missing.pt'))
        self.ratios = torch.load(os.path.join(data_dir, 'ratios.pt'))
        self.data_dir = data_dir

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        pose = self.poses[idx]
        pose_missing = self.poses_missing[idx]
        ratio = torch.tensor([self.ratios[idx]])
        return torch.cat((pose_missing, ratio)), torch.cat((pose, ratio))
    
def main():
    dataset = OpenPoseDataset('data')
    x, y = dataset[15000]
    print(x)
    print(y)

if __name__=='__main__':
    main()