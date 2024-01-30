import torch
from torch.utils.data import Dataset
from pathlib import Path

class OpenPoseDataset(Dataset):
    def __init__(self, data_dir, data_type):
        self.poses = torch.load(Path(data_dir, f'poses_{data_type}.pt'))
        self.poses_missing = torch.load(Path(data_dir, f'poses_missing_{data_type}.pt'))
        self.data_dir = data_dir

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        pose = self.poses[idx]
        pose_missing = self.poses_missing[idx]
        return pose_missing, pose
    
def main():
    dataset = OpenPoseDataset('data')
    x, y = dataset[15000]
    print(x)
    print(y)

if __name__=='__main__':
    main()