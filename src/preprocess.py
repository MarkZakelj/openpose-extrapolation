import pandas as pd
import torch
from einops import rearrange
import numpy as np
import math
import os
import random
from pathlib import Path
from paths import DATA_DIR

VERSION = 'v7'
random.seed(42)
torch.manual_seed(123)

NECK = 1
def scale_pose(poses, scale):
    poses = poses.clone()
    meanx = poses[:, :, 0].mean(dim=1)
    meany = poses[:, :, 1].mean(dim=1)
    poses[:, :, 0] -= meanx.unsqueeze(1)
    poses[:, :, 1] -= meany.unsqueeze(1)
    poses[:, :, 0] *= scale[0]
    poses[:, :, 1] *= scale[1]
    poses[:, :, 0] += meanx.unsqueeze(1)
    poses[:, :, 1] += meany.unsqueeze(1)
    return poses

def calculate_relative_points(poses):
    res = poses.clone()
    res -= poses[:, NECK:NECK+1, :] # subtract neck
    return res

df = pd.read_csv(Path(DATA_DIR, 'poses.csv'))
df = df[~(df['class'].isin(['squat']))] # remove squats
df = df.drop('class', axis=1) # remove class column
df = df[~(df == 0).any(axis=1)] # remove rows with any zeros - undefined ponints
poses = torch.tensor(df.values, dtype=torch.float32) # convert to tensor
poses = rearrange(poses, 'n (ps p) -> n ps p', p=2) # rearrange to (n, 18, 2)
poses = scale_pose(poses, (1.75, 1.0)) # fix x-axis points
poses = calculate_relative_points(poses) # change to relative to NECK

# augument the dataset with y-axis flipping
p1 = poses.clone()
p1[:, :, 0] *= -1
poses_flipped = torch.cat([poses, p1], dim=0)

# augument the dataset with scaling from 0.05 to 3.55
get_scale = lambda : (random.betavariate(alpha=0.9, beta=1.1)) * 3.5 + 0.05
REPETITIONS = 2
res = []
for n in range(REPETITIONS):
    p1 = poses_flipped.clone()
    scale = torch.tensor([get_scale() for _ in range(p1.shape[0])], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    p1 *= scale
    res.append(p1)
poses_scaled = torch.cat(res, dim=0)

# augument the dataset with x and y axis rotation up to +- 45 degrees
# this matrix represents y-axis rotation followed by x-axis rotation
def rotate_both_axis(thetax: float, thetay: float):
    t = [
        [math.cos(thetay), math.sin(thetay) * math.sin(thetax)],
        [0, math.cos(thetax)]
    ]
    return t
ROTATION_REPETITIONS = 2
get_rotationy = lambda: (random.betavariate(alpha=2.5, beta=2.5) - 0.5) * np.pi / 2
get_rotationx = lambda: (random.betavariate(alpha=7, beta=7) - 0.5) * np.pi / 2
res = []
for n in range(ROTATION_REPETITIONS):
    matrices = torch.tensor([rotate_both_axis(get_rotationx(), get_rotationy()) for _ in range(poses_scaled.shape[0])], dtype=torch.float32)
    p1 = poses_scaled.clone()
    p1 = torch.matmul(p1, matrices)
    res.append(p1)
poses_scaled = torch.cat(res, dim=0)

# augument dataset based on x-scaling
get_scale = lambda : (random.betavariate(alpha=2, beta=2)) + 0.5
REPETITIONS = 1
res = []
for n in range(REPETITIONS):
    p1 = poses_scaled.clone()
    scale = torch.tensor([get_scale() for _ in range(p1.shape[0])], dtype=torch.float32).reshape(-1, 1)
    p1[:, :, 0] *= scale
    res.append(p1)
poses_scaled = torch.cat(res, dim=0)

# # add image ratios
nposes = poses_scaled.shape[0]
ratios = [1344/768, 1.0, 768/1344]
res_scaled = []
for ratio in ratios:
    p1 = poses_scaled.clone()
    p1[:, :, 0] *= ratio
    res_scaled.append(p1)
poses_scaled = torch.cat(res_scaled, dim=0)

# augument the dataset with missing points
missing_conf = [
    [],
    [0],
    [10, 13],
    [9, 10, 12, 13],
    [8, 9, 10, 11, 12, 13],
    [4,7,8,9,10,11,12,13],
    [4,7,9,10,12,13],
    [3,4,6,7,8,9,10,11,12,13],
    [7,9,10,12,13],
    [4,9,10,12,13],
    [7],
    [4],
    [6,7],
    [3,4],
    [3,4,6,7],
    [10],
    [13],
    [16],
    [17],
    [14,16],
    [15,17],
    [14,15,16,17],
    [0, 4, 7, 14, 15]
]
res_missing = []
res_scaled = []
for conf in missing_conf:
    # missing
    p1 = poses_scaled.clone()
    p1[:, conf, :] = -10.0
    res_missing.append(p1)
    # non missing
    p2 = poses_scaled.clone()
    res_scaled.append(p2)
poses_missing = torch.cat(res_missing, dim=0)
poses_scaled = torch.cat(res_scaled, dim=0)

# rearrange from (n, 18, 2) to (n, 36)
poses_scaled_rearranged = rearrange(poses_scaled, 'n ps p -> n (ps p)')
poses_missing_rearranged = rearrange(poses_missing, 'n ps p -> n (ps p)')

# check if ratios and points are of the same length and 
print(poses_scaled_rearranged.shape, poses_missing_rearranged.shape)
assert poses_scaled_rearranged.shape[0] == poses_missing_rearranged.shape[0]
assert poses_scaled_rearranged.shape[1] == poses_missing_rearranged.shape[1] == 36
# if missing and scaled are the same except for the missing points
mask = poses_missing_rearranged != -10.0
assert (poses_missing_rearranged[mask] == poses_scaled_rearranged[mask]).all()

data_dir = Path(DATA_DIR, VERSION)
data_dir.mkdir(exist_ok=True)

# split into train, validation and test
total_size = poses_missing_rearranged.shape[0]
train_size = int(0.7 * total_size)
valid_size = int(0.15 * total_size)

# Create indices for train, validation, test
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
valid_indices = indices[train_size:train_size+valid_size]
test_indices = indices[train_size+valid_size:]

# save shuffled data
torch.save(poses_scaled_rearranged[train_indices], Path(data_dir, 'poses_train.pt'))
torch.save(poses_scaled_rearranged[valid_indices], Path(data_dir, 'poses_valid.pt'))
torch.save(poses_scaled_rearranged[test_indices], Path(data_dir, 'poses_test.pt'))
torch.save(poses_missing_rearranged[train_indices], Path(data_dir, 'poses_missing_train.pt'))
torch.save(poses_missing_rearranged[valid_indices], Path(data_dir, 'poses_missing_valid.pt'))
torch.save(poses_missing_rearranged[test_indices], Path(data_dir, 'poses_missing_test.pt'))
