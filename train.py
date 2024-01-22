from torch import utils
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import OpenPoseDataset
from model import LitAutoEncoder
from torch.utils.data import Subset
import torch

DATA_VERSION = 'v2'
DATA_DIR = f'data/{DATA_VERSION}'


def main():
    autoencoder = LitAutoEncoder(dropout=0.5)
    dataset = OpenPoseDataset(DATA_DIR)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename='{epoch}-{val_loss:.2f}'
    )

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    # Create indices for train, validation, test
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)

    trainer = L.Trainer(max_epochs=7, callbacks=[checkpoint_callback])
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(model=autoencoder, dataloaders=test_loader)


if __name__=='__main__':
    main()