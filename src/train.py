from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import OpenPoseDataset
from model import LitAutoEncoder
from lightning.pytorch import loggers as pl_loggers
from paths import DATA_DIR, ROOT_DIR
from pathlib import Path


DATA_VERSION = 'v7'
DATA_DIR = Path(DATA_DIR, DATA_VERSION)

def main():
    autoencoder = LitAutoEncoder(dropout=0.5, weight_decay=0.001)
    dataset_train = OpenPoseDataset(DATA_DIR, 'train')
    dataset_valid = OpenPoseDataset(DATA_DIR, 'valid')
    dataset_test = OpenPoseDataset(DATA_DIR, 'test')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename=DATA_VERSION+'-{epoch}-{val_loss:.6f}'
    )

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    valid_loader = DataLoader(dataset_valid, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ROOT_DIR)
    trainer = L.Trainer(max_epochs=8, callbacks=[checkpoint_callback], logger=tb_logger)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(ckpt_path='best', dataloaders=test_loader)


if __name__=='__main__':
    main()