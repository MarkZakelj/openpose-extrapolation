import sys
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import OpenPoseDataset
from model import SkeletonExtrapolator
from vae import SkeletonVAE
from lightning.pytorch import loggers as pl_loggers
from paths import DATA_DIR, ROOT_DIR
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Please provide the data version and model type as a command-line argument.")
        return

    DATA_VERSION = sys.argv[1]
    model_type = sys.argv[2]
    print(f"Training with data version: {DATA_VERSION}")
    DATA_DIR_VERSIONED = Path(DATA_DIR, DATA_VERSION)

    extrapolator = None
    if model_type == 'vae':
        extrapolator = SkeletonVAE(input_dim=36, hidden_dim=32, latent_dim=4, learning_rate=1e-3, weight_decay=0.005)
    elif model_type == 'fcn':
        extrapolator = SkeletonExtrapolator(dropout=0.5, weight_decay=0.001)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    dataset_train = OpenPoseDataset(DATA_DIR_VERSIONED, 'train')
    dataset_valid = OpenPoseDataset(DATA_DIR_VERSIONED, 'valid')
    dataset_test = OpenPoseDataset(DATA_DIR_VERSIONED, 'test')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=8,
        filename=model_type+'-'+DATA_VERSION+'-{epoch}-{val_loss:.6f}'
    )

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    valid_loader = DataLoader(dataset_valid, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ROOT_DIR)
    trainer = L.Trainer(max_epochs=4, callbacks=[checkpoint_callback], logger=tb_logger, val_check_interval=0.2)
    trainer.fit(model=extrapolator, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(ckpt_path='best', dataloaders=test_loader)


if __name__=='__main__':
    main()