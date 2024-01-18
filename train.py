from torch import utils
import lightning as L
from dataset import OpenPoseDataset
from model import LitAutoEncoder




def main():
    # init the autoencoder
    autoencoder = LitAutoEncoder()
    dataset = OpenPoseDataset('data')
    train_loader = utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=4, persistent_workers=True)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__=='__main__':
    main()