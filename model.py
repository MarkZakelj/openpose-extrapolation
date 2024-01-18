from torch import nn, optim
import lightning as L

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(37, 64), nn.ReLU(), 
                                     nn.Linear(64, 64), nn.ReLU(), 
                                     nn.Linear(64, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), 
                                     nn.Linear(64, 37))
    
    def forward(self, x):
        mask = x != -1.0
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat[mask] = x[mask]
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer