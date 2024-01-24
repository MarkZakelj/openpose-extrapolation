from torch import nn, optim
import lightning as L

class LitAutoEncoder(L.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-5, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Sequential(nn.Linear(36, 36), nn.ReLU(), 
                                     nn.Linear(36, 64), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(64, 36), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(36, 36))
    
    def forward(self, x):
        # inference
        mask = x != -10.0
        x_hat = self.layers(x)
        x_hat[mask] = x[mask]
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                               lr=self.hparams.learning_rate, 
                               weight_decay=self.hparams.weight_decay)
        return optimizer