from torch import nn, optim
import lightning as L

class SkeletonExtrapolator(L.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-5, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Sequential(nn.Linear(36, 36), nn.ReLU(), 
                                     nn.Linear(36, 64), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(64, 36))
    
    def forward(self, x):
        # inference
        mask = x != -10.0
        x_hat = self.layers(x)
        x_hat[mask] = x[mask]
        return x_hat
    
    def general_step(self, batch, batch_idx, loss_name):
        x, y = batch
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        self.log(loss_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train_loss")
    
    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test_loss")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val_loss")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                               lr=self.hparams.learning_rate, 
                               weight_decay=self.hparams.weight_decay)
        return optimizer