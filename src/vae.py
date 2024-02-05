from torch import nn, optim
import torch
import lightning as L

class SkeletonVAE(L.LightningModule):

    def __init__(self, input_dim=36, hidden_dim=32, latent_dim=4, learning_rate=0.001, weight_decay=0.001):
        super().__init__()
        self.save_hyperparameters()
        hidden_half = hidden_dim // 2
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_half), nn.LeakyReLU(0.2)
        )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_half, latent_dim)
        self.logvar_layer = nn.Linear(hidden_half, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_half), nn.LeakyReLU(0.2),
            nn.Linear(hidden_half, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)     
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
    def forward(self, x):
        # inference
        mask = x != -10.0
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        x_hat[mask] = x[mask]
        return x_hat
    
    def general_step(self, batch, batch_idx, loss_name):
        x, y = batch
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        loss = nn.functional.mse_loss(x_hat, y)
        KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss += KLD
        self.log(loss_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train_loss")
    
    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test_loss")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val_loss")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.hparams.learning_rate)
        return optimizer