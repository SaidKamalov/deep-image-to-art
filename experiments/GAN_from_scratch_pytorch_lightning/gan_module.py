import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from collections import OrderedDict
from matplotlib import pyplot as plt


# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor, so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(latent_dim, 7 * 7 * 64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)  # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  # 256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        return self.conv(x)


class GAN(pl.LightningModule):
    # Initialize. Define latent dim, learning rate, and Adam betas
    def __init__(self, latent_dim=100, lr=0.0002,
                 b1=0.5, b2=0.999, batch_size=128):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(6, self.hparams.latent_dim)

        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat.cpu(), y.cpu())

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim, device=self.device)

        opt_g, opt_d = self.optimizers()

        # train generator
        self.generated_imgs = self(z)
        predictions = self.discriminator(self.generated_imgs)
        g_loss = self.adversarial_loss(predictions, torch.ones(real_imgs.size(0), 1))
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # train discriminator
        real_preds = self.discriminator(real_imgs)
        real_loss = self.adversarial_loss(real_preds, torch.ones(real_imgs.size(0), 1))

        fake_preds = self.discriminator(self(z).detach())
        fake_loss = self.adversarial_loss(fake_preds, torch.zeros(real_imgs.size(0), 1))

        d_loss = (real_loss + fake_loss) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        tqdm_dict = {"d_loss": d_loss, 'g_loss': g_loss}
        self.log_dict(tqdm_dict, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g, opt_d

    def on_train_epoch_end(self):
        self.plot_imgs()

    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()

        print('epoch ', self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()
