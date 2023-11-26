import torch
from torch import nn
import pytorch_lightning as pl
import cv2
import numpy as np

from utils.img_preprocessing import RGB2BGR, tensor2numpy, denorm


class CycleGAN_Lightning(pl.LightningModule):
    def __init__(self, lr, reconstr_w=10, id_w=2, print_freq=1000, img_size=256):
        super(CycleGAN_Lightning, self).__init__()
        self.save_hyperparameters()

        # Models
        self.G_basestyle = Generator()
        self.G_stylebase = Generator()
        self.D_base = Discriminator()
        self.D_style = Discriminator()

        # Losses
        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()

        self.automatic_optimization = False

    def forward(self, x):
        return self.G_basestyle(x)

    def transform_image(self, image_tensor):
        self.eval()
        with torch.no_grad():
            out_image = self(image_tensor)

        out_image = RGB2BGR(tensor2numpy(denorm(out_image[0]))) * 255.0

        return out_image

    def configure_optimizers(self):
        self.g_basestyle_optimizer = torch.optim.Adam(self.G_basestyle.parameters(), lr=self.hparams.lr['G'],
                                                      betas=(0.5, 0.999))
        self.g_stylebase_optimizer = torch.optim.Adam(self.G_stylebase.parameters(), lr=self.hparams.lr['G'],
                                                      betas=(0.5, 0.999))
        self.d_base_optimizer = torch.optim.Adam(self.D_base.parameters(), lr=self.hparams.lr['D'], betas=(0.5, 0.999))
        self.d_style_optimizer = torch.optim.Adam(self.D_style.parameters(), lr=self.hparams.lr['D'],
                                                  betas=(0.5, 0.999))

        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer,
                self.d_style_optimizer], []

    def training_step(self, batch, batch_idx):
        g_basestyle_optimizer, g_stylebase_optimizer, d_base_optimizer, d_style_optimizer = self.optimizers()

        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).to(self.device)
        fake = torch.zeros(b, 1, 30, 30).to(self.device)

        # Train Generator
        # Validity
        # MSELoss
        val_base = self.generator_loss(self.D_base(self.G_stylebase(style_img)), valid)
        val_style = self.generator_loss(self.D_style(self.G_basestyle(base_img)), valid)
        val_loss = (val_base + val_style) / 2

        # Reconstruction
        reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)), base_img)
        reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)), style_img)
        reconstr_loss = (reconstr_base + reconstr_style) / 2

        # Identity
        id_base = self.mae(self.G_stylebase(base_img), base_img)
        id_style = self.mae(self.G_basestyle(style_img), style_img)
        id_loss = (id_base + id_style) / 2

        # Loss Weight
        G_loss = val_loss + self.hparams.reconstr_w * reconstr_loss + self.hparams.id_w * id_loss

        g_basestyle_optimizer.zero_grad()
        g_stylebase_optimizer.zero_grad()
        self.manual_backward(G_loss)
        g_basestyle_optimizer.step()
        g_stylebase_optimizer.step()

        # Train Discriminator
        # MSELoss
        D_base_gen_loss = self.discriminator_loss(self.D_base(self.G_stylebase(style_img)), fake)
        D_style_gen_loss = self.discriminator_loss(self.D_style(self.G_basestyle(base_img)), fake)
        D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
        D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)

        D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

        # Loss Weight
        D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

        d_base_optimizer.zero_grad()
        d_style_optimizer.zero_grad()
        self.manual_backward(D_loss)
        d_base_optimizer.step()
        d_style_optimizer.step()

        tqdm_dict = {'G_loss': G_loss, 'D_loss': D_loss}
        self.log_dict(tqdm_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.hparams.print_freq != 0:
            return

        base_img, style_img = batch

        fake_A2B = self.G_basestyle(base_img)
        fake_B2A = self.G_stylebase(style_img)

        A2B = np.zeros((self.hparams.img_size * 2, 0, 3))
        B2A = np.zeros((self.hparams.img_size * 2, 0, 3))

        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(base_img[0]))),
                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))), 0)), 1)

        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(style_img[0]))),
                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))), 0)), 1)

        cv2.imwrite(f'val_results/A2B/epoch{self.current_epoch}_{batch_idx}.png', A2B * 255.0)
        cv2.imwrite(f'val_results/B2A/epoch{self.current_epoch}_{batch_idx}.png', B2A * 255.0)


class Discriminator(nn.Module):
    def __init__(self, filter=64):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(3, filter, kernel_size=4, stride=2, apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=2),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


class Generator(nn.Module):
    def __init__(self, filter=64):
        super(Generator, self).__init__()
        self.downsamples = nn.ModuleList([
            Downsample(3, filter, kernel_size=4, apply_instancenorm=False),  # (b, filter, 128, 128)
            Downsample(filter, filter * 2),  # (b, filter * 2, 64, 64)
            Downsample(filter * 2, filter * 4),  # (b, filter * 4, 32, 32)
            Downsample(filter * 4, filter * 8),  # (b, filter * 8, 16, 16)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 8, 8)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 4, 4)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 2, 2)
        ])

        self.upsamples = nn.ModuleList([
            Upsample(filter * 8, filter * 8),
            Upsample(filter * 16, filter * 8),
            Upsample(filter * 16, filter * 8),
            Upsample(filter * 16, filter * 4, dropout=False),
            Upsample(filter * 8, filter * 2, dropout=False),
            Upsample(filter * 4, filter, dropout=False)
        ])

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_instancenorm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x
