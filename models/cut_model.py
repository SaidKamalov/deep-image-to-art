from typing import Any
import pytorch_lightning as pl
import models.cut_networks as cut_networks
import torch
import numpy as np
import cv2
from models.cut_patchnce import PatchNCELoss
from torch.optim import lr_scheduler
from utils.img_preprocessing import RGB2BGR, denorm, tensor2numpy


class CUT(pl.LightningModule):
    def __init__(self, img_size=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        # base options
        self.input_nc = kwargs.get("input_nc", 3)  # input channels
        self.output_nc = kwargs.get("output_nc", 3)  # output channels
        self.ngf = kwargs.get("ngf", 64)
        self.ndf = kwargs.get("ndf", 64)
        self.G_type = kwargs.get(
            "netG", "resnet_9blocks"
        )  # ['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat']
        self.normG = kwargs.get("normG", "instance")  # ['instance', 'batch', 'none']
        self.dropout = kwargs.get("dropout", False)
        self.init_type = kwargs.get(
            "init_type", "xavier"
        )  # ['normal', 'xavier', 'kaiming', 'orthogonal']
        self.init_gain = kwargs.get(
            "init_gain", 0.02
        )  # scaling factor for normal, xavier and orthogonal.
        # no_antialias = False
        # no_antialias_up = False
        self.D_type = kwargs.get("netD", "basic")
        self.n_layers_D = kwargs.get("n_layers_D", 3)
        self.normD = kwargs.get("normD", "instance")
        self.gan_mode = kwargs.get("gan_mode", "lsgan")  # [vanilla| lsgan | wgangp]
        self.batch_size = kwargs.get("batch_size", 16)
        self.lr = kwargs.get("lr", 0.0002)
        self.number_of_epochs = kwargs.get("number_of_epochs", 10)
        self.n_of_epochs_save_lr = int(self.number_of_epochs / 2)

        # specific options
        # nce loss related:
        self.nce_layers = kwargs.get("nce_layers", [0, 4, 8, 12, 16])
        self.nce_T = kwargs.get("nce_T", 0.07)  # "temperature for NCE loss"
        self.lambda_NCE = kwargs.get("lambda_NCE", 1.0)
        # nce_idt=True
        # GAN loss related:
        self.lambda_GAN = kwargs.get("lambda_GAN", 1.0)
        # Downsampling related:
        self.F_type = kwargs.get(
            "netF", "mlp_sample"
        )  # ["sample", "reshape", "mlp_sample"]
        self.netF_nc = kwargs.get("netF_nc", 256)
        # pathces related:
        self.num_patches = kwargs.get("num_patches", 256)

        # initializing part
        self.netG = cut_networks.define_G(
            input_nc=self.input_nc,
            output_nc=self.output_nc,
            ngf=self.ngf,
            netG=self.G_type,
            norm=self.normG,
            use_dropout=self.dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
        )
        self.netF = cut_networks.define_F(
            input_nc=self.input_nc,
            netF=self.F_type,
            norm=self.normG,
            use_dropout=self.dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
            netF_nc=self.netF_nc,
        )
        self.netD = cut_networks.define_D(
            input_nc=self.output_nc,
            ndf=self.ndf,
            netD=self.D_type,
            n_layers_D=self.n_layers_D,
            norm=self.normD,
            init_type=self.init_type,
            init_gain=self.init_gain,
        )

        # define loss functions
        self.betta1 = 0.5
        self.betta2 = 0.999
        self.criterionGAN = cut_networks.GANLoss(self.gan_mode)
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(
                PatchNCELoss(batch_size=self.batch_size, nce_T=self.nce_T)
            )
        self.criterionIdt = torch.nn.L1Loss().to(self.device)

    def training_step(self, batch, batch_idx):
        self.real_A, self.real_B = batch
        if self.current_epoch == 0 and batch_idx == 0:
            self.data_dependent_initialize()

        self.forward(True)
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.manual_backward(self.loss_D)
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.F_type == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.manual_backward(self.loss_G)
        self.optimizer_G.step()
        if self.F_type == "mlp_sample":
            self.optimizer_F.step()

        metrics = {"g_loss": self.loss_G, "d_loss": self.loss_D}
        self.log_dict(metrics, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.real_A, self.real_B = batch
        self.real = self.real_A
        self.fake = self.netG(self.real)
        A2B = np.zeros((self.real_A.size()[2] * 2, 0, 3))
        B2A = np.zeros((self.real_A.size()[2] * 2, 0, 3))

        A2B = np.concatenate(
            (
                A2B,
                np.concatenate(
                    (
                        RGB2BGR(tensor2numpy(denorm(self.real_A[0]))),
                        RGB2BGR(tensor2numpy(denorm(self.fake[0]))),
                    ),
                    0,
                ),
            ),
            1,
        )

        cv2.imwrite(
            f"val_results/A2B/epoch{self.current_epoch}_{batch_idx}.png", A2B * 255.0
        )

    def configure_optimizers(self):
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.betta1, self.betta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(self.betta1, self.betta2)
        )
        self.optimizers_ = []
        self.optimizers_.append(self.optimizer_G)
        self.optimizers_.append(self.optimizer_D)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.n_of_epochs_save_lr) / float(
                self.number_of_epochs - self.n_of_epochs_save_lr
            )
            return lr_l

        self.schedulers_ = [
            lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            for optimizer in self.optimizers_
        ]

        return self.optimizers_, self.schedulers_

    def forward(self, is_train=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0) if is_train else self.real_A
        )

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0) :]

    def data_dependent_initialize(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.forward()  # compute fake images: G(A)
        self.compute_D_loss().backward()  # calculate gradients for D
        self.compute_G_loss().backward()  # calculate graidents for G
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(),
            lr=self.lr,
            betas=(self.betta1, self.betta2),
        )
        self.optimizers_.append(self.optimizer_F)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.n_of_epochs_save_lr) / float(
                self.number_of_epochs - self.n_of_epochs_save_lr
            )
            return lr_l

        self.schedulers_.append(
            lr_scheduler.LambdaLR(self.optimizer_F, lr_lambda=lambda_rule)
        )

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = (
                self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
            )
        else:
            self.loss_G_GAN = 0.0

        if self.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers
        ):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def transform_image(self, image_tensor):
        self.eval()
        with torch.no_grad():
            out_image = self.netG(image_tensor)

        out_image = RGB2BGR(tensor2numpy(denorm(out_image[0]))) * 255.0

        return out_image
