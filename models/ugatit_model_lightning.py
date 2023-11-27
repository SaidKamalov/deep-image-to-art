import itertools
import cv2

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from utils.img_preprocessing import tensor2numpy, RGB2BGR, denorm, cam

import pytorch_lightning as pl


class UGATIT_Lightning(pl.LightningModule):
    def __init__(self, light=False, batch_size=1, print_freq=1000,
                 decay_flag=True, lr=0.0001, weight_decay=0.0001,
                 adv_weight=1, cycle_weight=10, identity_weight=10, cam_weight=1000,
                 ch=64, n_res=4, img_size=256, epochs=100):
        super(UGATIT_Lightning, self).__init__()
        self.save_hyperparameters()

        # Generators and Discriminators
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res, img_size=img_size,
                                      light=light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res, img_size=img_size,
                                      light=light)
        self.disGA = Discriminator(input_nc=3, ndf=ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=ch, n_layers=5)

        self.Rho_clipper = RhoClipper(0, 1)

        # Losses
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    def forward(self, x):
        fake_A2B, _, fake_A2B_heatmap = self.genA2B(x)

        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

        return fake_A2B2A

    def transform_image(self, image_tensor):
        self.eval()
        with torch.no_grad():
            out_image = self(image_tensor)

        out_image = RGB2BGR(tensor2numpy(denorm(out_image[0]))) * 255.0

        return out_image

    def configure_optimizers(self):
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        beta1 = 0.5
        beta2 = 0.999

        g_optimizer = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                       lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        d_optimizer = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(),
                            self.disLB.parameters()),
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

        return [g_optimizer, d_optimizer], []

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch

        G_optim, D_optim = self.optimizers()

        if self.hparams.decay_flag and self.current_epoch > (self.hparams.epochs // 2):
            D_optim.param_groups[0]['lr'] -= (self.hparams.lr / (self.hparams.epochs // 2))
            G_optim.param_groups[0]['lr'] -= (self.hparams.lr / (self.hparams.epochs // 2))

        # Update D
        D_optim.zero_grad()

        fake_A2B, _, _ = self.genA2B(real_A)
        fake_B2A, _, _ = self.genB2A(real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(
            fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit,
                                         torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(
            fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(
            fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,
                                         torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(
            fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(
            fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,
                                         torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(
            fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(
            fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,
                                         torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(
            fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

        D_loss_A = self.hparams.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.hparams.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        Discriminator_loss = D_loss_A + D_loss_B
        self.manual_backward(Discriminator_loss)
        D_optim.step()

        # Update G
        G_optim.zero_grad()

        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

        G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

        G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,
                                     torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(
            fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,
                                     torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(
            fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        G_loss_A = (
                self.hparams.adv_weight * (
                G_ad_loss_GA + G_ad_cam_loss_GA +
                G_ad_loss_LA + G_ad_cam_loss_LA) +
                self.hparams.cycle_weight * G_recon_loss_A +
                self.hparams.identity_weight * G_identity_loss_A +
                self.hparams.cam_weight * G_cam_loss_A
        )
        G_loss_B = (
                self.hparams.adv_weight * (
                G_ad_loss_GB + G_ad_cam_loss_GB +
                G_ad_loss_LB + G_ad_cam_loss_LB) +
                self.hparams.cycle_weight * G_recon_loss_B +
                self.hparams.identity_weight * G_identity_loss_B +
                self.hparams.cam_weight * G_cam_loss_B)

        Generator_loss = G_loss_A + G_loss_B
        self.manual_backward(Generator_loss)
        G_optim.step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        self.genA2B.apply(self.Rho_clipper)
        self.genB2A.apply(self.Rho_clipper)

        tqdm_dict = {"D_loss": Discriminator_loss, "G_loss": Generator_loss}
        self.log_dict(tqdm_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.hparams.print_freq != 0:
            return

        real_A, real_B = batch

        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

        A2B = np.zeros((self.hparams.img_size * 7, 0, 3))
        B2A = np.zeros((self.hparams.img_size * 7, 0, 3))

        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.hparams.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        cv2.imwrite(f'val_results/A2B/epoch{self.current_epoch}_{batch_idx}.png', A2B * 255.0)
        cv2.imwrite(f'val_results/B2A/epoch{self.current_epoch}_{batch_idx}.png', B2A * 255.0)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper:
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
