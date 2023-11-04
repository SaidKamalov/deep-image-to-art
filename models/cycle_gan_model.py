import torch
from torch import nn, Tensor
from torchvision.utils import save_image
from utils import img_preprocessing as img_pre
from utils import load_model

__all__ = [
    "PathDiscriminator", "CycleNet",
]


class CycleNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            device: str,
    ) -> None:
        super(CycleNet, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, channels, (7, 7), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Downsampling
            nn.Conv2d(channels, int(channels * 2), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(channels * 2), int(channels * 4), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4), track_running_stats=True),
            nn.ReLU(True),

            # Residual blocks
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),

            # Upsampling
            nn.ConvTranspose2d(int(channels * 4), int(channels * 2), (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(channels * 2), channels, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, (7, 7), (1, 1), (0, 0)),
            nn.Tanh(),
        )

        self.device = device

        self.init_weights()

    def init_weights(self):
        classname = self.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(self.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(self.weight, 1.0, 0.02)
            torch.nn.init.zeros_(self.bias)

        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)

    def transform_image(self, input_path, output_path):
        image = img_pre.preprocess_one_image_cv(input_path, True, False, self.device)

        with torch.no_grad():
            gen_image = self(image)
            save_image(gen_image.detach(), output_path, normalize=True)

    def load(self, checkpoint_path):
        load_model.load_pretrained_state_dict(self, False, checkpoint_path)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(_ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.res(x)

        x = torch.add(x, identity)

        return x


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
    ) -> None:
        super(PathDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 8)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 8), out_channels, (4, 4), (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x
