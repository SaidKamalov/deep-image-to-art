import torch
from torchvision.utils import save_image

from model import cycle_gan_model
from utils import img_preprocessing as img_pre
from utils import load_model


def run_one_image(input_path, output_path, model_weights_path, device):
    device = torch.device(device)
    g_model = cycle_gan_model.__dict__['cyclenet'](in_channels=3, out_channels=3, channels=64)
    g_model = g_model.to(device)

    # Load image
    image = img_pre.preprocess_one_image(input_path, True, False, device)

    # Load model weights
    g_model = load_model.load_pretrained_state_dict(g_model, False, model_weights_path)
    g_model.eval()

    with torch.no_grad():
        gen_image = g_model(image)
        save_image(gen_image.detach(), output_path, normalize=True)
        print(f"Gen image save to `{output_path}`")
