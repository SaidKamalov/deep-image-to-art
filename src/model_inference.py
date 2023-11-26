import os
import sys
import argparse

sys.path.append(os.getcwd())

import cv2

from models import cycle_gan_model_lightning, ugatit_model_lightning
from api.config import MODEL_PATHS
from utils.dataset import read_image_to_np
from src.prepare_dataset import DEFAULT_TRANSFORM

models_dict = {
    'cyclegan': cycle_gan_model_lightning.CycleGAN_Lightning,
    'ugatit_light': ugatit_model_lightning.UGATIT_Lightning
}

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_path", type=str, required=True, help="Path to the input image")
argparser.add_argument("--output_path", type=str, required=True, help="Path to the output image")
argparser.add_argument("--model_name", type=str, required=True, default="CycleGAN",
                       help="Model name from {CycleGAN, UGATIT_Light}")


def run_one_image(input_path, output_path, model_name, verbose=False):
    print("Loading model...") if verbose else None
    # Load model
    project_dir = '/'.join(__file__.split('/')[:-2])
    g_model = (
        models_dict[model_name]
        .load_from_checkpoint(os.path.join(project_dir,
                                           MODEL_PATHS[model_name]))
    )

    print("Loading image...") if verbose else None
    # Load image
    image_np = read_image_to_np(input_path, g_model.hparams.img_size, g_model.hparams.img_size)
    image_tensor = DEFAULT_TRANSFORM(image_np).unsqueeze_(0).to(g_model.device)

    print("Transforming image...") if verbose else None
    out_image = g_model.transform_image(image_tensor)

    print("Saving image...") if verbose else None
    # Save image
    cv2.imwrite(output_path, out_image)

    print("Done!") if verbose else None


if __name__ == '__main__':
    args = argparser.parse_args()
    run_one_image(args.input_path, args.output_path, args.model_name.lower(), verbose=True)
