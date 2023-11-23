import os
import sys
import argparse

sys.path.append(os.getcwd())

from models import cycle_gan_model, ugatit_model
from api.config import MODEL_PATHS, MODEL_CONFS

models_dict = {
    'cyclegan': cycle_gan_model.CycleNet,
    'ugatit': ugatit_model.UGATIT
}

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_path", type=str, required=True, help="Path to the input image")
argparser.add_argument("--output_path", type=str, required=True, help="Path to the output image")
argparser.add_argument("--model_name", type=str, required=True, default="CycleGAN",
                       help="Model name from {CycleGAN, UGATIT}")


def run_one_image(input_path, output_path, model_name, verbose=False):
    print("Creating model instance...") if verbose else None
    # Load model
    g_model = models_dict[model_name](**MODEL_CONFS[model_name])

    print("Loading model weights...") if verbose else None
    # Load model weights
    project_dir = '/'.join(__file__.split('/')[:-2])
    g_model.load(os.path.join(project_dir, MODEL_PATHS[model_name]))

    print("Transforming image...") if verbose else None
    g_model.eval()
    g_model.transform_image(input_path, output_path)

    print("Done!") if verbose else None


if __name__ == '__main__':
    args = argparser.parse_args()
    run_one_image(args.input_path, args.output_path, args.model_name.lower(), verbose=True)
