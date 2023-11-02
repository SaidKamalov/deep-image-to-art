from models import cycle_gan_model, ugatit_model
from api.config import MODEL_PATHS, MODEL_CONFS

models_dict = {
    'CycleGAN': cycle_gan_model.CycleNet,
    'UGATIT': ugatit_model.UGATIT
}


def run_one_image(input_path, output_path, model_name):
    g_model = models_dict[model_name](**MODEL_CONFS[model_name])

    # Load model weights
    g_model.load(MODEL_PATHS[model_name])

    g_model.eval()
    g_model.transform_image(input_path, output_path)
