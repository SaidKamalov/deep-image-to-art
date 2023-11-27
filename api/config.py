ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
INFERENCE_RESULTS_FOLDER = 'static/inference_results'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

DEVICE = "cpu"
MODEL_PATHS = {
    "cyclegan": "models/checkpoints/CycleGAN-landscape2cubism.ckpt",
    "ugatit_light": "models/checkpoints/UGATIT-light-landscape2cubism.ckpt",
    "cut": "models/checkpoints/CUT-landscape2cubism.ckpt"
}
MODEL_CONFS = {
    "cyclegan": {
        'lr': {
            'G': 0.0002,
            'D': 0.0002
        },
    },
    "ugatit_light": {
        "light": True,
        "img_size": 128,
    },
    "cut": {

    }
}
