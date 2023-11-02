ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
INFERENCE_RESULTS_FOLDER = 'static/inference_results'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

DEVICE = "cpu"
MODEL_PATHS = {
    "CycleGAN": "../models/checkpoints/CycleGAN-apple2orange.pth.tar",
    "UGATIT": "../models/checkpoints/UGATIT-light-landscape2cubism.pt"
}
MODEL_CONFS = {
    "CycleGAN": {
        "in_channels": 3,
        "out_channels": 3,
        "channels": 64,
        "device": DEVICE
    },
    "UGATIT": {
        'light': True,
        'ch': 64,
        'n_res': 4,
        'img_size': 512,
        'device': DEVICE
    }
}
