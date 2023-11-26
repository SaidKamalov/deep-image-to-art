ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
INFERENCE_RESULTS_FOLDER = 'static/inference_results'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

DEVICE = "cpu"
MODEL_PATHS = {
    "cyclegan": "models/checkpoints/CycleGAN-apple2orange.pth.tar",
    "ugatit_light": "models/checkpoints/UGATIT-light-landscape2cubism.ckpt"
}
MODEL_CONFS = {
    "cyclegan": {
        "in_channels": 3,
        "out_channels": 3,
        "channels": 64,
        "device": DEVICE
    },
    "ugatit_light": {
        "light": True,
        "img_size": 128,
    }
}
