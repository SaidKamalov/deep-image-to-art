from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "preprocess_one_image"
]


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch
    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def preprocess_one_image(image_path: str, range_norm: bool, half: bool, device: torch.device) -> Tensor:
    # read an image using OpenCV
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # BGR image channel data to RGB image channel data
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB image channel data to image formats supported by PyTorch
    tensor = image_to_tensor(image, range_norm, half).unsqueeze_(0)

    # Data transfer to the specified device
    tensor = tensor.to(device, non_blocking=True)

    return tensor