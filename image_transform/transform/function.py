import numpy as np
import PIL

from ..annotation import ImageAnnotation, Image
from .color import Normalize, GaussianBlur, GaussianNoise
from .geometric import Pad, PadTo, PadToMultiple, Crop, CentralCrop, HorizontalFlip,\
    VerticalFlip, Rotate, RotateRightAngle, Rescale, Resize, Transpose
from .random_geometric import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate,\
    RandomRotateRightAngle, RandomTranspose, RandomCrop
from .postprocess import NonMaximumSuppression, ConfidenceThreshold


_func_name_to_transform_type = {
    # geometric
    "pad": Pad, "pad_to": PadTo, "pad_to_multiple": PadToMultiple, "crop": Crop,
    "central_crop": CentralCrop, "horizontal_flip": HorizontalFlip, "vertical_flip": VerticalFlip,
    "rotate": Rotate, "rotate_right_angle": RotateRightAngle, "rescale": Rescale, "resize": Resize,
    "transpose": Transpose,
    # random geometric
    "random_horizontal_flip": RandomHorizontalFlip, "random_vertical_flip": RandomVerticalFlip,
    "random_rotate": RandomRotate, "random_rotate_right_angle": RandomRotateRightAngle,
    "random_transpose": RandomTranspose, "random_crop": RandomCrop,
    # color
    "normalize": Normalize, "gaussian_blur": GaussianBlur, "gaussian_noise": GaussianNoise,
    # post-processing
    "non_maximum_suppression": NonMaximumSuppression, "confidence_threshold": ConfidenceThreshold,
}


def make_functional(transform_type):
    def _wrap_transform(image, *args, **kwargs):
        if isinstance(image, np.ndarray):
            _image = Image(image)
        elif isinstance(image, PIL.Image.Image):
            _image = Image.from_pil(image)
        else:
            assert isinstance(image, ImageAnnotation)
            _image = image
        _image = transform_type(*args, **kwargs).transform(_image)
        if isinstance(image, np.ndarray):
            return image.data
        elif isinstance(image, PIL.Image.Image):
            return _image.to_pil()
        else:
            return _image

    return _wrap_transform


for func_name, transform_type in _func_name_to_transform_type.items():
    globals()[func_name] = make_functional(transform_type)
