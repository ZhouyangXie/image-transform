from .basic import Transform, EmptyTransform, SequentialTransforms,\
    MinibatchTransform, SequentialMinibatchTransforms, MinibatchRespectiveTransform
from .geometric import GeometricTransform, Pad, PadTo, PadToMultiple, Crop, CentralCrop, Rotate, RotateRightAngle,\
    Rescale, Resize, Transpose, HorizontalFlip, VerticalFlip, ResizeAndPad
from .random_geometric import RandomHorizontalFlip, RandomRotate, RandomVerticalFlip,\
    RandomTranspose, RandomRotateRightAngle, RandomCrop
from .color import ColorTransform, Normalize, GaussianBlur, ToDataType, GaussianNoise, RGB2Gray, Gray2RGB
from .conversion import Filter, ToNumpy
from .minibatch import MixUp, PadToMax, Stack
from .postprocess import NonMaximumSuppression, ConfidenceThreshold


all_names = {
    # bacis
    Transform.__name__: Transform,
    EmptyTransform.__name__: EmptyTransform,
    SequentialTransforms.__name__: SequentialTransforms,
    MinibatchTransform.__name__: MinibatchTransform,
    SequentialMinibatchTransforms.__name__: SequentialMinibatchTransforms,
    MinibatchRespectiveTransform.__name__: MinibatchRespectiveTransform,
    # geometric
    GeometricTransform.__name__: GeometricTransform,
    Pad.__name__: Pad,
    PadTo.__name__: PadTo,
    PadToMultiple.__name__: PadToMultiple,
    Crop.__name__: Crop,
    CentralCrop.__name__: CentralCrop,
    Rotate.__name__: Rotate,
    RotateRightAngle.__name__: RotateRightAngle,
    Rescale.__name__: Rescale,
    Resize.__name__: Resize,
    Transpose.__name__: Transpose,
    HorizontalFlip.__name__: HorizontalFlip,
    VerticalFlip.__name__: VerticalFlip,
    ResizeAndPad.__name__: ResizeAndPad,
    # random geometric
    RandomHorizontalFlip.__name__: RandomHorizontalFlip,
    RandomRotate.__name__: RandomRotate,
    RandomRotateRightAngle.__name__: RandomRotateRightAngle,
    RandomVerticalFlip.__name__: RandomVerticalFlip,
    RandomTranspose.__name__: RandomTranspose,
    RandomCrop.__name__: RandomCrop,
    # color
    ColorTransform.__name__: ColorTransform,
    Normalize.__name__: Normalize,
    GaussianBlur.__name__: GaussianBlur,
    ToDataType.__name__: ToDataType,
    GaussianNoise.__name__: GaussianNoise,
    RGB2Gray.__name__: RGB2Gray,
    Gray2RGB.__name__: Gray2RGB,
    # minibatch
    MixUp.__name__: MixUp,
    PadToMax.__name__: PadToMax,
    Stack.__name__: Stack,
    # conversion
    Filter.__name__: Filter,
    ToNumpy.__name__: ToNumpy,
    # post-processing
    NonMaximumSuppression.__name__: NonMaximumSuppression,
    ConfidenceThreshold.__name__: ConfidenceThreshold,
}

__all__ = list(all_names.keys()) + ["function", ]
