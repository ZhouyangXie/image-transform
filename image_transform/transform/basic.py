from abc import abstractmethod, ABC
from typing import Iterable

from ..annotation import ImageAnnotation


class Transform(ABC):
    """
    This abstract class defines the two states of a Transform class.
    State 1: When initialized, only self.transform can be called. self.get_inverse should not be called.
    State 2: When self.transform is called, the base image is set to the new image,
        and the inverse transfromation can be called for arbitrarily times.
    When self.clear() is called, it returns State 1.
    """
    def __init__(self):
        self.image = None

    @property
    def img_w(self):
        return self.image.img_w

    @property
    def img_h(self):
        return self.image.img_h

    @abstractmethod
    def _transform(self, image: ImageAnnotation):
        pass

    def transform_get_inverse(self, image: ImageAnnotation):
        return self.transform(image), self.get_inverse()

    def transform(self, image: ImageAnnotation):
        assert isinstance(image, ImageAnnotation),\
            f"Transfrom target must be an ImageAnnotation, got type {type(image)}"
        self.image = image
        return self._transform(image)

    def __call__(self, image: ImageAnnotation):
        return self.transform(image)

    def _get_inverse(self):
        raise TypeError(f'Transformation {type(self)} is not inversible.')

    def get_inverse(self):
        assert self.image is not None,\
            "Transform.get_inverse can be called only after Transform.transform is called."
        return self._get_inverse()

    def is_inversible(self):
        return self.image is not None

    def clear(self):
        self.image = None


class EmptyTransform(Transform):
    def _transform(self, image: ImageAnnotation):
        return image

    def _get_inverse(self):
        return EmptyTransform()


class SequentialTransforms(Transform):
    """
        Perform transformations sequentially on the input images.
    """
    def __init__(self, transforms: Iterable[Transform]):
        self.transforms = list(transforms)
        assert all([isinstance(t, Transform) for t in self.transforms]),\
            "SequentialTransforms must consist of Transform objects"

    def _transform(self, image: ImageAnnotation):
        for t in self.transforms:
            image = t(image)
        return image

    def _get_inverse(self):
        return SequentialTransforms(
            [t.get_inverse() for t in self.transforms[::-1]],
        )


class MinibatchTransform(ABC):
    """
        Similar as Transform, MinibatchTransform defines the state of the transformation,
        but on a mini-batch of images.

        Note that the transformed mini-batch objects may have changed mini-batch size and type.
    """
    def __init__(self):
        self.images = None

    @abstractmethod
    def _transform(self, images: Iterable[ImageAnnotation]):
        pass

    def transform(self, images: Iterable[ImageAnnotation]):
        assert len(images) > 0,\
            "MinibatchTransform target should be a non-empty list of ImageAnnotation"
        assert all(isinstance(li, ImageAnnotation) for li in images),\
            "MinibatchTransform target should be a non-empty list of ImageAnnotation"
        self.images = images
        return self._transform(images)

    def __call__(self, images: Iterable[ImageAnnotation]):
        return self.transform(list(images))

    def _get_inverse(self):
        raise TypeError(f'MinibatchTransform {type(self)} is not inversible.')

    def get_inverse(self):
        assert self.is_inversible(),\
            "MinibatchTransform.get_inverse can be called only "\
            "after MinibatchTransform.transform is called"
        return self._get_inverse()

    def is_inversible(self):
        return self.images is not None

    def clear(self):
        self.images = None


class MinibatchRespectiveTransform(MinibatchTransform):
    """
       Perform K transofmations respectively on each image of K-size mini-batch.
    """
    def __init__(self, transforms: Iterable[Transform]):
        super().__init__()
        self.transforms = list(transforms)

    def _transform(self, images: Iterable[ImageAnnotation]):
        assert len(images) == len(self.transforms),\
            f"MinibatchRespectiveTransform requires equal number of transforms"\
            f"({len(self.transforms)}) and mini-batch images({len(images)})"
        return [t(im) for t, im in zip(self.transforms, images)]

    def _get_inverse(self):
        return MinibatchRespectiveTransform([t.get_inverse() for t in self.transforms])


class SequentialMinibatchTransforms(MinibatchTransform):
    """
        Perform transformations sequentially on the mini-batch.
    """
    def __init__(self, transforms: Iterable[MinibatchTransform]):
        self.transforms = list(transforms)
        assert all([isinstance(t, MinibatchTransform) for t in self.transforms]),\
            "SequentialMinibatchTransforms must consist of MinibatchTransform objects"

    def _transform(self, images: Iterable[ImageAnnotation]):
        for t in self.transforms:
            images = t(images)
        return images

    def _get_inverse(self):
        return SequentialMinibatchTransforms(
            [t.get_inverse() for t in self.transforms[::-1]],
        )
