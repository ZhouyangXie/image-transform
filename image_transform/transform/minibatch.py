from random import shuffle
from typing import Union, List, Tuple

import numpy as np

from ..annotation import Image, EmptyImage, Mask, Composite,\
    Scoped, ScopedWithConfidence, ProbabilisticMultipleScoped
from .geometric import PadTo
from .basic import MinibatchTransform, MinibatchRespectiveTransform
from .conversion import ToNumpy


class PadToMax(MinibatchTransform):
    def __init__(self, fill_value=0):
        """
            Pad the images to the max image width/height in the mini-batch.

        Args:
            fill_value (int, optional): See Pad(Transform). Defaults to 0.
        """
        super().__init__()
        self.fill_value = fill_value
        self.transforms = None

    def _transform(self, images):
        max_img_w = max([im.img_w for im in images])
        max_img_h = max([im.img_h for im in images])
        self.transforms = MinibatchRespectiveTransform([
            PadTo(max_img_w, max_img_h, self.fill_value) for _ in range(len(images))
        ])
        return self.transforms.transform(images)

    def _get_inverse(self):
        return self.transforms.get_inverse()


class MixUp(MinibatchTransform):
    def __init__(self, alpha: float = 1.):
        """
        Merge each Image (or a single Image in Composite) with another random image in the mini-batch.
        The label on the Image is also merged as ProbabilisticMultipleScoped.
        Note that other annotation types are not affected.

        Args:
            alpha (float, optional): parameter of Beta(alpha, alpha) distribution that generate the random
                merging factor. Defaults to 1.0.
        """
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def get_image_with_prob(image: Union[Image, Composite]) -> Image:
        if isinstance(image, Composite):
            assert image.num_image == 1
            image = image.image
        else:
            assert isinstance(image, Image)

        if isinstance(image.label, (Scoped, ScopedWithConfidence)):
            image.label = image.label.to_probabilistic_multiple_scoped()
        elif isinstance(image.label, ProbabilisticMultipleScoped):
            pass
        else:
            raise TypeError(f"Label type {type(image.label)} invalid for MixUp")

        return image

    @staticmethod
    def merge_image(image_a: Image, image_b: Image, lam):
        assert 0. <= lam <= 1.
        assert isinstance(image_a.label, type(image_b.label))
        image_a = MixUp.get_image_with_prob(image_a)
        image_b = MixUp.get_image_with_prob(image_b)
        image_b = image_b.resize(image_a.img_w, image_a.img_h)

        image_c_data = (image_a.data * lam + image_b.data * (1. - lam)).astype(image_a.dtype)
        image_c_prob = image_a.label.probs * lam + image_b.label.probs * (1. - lam)
        image_c = Image(image_c_data, type(image_a.label)(image_c_prob))
        return image_c

    def _transform(self, images):
        original_indices = list(range(len(images)))
        paired_indices = original_indices.copy()
        shuffle(paired_indices)
        if any(i == j for i, j in zip(original_indices, paired_indices)):
            shuffle(paired_indices)

        return [
            MixUp.merge_image(images[i], images[j], np.random.beta(self.alpha, self.alpha))
            for i, j in zip(original_indices, paired_indices)
        ]


class Stack(MinibatchTransform):
    '''
        A mini-batch version of ToNumpy. Refer to the doc of ToNumpy first.

        For "images" and "masks", inputs are expected to have the same number and size of objects for each type.
            The np.ndarray will be stacked accodingly.
        For "image_labels" and "mask_confidences", the np.ndarray will be stacked is not None.

        Other objects are simply assembled as lists.
    '''
    def __init__(self, image_as_feature_map: bool = True, box_format: str = "xxyy"):
        self.image_as_feature_map = bool(image_as_feature_map)
        assert box_format in ("xxyy", "xyxy", "xywh", "cxcywh")
        self.box_format = box_format
        self.transforms = None
        super().__init__()

    @staticmethod
    def stack_labels(labels: List[Union[np.ndarray, Tuple[np.ndarray], None]]):
        if len(labels) == 0:
            return labels
        if isinstance(labels[0], np.ndarray):
            return np.stack(labels)
        elif labels[0] is None:
            return None
        elif isinstance(labels[0], tuple):
            return tuple(np.stack(_labels) for _labels in zip(*labels))
        else:
            raise TypeError

    def _transform(self, images):
        self.transforms = [ToNumpy(self.image_as_feature_map, self.box_format) for _ in range(len(images))]
        image_annotation_type = type(images[0])
        if len(images) > 1:
            assert all(isinstance(im, image_annotation_type) for im in images)
        np_dicts = [t(im) for t, im in zip(self.transforms, images)]

        if image_annotation_type in (Image, EmptyImage, Mask):
            images, labels = zip(*np_dicts)
            return np.stack(images), Stack.stack_labels(labels)
        elif image_annotation_type == Composite:
            result = dict()
            for k in ["images", "masks"]:
                result[k] = [np.stack(_images) for _images in zip(*[np_dict[k] for np_dict in np_dicts])]
            for k in ["image_labels", "mask_confidences"]:
                result[k] = [Stack.stack_labels(labels) for labels in zip(*[np_dict[k] for np_dict in np_dicts])]
            for k in [
                    "boxes", "box_labels", "oriented_boxes", "oriented_box_labels",
                    "points", "point_labels", "polygons", "polygon_labels"]:
                result[k] = [np_dict[k] for np_dict in np_dicts]
            return result
        else:
            images, labels = zip(*np_dicts)
            return list(images), list(labels)

    def _get_inverse(self):
        return self.transforms.get_inverse()
