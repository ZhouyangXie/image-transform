import pytest
import numpy as np
from numpy.random import randint, random
from math import pi

from image_transform.annotation import EmptyImage, Image, Point, PointArray, Box, BoxArray, OrientedBox,\
    OrientedBoxArray, Mask, Polygon, Composite, Scoped, MultipleScoped, ScopedMaskWithConfidence, ScopedWithConfidence,\
    ProbabilisticMultipleScoped, Empty, ArbitraryHashable

from image_transform.transform import Pad, PadTo, PadToMultiple, Crop, CentralCrop, Rotate, RotateRightAngle,\
    Rescale, Resize, Transpose, HorizontalFlip, VerticalFlip, RandomHorizontalFlip, RandomVerticalFlip, RandomRotate,\
    RandomRotateRightAngle, RandomTranspose, RandomCrop, Normalize, GaussianBlur, ToDataType, GaussianNoise, MixUp,\
    PadToMax, Filter, ToNumpy, NonMaximumSuppression, ConfidenceThreshold, SequentialTransforms, function


_scope = ["a", 3.14, False, None]


class MyArbitraryHashable(ArbitraryHashable):
    def to_numpy(self):
        # for compatibility with ToNumpy
        return randint(1)


class MyScoped(Scoped):
    scope = _scope


class MyMultipleScoped(MultipleScoped):
    scope = _scope


class MyScopedWidthConfidence(ScopedWithConfidence):
    scope = _scope


class MyProbabilisticMultipleScoped(ProbabilisticMultipleScoped):
    scope = _scope


class MyScopedMaskWithConfidence(ScopedMaskWithConfidence):
    scope = _scope


def get_random_annotation(image_annotation_type=Composite, img_w=100, img_h=100):
    if image_annotation_type == EmptyImage:
        return EmptyImage(img_w, img_h, label=MyScoped("a"))
    elif image_annotation_type == Image:
        image = get_random_annotation(EmptyImage, img_w, img_h).to_image()
        image.label = MyArbitraryHashable("qwertyuiop")
        return image
    elif image_annotation_type == Point:
        return Point(randint(0, img_w), randint(0, img_h), img_w, img_h, label=Empty())
    elif image_annotation_type == PointArray:
        return PointArray.from_points(
            [get_random_annotation(Point, img_w, img_h) for _ in range(randint(1, 20))])
    elif image_annotation_type == Box:
        xmin, xmax = randint(0, img_w), randint(0, img_w)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax) + 10
        ymin, ymax = randint(0, img_w), randint(0, img_w)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax) + 10
        return Box(xmin, xmax, ymin, ymax, img_w, img_h, label=MyScopedWidthConfidence(3.14, 0.8))
    elif image_annotation_type == BoxArray:
        return BoxArray.from_boxes(
            [get_random_annotation(Box, img_w, img_h) for _ in range(randint(1, 20))])
    elif image_annotation_type == OrientedBox:
        return OrientedBox(
            x=randint(0, img_w), y=randint(0, img_h), w=randint(img_w//8, img_w//2), h=randint(img_h//8, img_h//2),
            theta=random() * 2 * pi, img_w=img_w, img_h=img_h, label=MyMultipleScoped([False, 3.14])
        )
    elif image_annotation_type == OrientedBoxArray:
        return OrientedBoxArray.from_oriented_boxes([
            get_random_annotation(OrientedBox, img_w, img_h) for _ in range(randint(1, 20))])
    elif image_annotation_type == Polygon:
        points = get_random_annotation(OrientedBox, img_w, img_h).to_points()
        return Polygon(points, img_w, img_h, MyProbabilisticMultipleScoped([.4, .2, .3, .1]))
    elif image_annotation_type == Mask:
        polygon = get_random_annotation(Polygon, img_w, img_h)
        polygon.label = MyScoped(_scope[0])
        mask = polygon.to_mask()
        mask.label = MyScopedMaskWithConfidence(random(size=(img_h, img_w)).astype(np.float32))
        return mask
    else:
        return Composite([
            get_random_annotation(T, img_w, img_h) for T in
            [EmptyImage, Image, Point, PointArray, Box, BoxArray, OrientedBox, OrientedBoxArray, Polygon, Mask]
        ])


def test_transform_no_error():
    # Test that each of the Transform can process any type of ImageAnnotation.
    # But the correctness of the result if hard to test.
    # Basic geometric transformations have already been tested in image_transform.annotations.
    # We perform feasible tests like shape checks
    anno = get_random_annotation(img_w=100, img_h=100)
    anno.append(get_random_annotation(Polygon, 100, 100))
    invertible_transforms = [
        Pad(3, 4, 5, 6), PadTo(102, 109), PadToMultiple(10, 8), Crop(10, 90, 10, 90),
        CentralCrop(50, 89), Rotate(pi), RotateRightAngle(270), Rescale(0.4, 1.3),
        Resize(200, 120), Transpose(), HorizontalFlip(), VerticalFlip(),
        RandomCrop(), RandomCrop(target_w=20), RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotate(),
        RandomRotateRightAngle(), RandomTranspose(), GaussianBlur(3), GaussianNoise(),
        SequentialTransforms([ToDataType('fp64'), Normalize([.5, .5, .5], [.25, .25, .25])]),
        Filter([Image, Box, BoxArray], 'IsWithinImage.PARTIAL'),
    ]

    uninvertible_transforms = [
        NonMaximumSuppression(True, .5, polygon=False),
        NonMaximumSuppression(False, .5),
        ConfidenceThreshold(.45),
    ]

    for t in invertible_transforms:
        r = t(anno)
        inv_t = t.get_inverse()
        _ = inv_t(r)

    for t in uninvertible_transforms:
        r = t(anno)
        with pytest.raises(TypeError):
            t.get_inverse()


def test_minibatch_transform():
    class _MyProbabilisticMultipleScoped(ProbabilisticMultipleScoped):
        scope = [True, False, None]

    im1 = get_random_annotation(Image, img_w=110, img_h=90)
    im1.label = _MyProbabilisticMultipleScoped([.4, .3, .3])
    im2 = get_random_annotation(Image, img_w=100, img_h=100)
    im2.label = _MyProbabilisticMultipleScoped([.2, .3, .5])
    minibatch = [Composite([im1]), Composite([im2])]

    invertible_transforms = [PadToMax()]
    uninvertible_transforms = [MixUp()]

    for t in invertible_transforms:
        r = t(minibatch)
        inv_t = t.get_inverse()
        _ = inv_t(r)

    for t in uninvertible_transforms:
        r = t(minibatch)
        with pytest.raises(TypeError):
            t.get_inverse()


def test_to_numpy_transform():
    anno = get_random_annotation(img_w=100, img_h=200)
    t = ToNumpy()
    r = t(anno)

    assert isinstance(r, dict)

    images = r["images"]
    assert len(images) == 2
    empty_image, image = images
    assert isinstance(empty_image, np.ndarray)
    assert empty_image.dtype == np.uint8
    assert empty_image.shape == (3, 200, 100)
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == (3, 200, 100)

    image_labels = r["image_labels"]
    assert isinstance(image_labels, list) and len(image_labels) == 2
    assert np.isscalar(image_labels[0])
    assert image_labels[0].dtype == np.int64
    assert isinstance(image_labels[1], int)

    boxes = r["boxes"]
    assert isinstance(boxes, np.ndarray)
    assert boxes.dtype == np.int32
    assert boxes.ndim == 2 and boxes.shape[-1] == 4

    indices, confidences = r["box_labels"]
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int64
    assert indices.ndim == 1
    assert isinstance(confidences, np.ndarray)
    assert confidences.dtype == np.float32
    assert confidences.ndim == 1
    assert indices.shape == confidences.shape

    oriented_boxes = r["oriented_boxes"]
    assert isinstance(oriented_boxes, np.ndarray)
    assert oriented_boxes.dtype == np.float64
    assert oriented_boxes.ndim == 2 and oriented_boxes.shape[-1] == 5

    oriented_box_labels = r["oriented_box_labels"]
    assert isinstance(oriented_box_labels, np.ndarray)
    assert oriented_box_labels.dtype == np.int64
    assert oriented_box_labels.ndim == 2 and oriented_box_labels.shape[-1] == 4

    points = r["points"]
    assert isinstance(points, np.ndarray)
    assert points.dtype == np.int32
    assert points.ndim == 2 and points.shape[-1] == 2

    point_labels = r["point_labels"]
    assert all(pl is None for pl in point_labels)

    polygons = r["polygons"]
    assert isinstance(polygons, list) and len(polygons) == 1
    polygon = polygons[0]
    assert isinstance(polygon, np.ndarray)
    assert polygon.dtype == np.int32
    assert polygon.ndim == 2 and polygon.shape[-1] == 2

    polygon_labels = r["polygon_labels"]
    assert isinstance(polygon_labels, list) and len(polygon_labels) == 1
    polygon_label = polygon_labels[0]
    assert isinstance(polygon_label, np.ndarray)
    assert polygon_label.dtype == np.float32
    assert polygon_label.shape == (4, )

    masks = r["masks"]
    assert isinstance(masks, list) and len(masks) == 1
    mask = masks[0]
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.int32
    assert mask.shape == (200, 100)

    mask_confidences = r["mask_confidences"]
    assert isinstance(mask_confidences, list) and len(mask_confidences) == 1
    mask_confidence = mask_confidences[0]
    assert isinstance(mask_confidence, np.ndarray)
    assert mask_confidence.dtype == np.float32
    assert mask_confidence.shape == (200, 100)


def test_function():
    assert all(
        k in dir(function) for k in
        [
            "pad", "pad_to", "pad_to_multiple", "crop", "central_crop", "horizontal_flip",
            "vertical_flip", "rotate", "rotate_right_angle", "rescale", "resize", "transpose",
            "random_horizontal_flip", "random_vertical_flip", "random_rotate", "random_rotate_right_angle",
            "random_transpose", "random_crop", "normalize", "gaussian_blur", "gaussian_noise",
            "non_maximum_suppression", "confidence_threshold"
        ]
    )
    func = function.make_functional(Transpose)
    assert callable(func)
    _ = func(get_random_annotation(EmptyImage, img_w=100, img_h=200))
