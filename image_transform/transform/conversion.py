import numpy as np

from .basic import Transform, EmptyTransform
from ..annotation import ImageAnnotation, Composite, IsWithinImage, all_names,\
    Label, Empty, Box, BoxArray, PointArray, OrientedBoxArray, Polygon, Image,\
    EmptyImage, Mask, ScopedMaskWithConfidence


class Filter(Transform):
    def __init__(self, types=None, within_image=None):
        types = [all_names[t] if t in all_names else t for t in types]
        self.allow_types = types
        within_image = all_names[within_image] if within_image in all_names else within_image
        self.allow_within_image = IsWithinImage.NO if within_image is None else within_image

    def _transform(self, composite: ImageAnnotation):
        if not isinstance(composite, Composite):
            composite = Composite([composite])
        return composite.filter_type(self.allow_types)\
            .filter_within_image(self.allow_within_image)

    def _get_inverse(self):
        return EmptyTransform()


class ToNumpy(Transform):
    """
        This is a special Transform that convert an `ImageAnnotation` to `np.ndarray` by calling
        `ImageAnnotation.to_numpy()` and `ImageAnnotation.label.to_numpy()` if provided. So the transformation
        result is no longer an `ImageAnnotation`. This `Transform` is meant to be the end of a preprocessing pipeline.

        Because the semantics is lost in the result `np.ndarray`, this `Transform` is not invertible
        (to be compatible with the pipeline, `EmptyTransform` will be returned). Use the constructor of appropriate
        `ImageAnnotation` and `Label` to pack the raw `np.ndarray`.

        Here is how each type of `ImageAnnotation` will be treated:

        (1) `ImageAnnotation` except `Composite`:
            Return Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, None]
            `ImageAnnotation.to_numpy()` will be called and the result is set to the first np.ndarray.
            `Label.to_numpy()` will be called if not of type Empty. If Empty, the second element of the tuple is None.

        (2) `Composite`:
            Firstly, `self.compact()` will be called. Then each annotation's and label's `to_numpy()` method
            will be called. User should ensure the label of annotations of same type can be stacked(by `np.stack`).

            A `dict` object will be returned, containing keys and values:
            {
                "images": `List[np.ndarray]`,
                    # All `Image`/`EmptyImage` objects.
                "image_labels": `List[Union[np.ndarray, Tuple[np.ndarray], None]]`,
                    # Labels of the images. Same length as images. `None` if label.to_numpy() cannot be called.
                "boxes": `np.ndarray`,
                    # All `Box`/`BoxArray`. Data type np.int64. Shape (num_boxes, 4).
                "box_labels": `Union[Tuple[np.ndarray], np.ndarray, None]`
                    # Labels of the boxes. Same first dimensionality as the number of boxes.
                "oriented_boxes": `np.ndarray`,
                    # All `OrientedBox`/`OrientedBoxArray`. Data type np.float64. Shape (num_boxes, 5).
                "oriented_box_labels": `Union[Tuple[np.ndarray], np.ndarray, None]`
                    # Similar as "box_labels"
                "points": `np.ndarray` or `None`,
                    # All `Point`/`PointArray`. Data type np.int64. Shape (num_boxes, 2).
                "point_labels": `Union[Tuple[np.ndarray], np.ndarray, None]`
                    # Similar as "box_labels".
                "masks": `List[np.ndarray]`,
                    # All `Mask` objects.
                "mask_confidences": `List[Union[np.ndarray, Tuple[np.ndarray], None]]`,
                    # Similar as "image_labels".
                "polygons": `List[np.ndarray]`,
                    # All `Polygon` objects. Each ndarray is of shape (num_points, 2) and data type np.int64.
                "polygon_labels": `List[Union[np.ndarray, Tuple[np.ndarray], None]]`,
                    # Similar as "image_labels".
            }

        Here is how each type of `Label` will be treated:
        (1) `Empty`:
            Return `None`.
        (2) `ArbitraryHashable`:
            `AttributeError` will be raised if `to_numpy` is not defined in the subclass.
            If defined, converted np.ndarray will be assembled by `np.stack` for object arrays.
        (3) `Scoped`:
            The converted `np.ndarray`(`shape=(, )`) will be assembled by `np.stack` for object arrays.
        (4) `ScopedWithConfidence`:
            The converted two `np.ndarray` (both of `shape=(,)`) will be respectively assembled by `np.stack` for
            object arrays.
        (5) `MultipleScoped`:
            The converted `np.ndarray` (s`hape=(len(scope),)`) will be assembled by `np.stack` for object arrays.
        (6) `ProbabilisticMultipleScoped`:
            The converted `np.ndarray` (`shape=(len(scope),)`) will be assembled by `np.stack` for object arrays.
        (7) `ScopedMaskWithConfidence`:
            It always goes with a `Mask`, so it will not be processed seperately.
    """
    def __init__(self, image_as_feature_map: bool = True, box_format: str = "xxyy"):
        """
        Args:
            `image_as_feature_map` (`bool`, optional): image data of dimension `(H, W, C)` is transposed as `(C, H, W)`.
                Defaults to `True`.
            `box_format` (`str`, optional):
                boxes data coordinates will be returned as:
                    `(xmin, ymin, xmax, ymax)`, if `"xyxy"`
                    `(xmin, xmax, ymin, ymax)`, if `"xxyy"`;
                    `(xmin, ymin, w, h)`, if `"xywh"`;
                    `(x_center, y_center, w, h)`, if `"cxcywh"`;
                Defaults to `"xxyy"`.
        """
        super().__init__()
        self.image_as_feature_map = bool(image_as_feature_map)
        assert box_format in ("xxyy", "xyxy", "xywh", "cxcywh")
        self.box_format = box_format

    @staticmethod
    def transform_label(label: Label):
        if isinstance(label, list):
            labels = [ToNumpy.transform_label(_label) for _label in label]
            if len(label) > 0:
                if isinstance(labels[0], tuple):
                    # [(ndarray, ndarray), (ndarray, ndarray), ...]
                    # => (np.stack([ndarray, ...]), np.stack(ndarray, ...))
                    labels = list(np.stack(arr) for arr in zip(*labels))
                    return labels
                else:
                    return np.stack(labels)
            else:
                return np.empty((0, 0), dtype=np.int64)
        elif isinstance(label, Empty):
            return None
        else:
            label = label.to_numpy()
            return label

    @staticmethod
    def transform_non_composite(image: ImageAnnotation, image_as_feature_map, box_format):
        if isinstance(image, (EmptyImage, Image)) and image_as_feature_map:
            anno_data = image.to_numpy()
            anno_data = anno_data.transpose((2, 0, 1))
        elif isinstance(image, (Box, BoxArray)):
            if box_format == "xxyy":
                anno_data = image.to_numpy()
            elif box_format == "xyxy":
                anno_data = image.to_numpy_as_xyxy()
            elif box_format == "xywh":
                anno_data = image.to_numpy_as_xywh()
            else:
                # box_format == "cxcywh":
                anno_data = image.to_numpy_as_cxcywh()
        else:
            anno_data = image.to_numpy()

        if isinstance(image, Mask) and not isinstance(image.label, ScopedMaskWithConfidence):
            label_data = None
        else:
            label_data = ToNumpy.transform_label(image.label)

        return anno_data, label_data

    @staticmethod
    def transform_image_annotation(image: ImageAnnotation, image_as_feature_map, box_format):
        if isinstance(image, Composite):
            image = image.compact()
            result = {
                "images": [],
                "image_labels": [],
                "boxes": np.empty((0, 4), dtype=np.int64),
                "box_labels": np.empty((0, 0), dtype=np.int64),
                "oriented_boxes": np.empty((0, 5), dtype=np.float64),
                "oriented_box_labels": np.empty((0, 0), dtype=np.int64),
                "points": np.empty((0, 2), dtype=np.int64),
                "point_labels": np.empty((0, 0), dtype=np.int64),
                "masks": [],
                "mask_confidences": [],
                "polygons": [],
                "polygon_labels": [],
            }

            for annotation in image:
                anno_data, label = ToNumpy.transform_image_annotation(annotation, image_as_feature_map, box_format)
                if isinstance(annotation, (Image, EmptyImage)):
                    result["images"].append(anno_data)
                    result["image_labels"].append(label)
                elif isinstance(annotation, Mask):
                    result["masks"].append(anno_data)
                    result["mask_confidences"].append(label)
                elif isinstance(annotation, Polygon):
                    result["polygons"].append(anno_data)
                    result["polygon_labels"].append(label)
                elif isinstance(annotation, PointArray):
                    # only one PointArray after compact()
                    result["points"] = anno_data
                    result["point_labels"] = label
                elif isinstance(annotation, BoxArray):
                    # only one BoxArray after compact()
                    result["boxes"] = anno_data
                    result["box_labels"] = label
                elif isinstance(annotation, OrientedBoxArray):
                    # only one OrientedBoxArray after compact()
                    result["oriented_boxes"] = anno_data
                    result["oriented_box_labels"] = label
                else:
                    raise RuntimeError

            return result
        else:
            return ToNumpy.transform_non_composite(image, image_as_feature_map, box_format)

    def _transform(self, image: ImageAnnotation):
        return self.transform_image_annotation(image, self.image_as_feature_map, self.box_format)

    def _get_inverse(self):
        return EmptyTransform()
