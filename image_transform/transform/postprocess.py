from typing import Union, List
from functools import reduce

import numpy as np

from ..annotation import ImageAnnotation, BoxArray, OrientedBoxArray,\
    Polygon, Composite, PointArray, Point, Box, OrientedBox
from ..annotation.utils import nms
from ..annotation.label import group_objects_by_class, sort_objects_by_confidence, ScopedWithConfidence
from .basic import Transform


class NonMaximumSuppression(Transform):
    def __init__(self, class_aware=True, iou_threshold=0.5, box=True, oriented_box=True, polygon=True):
        """
        Perform NMS on BoxArray/OrientedBoxArray (standalone or in a Composite) and Polygons in Composite.
        If the labels of the objects to be NMS processed is:
            (1) ScopedWithConfidence. The objects are sorted by the confidence.
            (2) ProbabilisticMultipleScoped.
                If `class_aware==True`, the objects are sorted by the corresponding probability.
                Else, the objects are sorted by the max probability.
            (3) For other labels, the objects are taken as already sorted.

        Args:
            class_aware (bool, optional): Whether to perform NMS by class. Defaults to True.
            iou_threshold (float, optional): IoU threshold to decide object duplicate. Within [0, 1]. Defaults to 0.5.
        """
        super().__init__()
        self.class_aware = class_aware
        assert 0.0 <= iou_threshold <= 1.0
        self.iou_t = iou_threshold
        self.box = bool(box)
        self.oriented_box = bool(oriented_box)
        self.polygon = bool(polygon)

    @staticmethod
    def nms_class_agnostic(objects: Union[BoxArray, OrientedBoxArray, List[Polygon]], iou_t):
        objects = sort_objects_by_confidence(objects)
        if len(objects) <= 1:
            return objects
        if isinstance(objects, (BoxArray, OrientedBoxArray)):
            return objects.nms(iou_t)
        else:
            duplicate_matrix = np.eye(len(objects), dtype=bool)
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    if objects[i].iou(objects[j]) >= iou_t:
                        duplicate_matrix[i, j] = True
                        duplicate_matrix[j, i] = True
            kept = nms(duplicate_matrix)
            return [obj for i, obj in enumerate(objects) if kept[i]]

    def nms(self, objects: Union[BoxArray, OrientedBoxArray, List[Polygon]]):
        if self.class_aware:
            class_wise_objects = [
                NonMaximumSuppression.nms_class_agnostic(_objects, self.iou_t)
                for _objects in group_objects_by_class(objects).values()
            ]
            if isinstance(objects, BoxArray):
                init = BoxArray.from_boxes([], objects.img_w, objects.img_h)
            elif isinstance(objects, OrientedBoxArray):
                init = OrientedBoxArray.from_oriented_boxes([], objects.img_w, objects.img_h)
            else:
                init = []
            return reduce(lambda x, y: x + y, class_wise_objects, init)
        else:
            return NonMaximumSuppression.nms_class_agnostic(objects, self.iou_t)

    def _transform(self, image: ImageAnnotation):
        if isinstance(image, Composite):
            image = image.compact()
            new_annotations = []
            polygons = []
            for anno in image.annotations:
                if isinstance(anno, BoxArray) and self.box:
                    new_annotations.append(self.nms(anno))
                elif isinstance(anno, OrientedBoxArray) and self.oriented_box:
                    new_annotations.append(self.nms(anno))
                elif isinstance(anno, Polygon):
                    polygons.append(anno)
                else:
                    new_annotations.append(anno)

            if len(polygons) > 0 and self.polygon:
                polygons = self.nms(polygons)
            new_annotations.extend(polygons)

            return Composite(new_annotations, image.img_w, image.img_h)
        elif isinstance(image, BoxArray):
            return Composite(self.nms(image), image.img_w, image.img_h).boxes
        elif isinstance(image, OrientedBoxArray):
            return Composite(self.nms(image), image.img_w, image.img_h).oriented_boxes
        else:
            return image


class ConfidenceThreshold(Transform):
    def __init__(self, threshold: float):
        """
            Filter image annotation by thresholding the confidence.
            ImageAnnotation & Label types to be transformed:
            (1) {Point/Box/OrientedBox}Array with a list of ScopedWithConfidence label.
                Singleton ImageAnnotation in the array whose confidence below the threshold will be removed.
            (2) Composite. For annotations in the Composite:
                (2-1) Box, Point, OrientedBox, Polygon with ScopedWithConfidence label.
                    Will be removed if the confidence is below the threshold.
                (2-2) {Point/Box/OrientedBox}Array. Processed as (1).
            Other annotations are unmodified.
        """
        super().__init__()
        assert 0 <= threshold <= 1
        self.threshold = threshold

    def _transform(self, image: ImageAnnotation):
        if isinstance(image, (PointArray, BoxArray, OrientedBoxArray)) and\
                len(image) > 0 and isinstance(image.label[0], ScopedWithConfidence):
            return image.select([i for i, label in enumerate(image.label) if label.confidence >= self.threshold])
        elif isinstance(image, Composite):
            annotations = []
            for anno in image:
                if isinstance(anno, (Box, Point, OrientedBox, Polygon))\
                        and isinstance(anno.label, ScopedWithConfidence) and anno.label.confidence >= self.threshold:
                    annotations.append(anno)
                elif isinstance(anno, (BoxArray, PointArray, OrientedBoxArray)):
                    annotations.append(self._transform(anno))
                else:
                    annotations.append(anno)
            return Composite(annotations, image.img_w, image.img_h)
        else:
            return image
