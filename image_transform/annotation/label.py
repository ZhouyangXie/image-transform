from abc import ABC, abstractmethod, abstractclassmethod
from copy import deepcopy

import cv2
import numpy as np

from typing import Hashable, Iterable, Tuple, Union, Type

from .utils import rotate_image, pad_image

"""
    Subclasses of Label are semantic annotations on the instance of ImageAnnotation.
    The basic semantic annotations are hashable values, i.e. any object hashable.
    It also provides to_numpy() and from_numpy() to work in accord with ImageAnnotation.to/from_numpy().
"""


def is_hashable(value):
    try:
        if hasattr(value, "__hash__"):
            _ = hash(value)
            return True
        else:
            return False
    except TypeError:
        return False


class Label(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __eq__(self, another):
        """
            Values of the labels should be compared
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_numpy(self):
        """
            Convert the label to a (tuple of) np.ndarray.
            This behaviour should be symmetric to Label.from_numpy().
        """
        pass

    @abstractclassmethod
    def from_numpy(cls, ndarray: np.ndarray):
        """
            Convert a (tuple of) np.ndarray to a label
            This behaviour should be symmetric to to_numpy().
        """
        pass

    @staticmethod
    def hashable2str(hashable: Hashable) -> str:
        return str(hashable)

    def copy(self):
        return deepcopy(self)

    # geometric transformations

    def _clip(self):
        return self

    def _pad(self, up, down, left, right, fill_value=None):
        return self

    def _crop(self, xmin, xmax, ymin, ymax):
        return self

    def _horizontal_flip(self):
        return self

    def _vertical_flip(self):
        return self

    def _rotate(self, angle):
        return self

    def _rotate_right_angle(self, rotate_right_angle):
        return self

    def _resize(self, dst_w, dst_h):
        return self

    def _transpose(self):
        return self


class Empty(Label):
    """
        This type of label stands for the object is not labelled.
        Methods from_numpy/to_numpy are not defined.
    """
    def __eq__(self, another):
        return isinstance(another, Empty)

    def __repr__(self):
        return "None"

    def to_numpy(self):
        raise AttributeError("method not defined")

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        raise AttributeError("method not defined")


class ArbitraryHashable(Label):
    """
        The label value can be an arbitrary hashable value.
        Methods from_numpy/to_numpy are not defined.
    """
    def __init__(self, value: Hashable):
        assert is_hashable(value)
        self.value = value
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return self.value == another.value

    def __repr__(self):
        return Label.hashable2str(self.value)

    def to_numpy(self):
        raise AttributeError("method not defined")

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        raise AttributeError("method not defined")


class Scoped(Label):
    """
        Classification label that is hashable and belongs to a scope(set).
        Class attribute `scope` should be specified in subclasses.
    """
    scope = []

    def __init__(self, value: Hashable):
        """
        Args:
            value (Hashable): one of Scoped.scope
        """
        assert is_hashable(value)
        assert value in self.scope
        self.value = value
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return self.value == another.value

    def __repr__(self):
        return Label.hashable2str(self.value)

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the index of self.value in self.scope. Singleton, type = np.int64
        """
        return np.int64(self.scope.index(self.value))

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        """
        Args:
            ndarray (np.ndarray): scalar or singleton ndarray that represents the index, type = np.int64
        """
        assert (np.isscalar(ndarray) or ndarray.size == 1) and ndarray.dtype == np.int64
        return cls(cls.scope[ndarray.item()])

    derived_multiple_scoped_type = None

    def to_multiple_scoped(self, T: Type = None):
        """
            convert to singleton MultipleScoped label
        Args:
            T (Type): a subclass of MultipleScoped in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _MultipleScopedVariantOf{type(self).__name__} object will be returned.
        """
        if T is None:
            if type(self).derived_multiple_scoped_type is None:
                type(self).derived_multiple_scoped_type = type(
                    f"_MultipleScopedVariantOf{type(self).__name__}", (MultipleScoped, ), {"scope": type(self).scope})
            T = type(self).derived_multiple_scoped_type
        assert issubclass(T, MultipleScoped)
        assert T.scope == self.scope
        return T([self.value])

    derived_probabilistic_multiple_scoped_type = None

    def to_probabilistic_multiple_scoped(self, T: Type = None):
        """
            Convert to ProbabilisticMultipleScoped (1.0 for self.value and 0 for others)
        Args:
            T (Type): a subclass of ProbabilisticMultipleScoped in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _ProbabilisticMultipleScopedVariantOf{type(self).__name__}
                    object will be returned.
        """
        if T is None:
            if type(self).derived_probabilistic_multiple_scoped_type is None:
                type(self).derived_probabilistic_multiple_scoped_type = type(
                    f"_ProbabilisticMultipleScopedVariantOf{type(self).__name__}",
                    (ProbabilisticMultipleScoped, ),
                    {"scope": type(self).scope}
                )
            T = type(self).derived_probabilistic_multiple_scoped_type
        assert issubclass(T, ProbabilisticMultipleScoped)
        assert T.scope == self.scope
        probs = np.zeros(len(self.scope), dtype=np.float32)
        probs[self.scope.index(self.value)] = 1
        return T(probs)


class ScopedWithConfidence(Label):
    """
        Scope plus a confidence score (float in [0, 1])
    """
    scope = []

    def __init__(self, value: Hashable, confidence: float):
        """
        Args:
            value (Hashable): one of self.scope.
            confidence (float): within interval [0, 1].
        """
        assert is_hashable(value)
        assert value in self.scope
        assert 0 <= confidence <= 1
        self.value = value
        self.confidence = confidence
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return self.value == another.value and np.allclose(self.confidence, another.confidence)

    def __repr__(self):
        return Label.hashable2str(self.value) + f":{self.confidence:.2f}"

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: value(np.int64) and confidence(np.float32), both singleton
        """
        return np.int64(self.scope.index(self.value)), np.float32(self.confidence)

    @classmethod
    def from_numpy(cls, ndarray: Tuple[np.ndarray, np.ndarray]):
        """
        Args:
            ndarray (np.ndarray): singleton int array that represents the index
        """
        value, confidence = ndarray
        assert (value.shape == () or value.shape == (1, )) and value.dtype == np.int64
        assert (confidence.shape == () or confidence.shape == (1, )) and confidence.dtype == np.float32
        assert confidence.shape == () or confidence.shape == (1, )
        return cls(value=cls.scope[value.item()], confidence=confidence.item())

    derived_probabilistic_multiple_scoped_type = None

    def to_probabilistic_multiple_scoped(self, T):
        """
            Convert to ProbabilisticMultipleScoped (self.confidence for self.value and same confidence for others)
        Args:
            T (Type): a subclass of ProbabilisticMultipleScoped in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _ProbabilisticMultipleScopedVariantOf{type(self).__name__}
                    object will be returned.
        """
        if T is None:
            if type(self).derived_probabilistic_multiple_scoped_type is None:
                type(self).derived_probabilistic_multiple_scoped_type = type(
                    f"_ProbabilisticMultipleScopedVariantOf{type(self).__name__}",
                    (ProbabilisticMultipleScoped, ),
                    {"scope": type(self).scope}
                )
            T = type(self).derived_probabilistic_multiple_scoped_type
        assert issubclass(T, ProbabilisticMultipleScoped)
        assert T.scope == self.scope
        probs = np.zeros(len(self.scope), dtype=np.float32)
        if len(self.scope) > 1:
            probs.fill((1 - self.confidence)/(len(self.scope) - 1))
            probs[self.scope.index(self.value)] = self.confidence
        else:
            probs.fill(1)
        return T(probs)


class MultipleScoped(Label):
    """
        Scoped with multiple non-repeative values. The values are ordered.
    """
    scope = []

    def __init__(self, values: Iterable[Hashable]):
        """
        Args:
            values (Iterable[Hashable]): a set of labels in self.scope. Can have repeated values but ignored.
        """
        assert all((is_hashable(v) and v in self.scope) for v in values)
        self.values = list(set(values))
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return set(self.values) == set(another.values)

    def __repr__(self):
        return str(set(self.values))

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            ndarray (np.ndarray): the 0/1 encoding of the labels, shape = (len(self.scope), ), type = np.int64
        """
        x = np.zeros(len(self.scope), dtype=np.int64)
        x[[self.scope.index(v) for v in self.values]] = 1
        return x

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        """
        Args:
            ndarray (np.ndarray): vector where nonzero values represent true for the label, shape = (len(cls.scope), )
        """
        assert ndarray.shape == (len(cls.scope), )
        indices = np.nonzero(ndarray)[0].tolist()
        return cls([cls.scope[i] for i in indices])

    derived_scoped_type = None

    def to_scoped(self, T: Type = None):
        """
            convert singleton MultipleScoped to Scoped label
        Args:
            T (Type): a subclass of Scoped in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _ScopedVariantOf{type(self).__name__} object will be returned.
        """
        assert len(self.values) == 1, "Only singleton MultipleScoped can be converted to Scoped."
        if T is None:
            if type(self).derived_scoped_type is None:
                type(self).derived_scoped_type = type(
                    f"_ScopedVariantOf{type(self).__name__}",
                    (Scoped, ),
                    {"scope": type(self).scope}
                )
            T = type(self).derived_scoped_type
        assert issubclass(T, Scoped)
        assert T.scope == self.scope
        return T(self.values[0])


class ProbabilisticMultipleScoped(Label):
    """
        The label is a vector that represents a discrete probabilistic distribution of exclusive labels,
        same length and order as self.scope.
    """

    scope = []

    def __init__(self, probs: Union[Iterable[float], np.ndarray]):
        """
        Args:
            probs (Union[Iterable[float], np.ndarray]): len(self.scope) non-negative float values.
        """
        probs = np.array(probs, dtype=np.float32)
        assert probs.shape == (len(self.scope), ) and np.all(probs >= 0)
        self.probs = probs/probs.sum()
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return np.all(self.probs == another.probs)

    def __repr__(self):
        return ",".join(Label.hashable2str(s) + f":{p:.2f}" for s, p in zip(self.scope, self.probs))

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            np.ndarray shape = (len(self.scope), ), type = np.float32
        """
        return self.probs

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        """
        Args:
            ndarray (np.ndarray): same as __init__.
        """
        return cls(ndarray)

    derived_scoped_type = None

    def to_scoped(self, T: Type = None):
        """
            Convert to Scoped (argmax label will be the label value)
        Args:
            T (Type): a subclass of Scope in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _ScopedVariantOf{type(self).__name__} object will be returned.
        """
        if T is None:
            if type(self).derived_scoped_type is None:
                type(self).derived_scoped_type = type(
                    f"_ScopedVariantOf{type(self).__name__}",
                    (Scoped, ),
                    {"scope": type(self).scope}
                )
            T = type(self).derived_scoped_type
        assert issubclass(T, Scoped)
        assert T.scope == self.scope
        return T(self.scope[np.argmax(self.probs)])

    derived_scoped_with_confidence_type = None

    def to_scoped_with_confidence(self, T: Type = None):
        """
            Convert to ScopedWithConfidence (argmax label and prob will be the label value)
        Args:
            T (Type): a subclass of ScopedWithConfidence in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _ScopedWithConfidenceVariantOf{type(self).__name__}
                    object will be returned.
        """
        if T is None:
            if type(self).derived_scoped_with_confidence_type is None:
                type(self).derived_scoped_with_confidence_type = type(
                    f"_ScopedWithConfidenceVariantOf{type(self).__name__}",
                    (ScopedWithConfidence, ),
                    {"scope": type(self).scope}
                )
            T = type(self).derived_scoped_with_confidence_type
        assert issubclass(T, ScopedWithConfidence)
        assert T.scope == self.scope
        i = np.argmax(self.probs)
        return T(self.scope[i], self.probs[i])

    derived_multiple_scoped_type = None

    def to_multiple_scoped(self, T: Type = None, threshold: float = 0.5):
        """
            Convert to MultipleScoped (prob above or equal to threshold will be preserved)
        Args:
            T (Type): a subclass of MultipleScoped in which only T.scope is overriden.
                The overriden scope must the same as this class.
                If None, a runtime defined type _MultipleScopedVariantOf{type(self).__name__}
                    object will be returned.
            threshold (float): values with probability above or equal to the thresold will be preserved.
                Defaults to 0.5.
        """
        if T is None:
            if type(self).derived_multiple_scoped_type is None:
                type(self).derived_multiple_scoped_type = type(
                    f"_MultipleScopedVariantOf{type(self).__name__}", (MultipleScoped, ), {"scope": type(self).scope})
            T = type(self).derived_multiple_scoped_type
        assert issubclass(T, MultipleScoped)
        assert T.scope == self.scope
        return T([v for p, v in zip(self.probs, self.scope) if p >= threshold])


class ScopedMaskWithConfidence(Label):
    """
        Pixel-wise confidence for annotation.Mask. This class is special, because the associated Mask object itself is
         part of the label. But ScopedMaskWithConfidence objects have no access to the associated Mask object.
        self.scope and a confidence matrix(same shape as the mask, dtype=np.float32) is maintained, value within [0, 1].
    """
    scope = []

    def __init__(self, confidence: np.ndarray):
        assert isinstance(confidence, np.ndarray) and confidence.ndim == 2 and confidence.dtype == np.float32
        assert np.all((1 >= confidence) & (confidence >= 0))
        self.confidence = confidence
        super().__init__()

    def __eq__(self, another):
        assert isinstance(another, type(self))
        return np.all(self.confidence == another.confidence)

    def __repr__(self):
        return str(self.confidence)

    def to_numpy(self) -> np.ndarray:
        """
        Returns:
            (np.ndarray): same shape as the associated Mask.mask, dtype=np.float32
        """
        return self.confidence

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray):
        """
        Args:
            ndarray (np.ndarray): same shape as the associated Mask.mask, dtype=np.float32
        """
        return cls(ndarray)

    # geometric transformation

    def _pad(self, up, down, left, right, fill_value=None):
        return type(self)(pad_image(self.confidence, up, down, left, right, 0))

    def _crop(self, xmin, xmax, ymin, ymax):
        return type(self)(self.confidence[ymin:ymax + 1, xmin:xmax + 1])

    def _horizontal_flip(self):
        return type(self)(self.confidence[:, ::-1])

    def _vertical_flip(self):
        return type(self)(self.confidence[::-1, :])

    def _rotate(self, angle):
        return type(self)(rotate_image(self.confidence, angle))

    def _rotate_right_angle(self, rotate_right_angle):
        if rotate_right_angle == 90:
            return type(self)(self.confidence.transpose((1, 0))[:, ::-1])
        elif rotate_right_angle == 180:
            return type(self)(self.confidence[::-1, ::-1])
        elif rotate_right_angle == 270:
            return type(self)(self.confidence.transpose((1, 0))[::-1, :])
        else:
            return type(self)(self.confidence)

    def _resize(self, dst_w, dst_h):
        return type(self)(cv2.resize(self.confidence, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST))

    def _transpose(self):
        return type(self)(self.confidence.transpose((1, 0)))


def group_objects_by_class(objects, scope=None):
    """
        Group image annotations by labels. The argument `objects` is either:
        (1) A list of objects of type ImageAnnotation except BoxArray/PointArray/OrientedBoxArray.
        (2) BoxArray/PointArray/OrientedBoxArray.
        If (1), the labels should all have a hashable `label.value`.
        If` label.value` is in `scope`, the annotation will be grouped under the key of this value.
        Else, the annotation will not show in the result.
        Returns a `dict` will keys same as `scope`, annotations are grouped by label value.
    """
    labels = [obj.label for obj in objects]
    indices_dict = dict()
    if scope is not None:
        for v in scope:
            indices_dict[v] = []

    for i, label in enumerate(labels):
        if isinstance(label, MultipleScoped):
            for v in label.values:
                if scope is None or v in scope:
                    if v in indices_dict:
                        indices_dict[v].append(i)
                    else:
                        indices_dict[v] = [i]
        elif isinstance(label, (Scoped, ScopedWithConfidence, MultipleScoped, ArbitraryHashable)):
            if scope is None or label.value in scope:
                if label.value in indices_dict:
                    indices_dict[label.value].append(i)
                else:
                    indices_dict[label.value] = [i]

    return dict(
        (k, [objects[i] for i in indices] if isinstance(objects, list) else objects.select(indices))
        for k, indices in indices_dict.items()
    )


def sort_objects_by_confidence(objects):
    """
        Sort image annotations by confidence of labels descentally. The argument `objects` is either:
        (1) A list of objects of type ImageAnnotation except Box/Point/OrientedBoxArray.
        (2) BoxArray/PointArray/OrientedBoxArray.
        (3) An empty list(return the same empty list).
        The labels should be of the same type, either:
        (1) ScopedWithConfidence. The objects are sorted by the confidence.
        (2) Others. No sorting is performed. The same input objects are returned.
    """
    if len(objects) <= 1 or not isinstance(objects[0].label, ScopedWithConfidence):
        return objects

    confidences = [obj.label.confidence for obj in objects]
    sorted_indices = np.argsort(confidences)[::-1]
    if isinstance(objects, list):
        return [objects[i] for i in sorted_indices]
    else:
        return objects.select(sorted_indices)
