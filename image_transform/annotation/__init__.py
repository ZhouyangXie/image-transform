from .basic import ImageAnnotation
from .image import Image, EmptyImage
from .point import Point
from .point_array import PointArray
from .box import Box
from .box_array import BoxArray
from .oriented_box import OrientedBox
from .oriented_box_array import OrientedBoxArray
from .polygon import Polygon
from .mask import Mask
from .composite import Composite
from .utils import IsWithinImage
from .label import Label, Empty, ArbitraryHashable, Scoped, ScopedWithConfidence, MultipleScoped,\
    ScopedMaskWithConfidence, ProbabilisticMultipleScoped


all_names = {
    ImageAnnotation.__name__: ImageAnnotation,
    Image.__name__: Image,
    EmptyImage.__name__: EmptyImage,
    Point.__name__: Point,
    PointArray.__name__: PointArray,
    Box.__name__: Box,
    BoxArray.__name__: BoxArray,
    OrientedBox.__name__: OrientedBox,
    OrientedBoxArray.__name__: OrientedBoxArray,
    Polygon.__name__: Polygon,
    Mask.__name__: Mask,
    Composite.__name__: Composite,
    IsWithinImage.__name__: IsWithinImage,
    IsWithinImage.__name__+".YES": IsWithinImage.YES,
    IsWithinImage.__name__+".PARTIAL": IsWithinImage.PARTIAL,
    IsWithinImage.__name__+".NO": IsWithinImage.NO,
    Label.__name__: Label,
    Empty.__name__: Empty,
    ArbitraryHashable.__name__: ArbitraryHashable,
    Scoped.__name__: Scoped,
    ScopedWithConfidence.__name__: ScopedWithConfidence,
    MultipleScoped.__name__: MultipleScoped,
    ScopedMaskWithConfidence.__name__: ScopedMaskWithConfidence,
    ProbabilisticMultipleScoped.__name__: ProbabilisticMultipleScoped,
}


__all__ = list(all_names.keys())
