import json
from typing import Optional, Union, IO
from io import BytesIO
from base64 import b64decode, b64encode
from os.path import join, basename

import numpy as np
from PIL import Image as PIL_Image

from .. import Composite, ImageAnnotation, Image, EmptyImage, Point, Box, OrientedBox, Polygon,\
    Scoped, ArbitraryHashable, Empty, ScopedWithConfidence


def from_labelme(label_dict: Union[str, dict, IO], encoding: Optional[str]=None, image_path=None, type_scoped=None) -> Composite:
    """
    Return a Composite from reading annotations created by labelme.

    Args:
        label_dict (dict, str, IO): dict read from .json file created by labelme, or file pointer to
            the json file, or the path to the json file.
        encoding: encoding of the JSON file, effective only when `label_dict` is str.
        image_path (str, optional): Image file of the annotaion.
            If "imageData" is in label_dict, image_path is ignored.
            If None, the image is read from label_dict["imageData"] or created as EmptyImage.
            If not None, image is read from image_path by PIL.Image.open().
            Defaults to None.
        type_scoped (type, optional): a subclass of Scoped to wrap the string labels.
            If None, ArbitraryHashable is used instead. Defaults to None.

    Returns:
        Composite: containing all image and annotations in label_dict.
    """
    if isinstance(label_dict, dict):
        pass
    elif isinstance(label_dict, str):
        assert label_dict.endswith('.json'), "label_dict as path should end with .json"
        with open(label_dict, 'r', encoding=encoding) as fp:
            label_dict = json.load(fp)
    else:
        assert hasattr(label_dict, "read"), "label_dict should be a file pointer"
        label_dict = json.load(label_dict)

    img_w = label_dict['imageWidth']
    img_h = label_dict['imageHeight']

    if "imageData" in label_dict and label_dict["imageData"] is not None:
        image = b64decode(label_dict["imageData"])
        stream = BytesIO(image)
        image = PIL_Image.open(stream)
        image = Image(np.array(image))
        stream.close()
    elif image_path is not None:
        image = PIL_Image.open(image_path)
        image = Image(np.array(image))
    else:
        image = EmptyImage(img_w, img_h)

    assert image.img_w == img_w and image.img_h == img_h,\
        f"Image size mismatch: actual image(img_w={image.img_w}, img_h={image.img_h}) vs"\
        f" labelme field imageWidth={img_w} imageHeight={img_h}."

    if type_scoped is not None:
        assert issubclass(type_scoped, Scoped)
        T = type_scoped
    else:
        T = ArbitraryHashable

    annotations = [image]
    shapes = label_dict["shapes"] if "shapes" in label_dict else []
    for shape in shapes:
        shape_type = shape["shape_type"]
        label = T(shape['label'] if 'label' in shape else None)
        points = np.array(shape["points"])
        if shape_type == "polygon":
            if len(points) == 4:
                points = [Point(p[0], p[1], img_w, img_h, label) for p in points]
                anno = OrientedBox.from_points(points, img_w, img_h)
            elif len(points) > 2:
                anno = Polygon(points, img_w, img_h, label)
            else:
                raise UserWarning("TODO")
        elif shape_type == "rectangle":
            top_left_point = Point(points[0, 0], points[0, 1], img_w, img_h, label)
            bottom_right_point = Point(points[1, 0], points[1, 1], img_w, img_h, label)
            anno = Box.from_corner_points(top_left_point, bottom_right_point)
        elif shape_type == "point":
            anno = Point(points[0, 0], points[0, 1], img_w, img_h, label)
        else:
            # unexpected shape type
            continue

        annotations.append(anno)

    return Composite(annotations, img_w, img_h)


def to_labelme(composite: ImageAnnotation, dst: Union[str, IO, None] = None, image_path=None):
    if not isinstance(composite, Composite):
        composite = Composite([composite])
    composite = composite.flatten()
    num_image = composite.num_image
    if num_image == 0:
        composite.append(EmptyImage(composite.img_w, composite.img_h, 1))
    else:
        assert num_image == 1

    label_dict = {
        "version": "5.1.1",
        "flags": {},
        "imageWidth": composite.img_w,
        "imageHeight": composite.img_h
    }
    shapes = []

    for annotation in composite.annotations:
        if isinstance(annotation, (EmptyImage, Image)):
            image = annotation.to_pil()
            if image_path is None:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                label_dict["imageData"] = b64encode(buffer.getvalue()).decode()
                label_dict["imagePath"] = None
            else:
                assert isinstance(dst, str), "dst must be str if the image is saved to file"
                assert dst.endswith('.json')
                fn = basename(dst)[:-5] + ".png"
                image.save(join(image_path, fn))
                label_dict["imageData"] = None
                label_dict["imagePath"] = fn
        else:
            shape = {"group_id": None, "flags": {}}
            if isinstance(annotation, Box):
                shape["shape_type"] = "rectangle"
                tl, _, _, br = annotation.to_points()
                shape["points"] = [[tl.x, tl.y], [br.x, br.y]]
            elif isinstance(annotation, (OrientedBox, Polygon)):
                shape["shape_type"] = "polygon"
                if isinstance(annotation, OrientedBox):
                    annotation = annotation.to_polygon()
                shape["points"] = [[p.x, p.y] for p in annotation.points]
            elif isinstance(annotation, Point):
                shape["shape_type"] = "point"
                shape["points"] = [[annotation.x, annotation.y]]
            else:
                continue

            label = annotation.label
            if isinstance(label, Empty):
                label = None
            elif isinstance(label, (Scoped, ScopedWithConfidence, ArbitraryHashable)):
                label = str(label.value)
            else:
                raise TypeError
            shape["label"] = label
            shapes.append(shape)

    label_dict["shapes"] = shapes

    if isinstance(dst, str):
        assert dst.endswith('.json'), "label_dict as path should end with .json"
        with open(dst, 'w', encoding="utf-8") as fp:
            json.dump(label_dict, fp)
    elif dst is None:
        return label_dict
    else:
        assert hasattr(dst, "write"), "label_dict should be a file pointer"
        json.dump(label_dict, fp)
