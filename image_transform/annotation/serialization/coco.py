import json
import collections
from os.path import join
from typing import Optional, Union, IO

from PIL import Image as PIL_Image

from .. import Composite, Image, EmptyImage, Box, Scoped, ScopedWithConfidence


class _CocoObjectDetectionAccessor(collections.abc.Sequence):
    def __init__(self, scope, images, image_dir):
        self.scope = scope
        self.prediction_label_type = type(
            "TempCOCOScopedWithConfidence", (ScopedWithConfidence, ), {"scope": scope}
        )
        self.images = images
        self.image_dir = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.image_dir is None:
            image = EmptyImage(self.images[i]["width"], self.images[i]["height"])
        else:
            image = Image.from_pil(PIL_Image.open(join(self.image_dir, self.images[i]["file_name"])))
        boxes = self.images[i]["box"]
        return Composite([image] + boxes).compact()


def make_coco_object_detection_accessor(annotation_file: Union[str, dict, IO], encoding: Optional[str]=None, image_dir=None):
    """
    Return a list of Composite objects from reading annotations created by labelme.

    Args:
        annotation_file (dict, str, IO): dict read from .json annotation file, or file pointer to
            the json file, or the path to the json file.
        encoding: encoding of the JSON file, effective only when `label_dict` is str.
        image_dir (str, optional): Image file directory.
            If None, the image is set to an EmptyImage.
            If not None, image is read from join(image_dir, filename) by PIL.Image.open.
            Defaults to None.

    Returns:
        Sequence: a read-only list to access image with annotations by index
    """
    if isinstance(annotation_file, dict):
        pass
    elif isinstance(annotation_file, str):
        assert annotation_file.endswith('.json'), "label_dict as a path should ends with .json"
        with open(annotation_file, "r", encoding=encoding) as fp:
            annotation_file = json.load(fp)
    else:
        assert hasattr(annotation_file, "read"), "label_dict should be a file pointer"
        annotation_file = json.load(annotation_file)

    images = dict(
        (im["id"], {"file_name": im["file_name"], "width": im["width"], "height": im["height"], "box": []})
        for im in annotation_file["images"]
    )

    ids = [c["id"] for c in annotation_file["categories"]]
    names = [c["name"] for c in annotation_file["categories"]]
    category_id2name = dict(zip(ids, names))

    max_id = max(int(_id) for _id in ids)
    scope = [f"UNKOWN ID {i}" for i in range(1, max_id + 1)]
    for _id, name in zip(ids, names):
        scope[int(_id) - 1] = name

    type_scoped = type("TempCOCOScoped", (Scoped, ), {"scope": scope})

    for a in annotation_file["annotations"]:
        xmin, ymin, w, h = a["bbox"]
        xmax, ymax = int(xmin + w), int(ymin + h)
        xmax = xmax if xmax > xmin else xmin + 1
        ymax = ymax if ymax > ymin else ymin + 1
        img_w, img_h = images[a["image_id"]]["width"], images[a["image_id"]]["height"]
        class_name = category_id2name[a["category_id"]]
        label = type_scoped(class_name)
        images[a["image_id"]]["box"].append(Box(xmin, xmax, ymin, ymax, img_w, img_h, label))

    images = list(images.values())
    return _CocoObjectDetectionAccessor(scope, images, image_dir)
