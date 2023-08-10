import collections
from os.path import join
from typing import Union, IO
from xml.etree.ElementTree import parse as ET_parse

from PIL import Image as PIL_Image

from .. import Composite, Image, EmptyImage, Box, Scoped


def _parse_voc_xml(xml_file):
    if isinstance(xml_file, str):
        node = ET_parse(xml_file).getroot()
    else:
        node = xml_file
    voc_dict = dict()
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(_parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


class PascalVocScoped(Scoped):
    scope = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def from_voc(
        label_dict: Union[str, dict, IO], image_dir=None, mask_dir=None,
        difficult=False, truncated=True, type_scoped=PascalVocScoped
        ) -> Composite:
    """
    Return a Composite from reading annotations created by labelme.

    Args:
        label_dict (dict, str, IO): dict read from .json file created by labelme, or file pointer to
            the json file, or the path to the json file.
        image_path (str, optional): Image file directory.
            If None, the image is set to an EmptyImage.
            If not None, image is read from join(image_dir, filename) by PIL.Image.open.
            Defaults to None.
        mask_dir (str, optional): mask annotation file directory.
            If None, the mask will not be read.
            If not None, mask is read from join(mask_dir, filename) by PIL.Image.open if segmented.
            Defaults to None.
        type_scoped (type, optional): a subclass of Scoped to wrap the string labels.
            Defaults to PascalVocScoped(20 classes of Pascal VOC).

    Returns:
        Composite: containing all image and annotations in label_dict.
    """
    if isinstance(label_dict, dict):
        pass
    elif isinstance(label_dict, str):
        assert label_dict.endswith('.xml'), "label_dict as a path should ends with .xml"
        label_dict = _parse_voc_xml(label_dict)
    else:
        assert hasattr(label_dict, "read"), "label_dict should be a file pointer"
        label_dict = _parse_voc_xml(label_dict)

    label_dict = label_dict["annotation"]
    annotations = []

    size = label_dict["size"]
    img_w, img_h, channels = int(size["width"]), int(size["height"]), int(size["depth"])
    if image_dir is not None:
        image_file_name = label_dict["filename"]
        image = PIL_Image.open(join(image_dir, image_file_name))
        image = Image.from_pil(image)
        assert img_w == image.img_w and img_h == image.img_h and channels == image.channels,\
            f"Image size in annotation(w={img_w}, h={img_h}, c={channels}) file mismatches"\
            f" that of the image file(w={image.img_w}, h={image.img_h}, c={image.channels})"
    else:
        image = EmptyImage(img_w, img_h)
    annotations.append(image)

    if mask_dir is not None and label_dict['segmented'] != "0":
        # TODO
        raise NotImplementedError("VOC mask parsing will be implemented soon...")

    for obj in label_dict["object"]:
        if not truncated and obj["truncated"] == "1":
            continue
        if not difficult and obj["difficult"] == "1":
            continue
        class_name = obj["name"]
        label = type_scoped(class_name)
        # TODO: ignore pose estimation?
        # pose = obj["pose"]
        xmin = int(obj["bndbox"]["xmin"])
        xmax = int(obj["bndbox"]["xmax"])
        ymin = int(obj["bndbox"]["ymin"])
        ymax = int(obj["bndbox"]["ymax"])
        annotations.append(Box(xmin, xmax, ymin, ymax, img_w, img_h, label))

    return Composite(annotations)
