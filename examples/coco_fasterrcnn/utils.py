from typing import Union, Dict, List
from image_transform.transform import all_names, SequentialTransforms, Transform,\
    MinibatchTransform, SequentialMinibatchTransforms


def parse_config(transform: Union[Dict, List]) -> Union[Transform, MinibatchTransform]:
    if isinstance(transform, dict):
        assert "type" in transform
        transform = transform.copy()
        transform_type_name = transform.pop("type")
        assert transform_type_name in all_names
        return all_names[transform_type_name](**transform)
    elif isinstance(transform, list):
        assert len(transform) > 0
        transforms = [parse_config(t) for t in transform]
        if isinstance(transforms[0], MinibatchTransform):
            return SequentialMinibatchTransforms(transforms)
        else:
            return SequentialTransforms(transforms)
    else:
        raise ValueError
