from functools import partial

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pprint import pprint

import train_test
from data import COCODetection
from config import transforms
from utils import parse_config


def main():
    train_test.device = "cuda"
    train_test.batch_size = 1
    train_test.get_minibatch_transform = partial(parse_config, transforms["minibatch_transform"])
    train_test.get_post_processing_transform = partial(parse_config, transforms["test_post_processing"])

    # prepare dataset
    train_dataset = COCODetection(
        annotation_file="path/to/coco/instances_val2017.json",
        image_path="path/to/coco/val2017",
        get_transform=partial(parse_config, transforms["training_transform"]),
    )
    validation_dataset = COCODetection(
        annotation_file="path/to/coco/instances_val2017.json",
        image_path="path/to/coco/val2017",
        get_transform=partial(parse_config, transforms["test_transform"]),
    )

    # prepare model
    # len(train_dataset.scope) + 1 because torchvision faster-rcnn assume label = 0 to be the background
    model = fasterrcnn_resnet50_fpn(num_classes=len(train_dataset.scope) + 1, weights_backbone=None)
    model.load_state_dict(torch.load("some/pretrained/weight.pth", "cpu"))
    # model = train_test.train(model, train_dataset, validation_dataset)
    scores = train_test.validate(model, validation_dataset)
    pprint(scores)

    # save the model
    pass


if __name__ == "__main__":
    main()
