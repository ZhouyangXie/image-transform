from typing import List, Dict, Union
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from image_transform.annotation import BoxArray
from image_transform.metric.detection import AccuracyWithConfidence

from data import COCODetection


num_epoch = 1
batch_size = 4
device = "cpu"
sgd_lr = 1e-4
# to be set
get_minibatch_transform = None
get_post_processing_transform = None


def train(model, train_dataset: COCODetection, validation_dataset: COCODetection):
    optimizer = SGD(model.parameters(), lr=sgd_lr)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate)
    model = model.train()
    model = model.to(device)

    for epoch in range(num_epoch):
        for i, data_and_transforms in enumerate(dataloader):
            input_images, _ = data_and_transforms  # input_images: List[Composite]

            # from ImageAnnotation to dict of np.ndarray
            data: List[Dict[str, np.ndarray]] = get_minibatch_transform().transform(input_images)

            # from np.ndarray to torch.Tensor
            images: torch.Tensor = torch.from_numpy(data['images'][0]).to(device=device)
            # Here data['images'][0] because Stack allows multiple images in one Composite.
            # In this example, only one image presents, so directly access the 0-th object.
            boxes: List[torch.Tensor] =\
                [torch.from_numpy(boxes.astype(np.float32)).to(device=device) for boxes in data['boxes']]
            labels: List[torch.Tensor] =\
                [torch.from_numpy(labels.astype(np.int64)).to(device=device) for labels in data['box_labels']]

            # Forward/backward. This step depends on what the model input requires.
            # For the input format of Faster-RCNN implemented by torchvision, see:
            # https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
            targets = [{"boxes": boxes[i], "labels": labels[i]} for i in range(batch_size)]

            loss = model(images, targets)

            # compute loss, gradients and updated parameters
            loss = loss["loss_classifier"] + loss["loss_box_reg"] + loss["loss_objectness"] + loss["loss_rpn_box_reg"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(f"epoch {epoch} iteration {i} loss {loss.item():.4e}")

        scores: Dict[str, Union[np.ndarray, float, int]] = validate(model, validation_dataset)
        mAP: float = scores["mean average precision"]
        logging.info(f"epoch {epoch} validation mAP {mAP:.2f}")

    return model


def validate(model, dataset: COCODetection):
    # in this function, we omit tons of type annotation and comments to avoid duplication of those in train()
    model = model.to(device)
    model = model.eval()
    metric = AccuracyWithConfidence(dataset.scope, iou_thresholds=0.5)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)

    for data_and_transforms in tqdm(dataloader):
        input_images, transforms = data_and_transforms  # data: List[Composite], transforms: List[Transform]

        x = get_minibatch_transform().transform(input_images)
        x = torch.from_numpy(x['images'][0]).to(device=device)
        with torch.no_grad():
            outputs: List[Dict[str, torch.Tensor]] = model(x)
            # type of outputs: [
            #   {
            #       "boxes": Tensor(shape=(N, 4), dtype=float32),
            #       "labels": Tensor(shape=(N, ), dtype=int64),
            #       "scores": Tensor(shape=(N, ), dtype=float32)
            #   },
            #   ...
            # ]

        # convert model outputs to predictions (a list of BoxArray with ScopedWithConfidence labels)
        # `output['labels'] - 1` because torchvision faster-rcnn assumes label=0 to be the background,
        # while label=1 stands for the 0-th category
        predictions = dataset.model_minibatch_output_to_predictions(
            minibatch_boxes=[output['boxes'] for output in outputs],
            minibatch_labels=[output['labels'] - 1 for output in outputs],
            minibatch_scores=[output['scores'] for output in outputs],
            minibatch_input_images=input_images,
        )

        # post-process the predictions
        predictions = [get_post_processing_transform().transform(p) for p in predictions]

        # get the targets (extract BoxArray from Composite)
        targets: List[BoxArray] = [image.boxes for image in input_images]

        # inverse transformation on the predictions and targets
        inverse_transforms = [transform.get_inverse() for transform in transforms]
        targets = [it(t) for t, it in zip(targets, inverse_transforms)]
        predictions = [it(p) for p, it in zip(predictions, inverse_transforms)]

        # update the metric
        metric.update(predictions, targets)

    return metric.get_score()
