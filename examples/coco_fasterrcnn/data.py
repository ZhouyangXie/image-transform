from typing import Callable, Sequence, List, Tuple, Union

import torch
from torch.utils.data import Dataset
import numpy as np

from image_transform.annotation import Composite, BoxArray
from image_transform.annotation.serialization import make_coco_object_detection_accessor
from image_transform.transform import Transform


# coco_2017_classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#     "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
#     "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag",
#     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon",
#     "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#     "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
#     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#     "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
# ]
# # these class names are stored in COCO annotation files


class COCODetection(Dataset):
    def __init__(
            self, annotation_file: str, image_path: str,
            get_transform: Callable[[None], Transform] = None,
           ):
        """
        Args:
            annotation_path (str): path to .json annotation file
            image_path (str): path to images
            get_transform (callable, optional):
                Called to get single example Transform in __getitem__().
            get_minibatch_transform (callable, optional):
                Called to get the transformation for minibatch data during training/validation.
        """
        self.accessor: Sequence = make_coco_object_detection_accessor(annotation_file, image_path)
        self.scope = self.accessor.scope
        self.get_transform: Callable = get_transform

    def __len__(self):
        return len(self.accessor)

    def __getitem__(self, index: int):
        # load image and annotations
        annotation: Composite = self.accessor[index]
        # initialize one Transform object for each image/annotation
        transform: Transform = self.get_transform()
        # transform
        annotation: Composite = transform(annotation)
        # assembly all boxes together
        annotation = annotation.compact()
        # return the transformed annotation with the Transform object
        # the Transform object is useful to make an inverse transformation
        return annotation, transform

    @staticmethod
    def collate(annotation_and_transform_pairs: List[Tuple[Composite, Transform]]):
        """
            Use in Dataloader as collate_fn
            [(annotation, transform), (annotation, transform), ...]
            =>
            ([annotation, annotation, ...], [transform, transform, ...])
        """
        return tuple(zip(*annotation_and_transform_pairs))

    def model_minibatch_output_to_predictions(
            self,
            minibatch_boxes: List[Union[np.ndarray, torch.Tensor]],
            minibatch_labels: List[Union[np.ndarray, torch.Tensor]],
            minibatch_scores: List[Union[np.ndarray, torch.Tensor]],
            minibatch_input_images: List[Composite],
            ) -> BoxArray:
        """
        convert detection model output to BoxArray

        Args:
            minibatch_boxes (List[Union[np.ndarray, torch.Tensor]]):
                bounding box coordinates (xmin, ymin, xmax, ymax)
                shape = (N, 4), dtype = float32/float64/int64
            minibatch_labels (List[Union[np.ndarray, torch.Tensor]]):
                category index (as in `self.scope`) of each bounding box
                shape = (N, ), dtype = int64
            minibatch_scores (List[Union[np.ndarray, torch.Tensor]]):
                confidence of each bounding box
                shape = (N, ), dtype = float32/float64
            minibatch_input_images (List[Composite]):
                the input images to the model (to get the input image width/height)
        Returns:
            BoxArray
        """
        predictions = []
        for boxes, scores, labels, input_image in\
                zip(minibatch_boxes, minibatch_scores, minibatch_labels, minibatch_input_images):
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            assert boxes.ndim == 2 and boxes.shape[1] == 4
            num_boxes = boxes.shape[0]

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            assert labels.shape == (num_boxes, ) and np.issubdtype(labels.dtype, np.integer)

            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            assert scores.shape == (num_boxes, ) and np.issubdtype(scores.dtype, np.floating)

            box_array = BoxArray.from_numpy_as_xyxy(
                boxes, input_image.img_w, input_image.img_h,
                label=[self.accessor.prediction_label_type(self.scope[v], s) for v, s in zip(labels, scores)]
            )
            predictions.append(box_array)

        return predictions
