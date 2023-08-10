import numpy as np
from typing import Iterable, Hashable, Dict, Union
from itertools import product

from .base import Metric, compute_acc
from ..annotation import ScopedWithConfidence, Scoped, BoxArray, OrientedBoxArray
from ..annotation.label import group_objects_by_class


class ClassAgnosticAccuracyWithConfidence(Metric):
    # metric_names = {
    #         "recall": float,
    #         "average precision": float,
    #         "count": int
    # }

    def __init__(self):
        # confidence of the shots, length is the number of shots
        self.confidences = []
        # how many valid hits(a shot can hit multiple targets) does the shot make
        # length is the number of shots
        self.valid_hits = []
        # total number of target(not image instance)
        self.num_targets = 0
        # total number of hit target
        self.num_recalled_targets = 0
        super().__init__()

    def clear(self) -> None:
        self.confidences.clear()
        self.valid_hits.clear()
        self.num_targets = 0
        self.num_recalled_targets = 0

    def update(
            self, hit_matrix: np.ndarray, confidences: Union[np.ndarray, Iterable[float]],
            max_num_predictions: int = None
            ):
        ordered_indices = np.argsort(confidences)[::-1]
        if max_num_predictions is not None and len(ordered_indices) > max_num_predictions:
            ordered_indices = ordered_indices[:max_num_predictions]

        hit_matrix = hit_matrix.astype(bool)
        hit_matrix = hit_matrix[ordered_indices, :]
        confidences = np.array(confidences)
        confidences = confidences[ordered_indices]
        num_predctions, num_targets = hit_matrix.shape
        assert confidences.shape == (num_predctions, )

        # is the target hit by at least one shot
        has_hit = np.zeros(num_targets, bool)
        # how many new targets are hit by a shot(excluding duplicate hits)
        valid_hits = np.zeros(num_predctions, int)
        for i in range(num_predctions):
            valid_hits[i] = ((~has_hit) & hit_matrix[i]).sum()
            # a major flaw of this metric: if one shot can hit multiple targets here
            # other shots of lower confidence on same targets count as false positive
            has_hit |= hit_matrix[i]

        self.num_targets += num_targets
        self.num_recalled_targets += has_hit.sum()
        self.confidences.append(confidences)
        self.valid_hits.append(valid_hits)

    def get_score(self) -> Dict:
        confidences = np.concatenate(self.confidences)
        valid_hits = np.concatenate(self.valid_hits)
        assert confidences.shape == valid_hits.shape

        recall = compute_acc(self.num_recalled_targets, self.num_targets)

        if len(valid_hits) == 0 or self.num_targets == 0:
            if len(valid_hits) == 0 and self.num_targets == 0:
                average_precision = 1.0
            else:
                average_precision = 0.0
        else:
            # sort all hits(across instances) by confidence
            hits = valid_hits[np.argsort(confidences)[::-1]]
            has_hit = hits > 0

            # compute cumulative precision and recall
            precision = np.cumsum(has_hit)/(np.arange(len(has_hit)) + 1)
            recall_diff = hits/self.num_targets

            # choose the points where the recall changes(COCO-style)
            precision = precision[has_hit]
            recall_diff = recall_diff[has_hit]
            for i in range(len(precision) - 2, -1, -1):
                precision[i] = max(precision[i], precision[i+1])

            average_precision = np.sum(precision * recall_diff).item()

        return {
            "recall": recall,
            "average precision": average_precision,
            "count": self.num_targets,
        }


class AccuracyWithConfidence(Metric):
    # metric_names = {
    #     "recall": np.ndarray,
    #     # shape = (len(scope), len(iou_thresholds), len(max_num_predictions)).squeeze()
    #     "mean_recall": Union[np.ndarray, float],
    #     # shape = (len(iou_thresholds), len(max_num_predictions)).squeeze()
    #     "average_precision": np.ndarray,
    #     # shape = (len(scope), len(iou_thresholds), len(max_num_predictions)).squeeze()
    #     "mean_average_precision": Union[np.ndarray, float],
    #     # shape = (len(iou_thresholds), len(max_num_predictions)).squeeze()
    #     "count": np.ndarray,
    #     # shape = (len(scope), )
    # }

    def __init__(
            self, scope: Iterable[Hashable],
            iou_thresholds: Union[float, Iterable[float]] = 0.5,
            max_num_predictions: Union[int, Iterable[int], None] = None,
            ) -> None:
        scope = list(scope)
        assert len(scope) == len(set(scope))
        assert len(scope) > 0
        self.scope = scope

        if isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [iou_thresholds]
        assert all(0 < t <= 1.0 for t in iou_thresholds)
        self.iou_thresholds = list(iou_thresholds)

        if isinstance(max_num_predictions, int) or max_num_predictions is None:
            max_num_predictions = [max_num_predictions]
        assert all((n is None or (n > 0)) for n in max_num_predictions)
        self.max_num_predictions = max_num_predictions

        self.accs = [[[
                ClassAgnosticAccuracyWithConfidence()
                for _ in max_num_predictions]
                for _ in iou_thresholds]
                for _ in scope]

        super().__init__()

    def reset(self):
        for c in range(len(self.scope)):
            for i in range(len(self.iou_thresholds)):
                for j in range(len(self.max_num_predictions)):
                    self.accs[c][i][j].reset()

    def update_instance(self, prediction: Union[OrientedBoxArray, BoxArray], target: Union[OrientedBoxArray, BoxArray]):
        if len(prediction) > 0:
            assert isinstance(prediction.label[0], ScopedWithConfidence)
            assert set(prediction.label[0].scope) == set(self.scope)
        if len(target) > 0:
            assert isinstance(target.label[0], (Scoped, ScopedWithConfidence))
            assert set(target.label[0].scope) == set(self.scope)
        if isinstance(prediction, BoxArray) and isinstance(target, OrientedBoxArray):
            prediction = prediction.to_oriented_box_array()
        if isinstance(target, BoxArray) and isinstance(prediction, OrientedBoxArray):
            target = target.to_oriented_box_array()

        prediction = group_objects_by_class(prediction, self.scope)
        target = group_objects_by_class(target, self.scope)

        for c, label in enumerate(self.scope):
            class_prediction = prediction[label]
            class_target = target[label]
            confidences = [label.confidence for label in class_prediction.label]
            iou_matrix = class_prediction.iou(class_target)
            for (i, t), (j, m) in product(enumerate(self.iou_thresholds), enumerate(self.max_num_predictions)):
                self.accs[c][i][j].update(iou_matrix >= t, confidences, m)

    def update(
            self,
            predictions: Iterable[Union[OrientedBoxArray, BoxArray]],
            targets: Iterable[Union[OrientedBoxArray, BoxArray]],
            ) -> None:
        predictions, targets = list(predictions), list(targets)
        assert len(predictions) == len(targets)
        for prediction, target in zip(predictions, targets):
            self.update_instance(prediction, target)

    def get_score(self) -> Dict:
        recalls = np.zeros((len(self.scope), len(self.iou_thresholds), len(self.max_num_predictions)), float)
        aps = np.zeros((len(self.scope), len(self.iou_thresholds), len(self.max_num_predictions)), float)
        counts = np.zeros(len(self.scope), int)

        for c, i, j in product(
            range(len(self.scope)), range(len(self.iou_thresholds)), range(len(self.max_num_predictions))
                ):
            score = self.accs[c][i][j].get_score()
            recalls[c][i][j] = score["recall"]
            aps[c][i][j] = score["average precision"]
            counts[c] = score["count"]

        mean_recall = recalls.mean(axis=0).squeeze()
        mean_recall = mean_recall.item() if mean_recall.size == 1 else mean_recall
        mean_ap = aps.mean(axis=0).squeeze()
        mean_ap = mean_ap.item() if mean_ap.size == 1 else mean_ap

        return {
            "recalls": recalls,
            "mean recall": mean_recall,
            "average precisions": aps,
            "mean average precision": mean_ap,
            "counts": counts,
        }
