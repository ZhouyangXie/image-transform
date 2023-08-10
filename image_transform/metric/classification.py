import numpy as np
from typing import Iterable, Hashable, Dict, Union

from .base import Metric, compute_acc
from ..annotation import Label, Scoped, ScopedWithConfidence, MultipleScoped,\
    ProbabilisticMultipleScoped, ImageAnnotation


class MulticlassAccuracy(Metric):
    # metric_names = {
    #     "recalls": np.ndarray,
    #     "mean recall": float,
    #     "precisions": np.ndarray,
    #     "mean precision": float,
    #     "f1scores": np.ndarray,
    #     "mean f1score": float,
    #     "confusion": np.ndarray,
    #     "accuracy": float,
    #     "count": np.ndarray,
    # }

    def __init__(self, scope: Iterable[Hashable]) -> None:
        """Accuracy metrics for multiclass classification
            Recall(class-specific), mean recall,
            Precision(class-specific), mean precision,
            F1-Score(class-specific), mean F1-Score,
            Confusion matrix

        Args:
            scope (Iterable[Hashable]):
                An interable of unique hashable values,
                str, int, float etc.
        """
        scope = list(scope)
        assert len(scope) == len(set(scope))
        assert len(scope) > 1
        self.scope = scope
        self.confusion = np.zeros(
            (len(self.scope), len(self.scope)),
            dtype=np.int64,
        )
        super().__init__()

    def reset(self) -> None:
        self.confusion[...] = 0

    @staticmethod
    def get_value(label, scope) -> int:
        if isinstance(label, (Scoped, ScopedWithConfidence)):
            return scope.index(label.value)
        elif isinstance(label, ProbabilisticMultipleScoped):
            return scope.index(label.to_scoped().value)
        elif isinstance(label, ImageAnnotation):
            return MulticlassAccuracy.get_value(label.label)
        else:
            return scope.index(label)

    @staticmethod
    def get_values(labels, scope) -> np.ndarray:
        if isinstance(labels, np.ndarray):
            assert np.issubdtype(labels.dtype, np.integer) and labels.ndim == 1
            labels = labels.astype(np.int64)
        else:
            labels = np.array([MulticlassAccuracy.get_value(label, scope) for label in labels], np.int64)

        return labels

    def update(
            self,
            predictions: Union[Iterable[Union[Label, ImageAnnotation, Hashable]], np.ndarray],
            targets: Union[Iterable[Union[Label, ImageAnnotation, Hashable]], np.ndarray]
            ) -> None:
        predictions = MulticlassAccuracy.get_values(predictions, self.scope)
        targets = MulticlassAccuracy.get_values(targets, self.scope)
        assert predictions.shape == targets.shape
        for p, t in zip(predictions, targets):
            self.confusion[p, t] += 1

    def get_score(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Get the metrics. For those classes with zero instance evaluated,
        the recall is 1.0; with zero positive predcition, the precision is 1.0.

        Returns:
            Dict:{
            "recalls": np.array, shape=(len(self.scope), ), # recall of each class
            "mean_recall": float, # mean of recalls
            "precisions": np.array, shape=(len(self.scope), ), # precision of each class
            "mean_precision": float, # mean of precisions
            "f1scores": np.array, shape=(len(self.scope), ), # F1-score of each class
            "mean_f1score": float, # mean F1-score
            "accuracy": float, # class-agnostic accuracy
            "confusions": np.ndarray, shape=(len(self.scope), len(self.scope)),
                # confusions[i, j] (number of) instances are predicted as i while the ground truth is j.
            "counts": np.array, shape=(len(self.scope), ),
                # the number of ground truth instances of each class
        }
        """
        # compute true_positive, positive, true
        true_positive = self.confusion.diagonal()
        num_instances = self.confusion.sum(axis=0)
        num_positive = self.confusion.sum(axis=1)

        # compute metrics regardless of zero instance classes
        recalls = compute_acc(true_positive, num_instances)
        precisions = compute_acc(true_positive, num_positive)
        r_add_p = recalls + precisions
        r_add_p[r_add_p == 0] = 1.0
        f1scores = 2 * recalls * precisions/r_add_p

        num_tp = true_positive.sum()
        num_total = num_instances.sum()
        accuracy = num_tp/num_total if num_total > 0 else 1.0

        return {
            "recalls": recalls,
            "mean recall": recalls.mean(),
            "precisions": precisions,
            "mean precision": precisions.mean(),
            "f1scores": f1scores,
            "mean f1score": f1scores.mean(),
            "accuracy": accuracy,
            "confusion": self.confusion,
            "count": num_instances
        }


class MultilabelAccuracy(Metric):
    # metric_names = {
    #     "recall": np.ndarray,
    #     "mean_recall": float,
    #     "precision": np.ndarray,
    #     "mean_precision": float,
    #     "f1score": np.ndarray,
    #     "mean_f1score": float,
    #     "confusion": np.ndarray,
    #     "count": int
    # }

    def __init__(self, scope: Iterable[Hashable]) -> None:
        scope = list(scope)
        assert len(scope) == len(set(scope))
        assert len(scope) > 1
        self.scope = scope
        self.confusions = np.zeros((len(self.scope), 2, 2), dtype=np.int64)
        super().__init__()

    def clear(self):
        self.confusions[...] = 0

    @staticmethod
    def get_value(label, scope) -> np.ndarray:
        if isinstance(label, ImageAnnotation):
            return MultilabelAccuracy.get_value(label.label, scope)
        else:
            multihot = np.zeros(len(scope), dtype=np.int64)
            if isinstance(label, Label):
                if isinstance(label, (Scoped, ScopedWithConfidence)):
                    multihot[scope.index(label.value)] = 1
                elif isinstance(label, MultipleScoped):
                    multihot[[scope.index(v) for v in label.values]] = 1
                else:
                    raise TypeError()
            else:
                multihot[scope.index(label)] = 1
            return multihot

    @staticmethod
    def get_values(labels, scope) -> np.ndarray:
        if isinstance(labels, np.ndarray):
            assert (np.issubdtype(labels.dtype, np.integer) or labels.dtype == bool) and labels.ndim == 2
            assert labels.shape[-1] == len(scope)
            return labels.astype(np.int64)
        else:
            labels = np.stack([MulticlassAccuracy.get_value(label, scope) for label in labels])

    def update(
            self,
            predictions: Union[Iterable[Union[Label, ImageAnnotation, Hashable]], np.ndarray],
            targets: Union[Iterable[Union[Label, ImageAnnotation, Hashable]], np.ndarray]
            ) -> None:
        predictions = MultilabelAccuracy.get_values(predictions, self.scope)
        targets = MultilabelAccuracy.get_values(targets, self.scope)
        assert predictions.shape == targets.shape
        for p, t in zip(predictions, targets):
            self.confusions[np.arange(len(self.scope)), p, t] += 1

    def get_score(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Get the metrics. For those classes with zero instance evaluated,
        the class-specific scores will be np.inf or np.nan

        Returns:
            Dict:{
            "recalls": np.array, shape=(len(self.scope), ), # recall of each class
            "mean_recall": float, # mean of recalls
            "precisions": np.array, shape=(len(self.scope), ), # precision of each class
            "mean_precision": float, # mean of precisions
            "f1scores": np.array, shape=(len(self.scope), ), # F1-score of each class
            "mean_f1score": float, # mean F1-score
            "accuracies": np.array, shape=(len(self.scope), ), # accurracy of each class
            "mean_accuracy": float, # mean of accuracies
            "confusions": np.ndarray, shape=(len(self.scope), 2, 2),
                # confusions[c, 0, 1] (number of) class c instances predicted as negative and the gt is positive.
            "counts": np.array, shape=(len(self.scope), ),
                # the number of ground truth instances of each class
        }
        """
        true_positive = self.confusions[:, 1, 1]
        true_negative = self.confusions[:, 0, 0]
        false_positive = self.confusions[:, 1, 0]
        false_negative = self.confusions[:, 0, 1]

        recalls = compute_acc(true_positive, true_positive + false_negative)
        precisions = compute_acc(true_positive, true_positive + false_positive)
        accuracies = compute_acc(true_positive + true_negative, self.confusions.sum(axis=(1, 2)))
        r_add_p = recalls + precisions
        r_add_p[r_add_p == 0] = 1.0
        f1scores = 2 * recalls * precisions/r_add_p

        return {
            "recalls": recalls,
            "mean recall": recalls.mean(),
            "precisions": precisions,
            "mean precision": precisions.mean(),
            "f1scores": f1scores,
            "mean f1score": f1scores.mean(),
            "accuracies": accuracies,
            "mean accuracy": accuracies.mean(),
            "confusions": self.confusions,
            "counts": self.confusions[:, :, 1].sum(axis=1)
        }
