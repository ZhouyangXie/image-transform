from abc import abstractmethod, ABC
from typing import Dict, Union

import numpy as np


class Metric(ABC):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """ reset counter and other accumulated values """
        pass

    @abstractmethod
    def update(self, predictions, targets) -> None:
        pass

    @abstractmethod
    def get_score(self) -> Dict:
        """
        Return metric name and values

        Returns:
            Dict[str, Any]
        """
        pass


def compute_acc(correct: Union[np.ndarray, int, float], total: Union[np.ndarray, int, float]):
    """ for each position, if total[i] == 0, result[i] == 1; else result[i] = correct[i]/total[i]"""
    if np.isscalar(total):
        if total == 0:
            if isinstance(correct, np.ndarray):
                return np.ones_like(correct, dtype=float)
            else:
                return 1.
        else:
            return correct/total
    else:
        result = correct/np.maximum(total, 1)
        result[total == 0] = 1
        return result
