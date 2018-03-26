"""
Module containing the kNN class.
"""
from collections import Counter
from typing import Callable

import numpy as np


class KNearestNeighbors:
    """
    Implementation of K Nearest Neighbor classification
    algorithm.
    """
    def __init__(self, k: int, distance_func: Callable) -> None:
        """
        Initialize class with the:
        k -> minimum number of neighbors to classify data.
        distance -> distance function to use for determination.
        """
        self.k = k
        self.distance_func = distance_func

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the data."""
        self.X = X
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> str:
        """
        Predict the class of X based on k and distance measurement.
        """
        distances = np.zeros((len(X), len(self.X)))
        for idx, row in enumerate(X):
            distances[idx, :] = self.distance_func(row, self.X).flatten()
        shortest_dist_idx = np.argsort(distances, axis=1)[:, :self.k]
        predictions = np.empty(len(shortest_dist_idx))
        for idx, dist in enumerate(shortest_dist_idx):
            predictions[idx] = Counter(self.y[dist]).most_common(1)[0][0]
        return predictions

    def score(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate the accuracy given the actual and predicted labels.
        """
        both_equal = sum(actual == predicted)
        return both_equal / len(predicted)
