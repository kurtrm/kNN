"""
Houses the distance functions for use by knn.
"""
import numpy as np


def cosine_distance(x: np.ndarray, other: np.ndarray) -> float:
    dot_product = other.dot(x.reshape(-1, 1))
    mag_x = np.sqrt(np.sum(x**2))
    mag_other = np.sqrt(np.sum(other**2))

    return 1 - (dot_product / (mag_x * mag_other))


def euclidean_distance(x: np.ndarray, other: np.ndarray) -> float:
    return np.sqrt(np.sum((other - x)**2, axis=1))
