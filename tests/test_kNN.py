"""
Test the kNN class and distance functions.
"""
import numpy as np
import pytest

from scipy.spatial.distance import cosine, euclidean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from src.distance_funcs import cosine_distance, euclidean_distance
from src.kNN import KNearestNeighbors


def test_instantiation():
    """
    Basic instantiation test.
    """
    assert isinstance(KNearestNeighbors(3, cosine_distance), KNearestNeighbors)


def test_cosine_dist_func():
    """
    Simple test for cosine distance function.
    """
    scratch_cosine = cosine_distance(np.array([4, 5, 6]),
                                     np.array([7, 8, 9]))[0]
    scipy_cosine = cosine(np.array([4, 5, 6]),
                          np.array([7, 8, 9]))

    assert scratch_cosine == pytest.approx(scipy_cosine, .01)


def test_euclidean_dist_func():
    """
    Simple test for euclidean distance function.
    """
    scratch_euclid = euclidean_distance(np.array([4, 5, 6]),
                                        np.array([7, 8, 9]))
    scipy_euclid = euclidean(np.array([4, 5, 6]),
                             np.array([7, 8, 9]))
    assert scratch_euclid == pytest.approx(scipy_euclid, .01)
