"""
Test the kNN class and distance functions.
"""
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
