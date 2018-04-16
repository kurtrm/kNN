"""Setup for decision tree module."""
from setuptools import setup


extra_packages = {
    'testing': ['pytest']
}


setup(
    name='k-Nearest Neighbors Implementation',
    description='Provides a basic implementation of th kNN algorithm.',
    version=0.0,
    author='Kurt Maurer',
    author_email='kurtrm@gmail.com',
    extras_require=extra_packages
)
