# setup.py
from setuptools import setup, find_packages

setup(
    name="arxiv_ml",
    version="0.1",
    # On ne prend que classifier et ses sous-packages
    packages=find_packages(include=["classifier", "classifier.*"]),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "umap-learn",
        "scikit-learn",
    ],
)