from setuptools import setup, find_packages

setup(
    name="ts-metric",
    version="0.2.0",
    description="Time series metric computation library for prediction, imputation, generation, anomaly detection, and classification",
    author="songzy",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
