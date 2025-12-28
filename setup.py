#!/usr/bin/env python3
"""
Setup configuration for Perceptual Interdependence research project.
"""

from setuptools import setup, find_packages

setup(
    name="perceptual-interdependence",
    version="1.0.0",
    description="Geometry-Aware Asset Binding Protocol for Collusion-Resistant 3D Texture Protection",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security :: Cryptography",
    ],
)