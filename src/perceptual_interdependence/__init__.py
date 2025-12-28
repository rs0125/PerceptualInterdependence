"""
Perceptual Interdependence: Geometry-Aware Photometric Binding System

A research-grade implementation of geometry-aware asset binding for 3D texture protection.
This system provides mathematically guaranteed restoration for legitimate users while
maintaining security against unauthorized access.

Key Features:
- CPU-optimized mathematical operations with Numba JIT compilation
- Analytically safe one-way binding with strict algebraic cancellation
- Geometry-aware normal map processing with constraint validation
- Comprehensive forensic analysis and verification tools
- Interactive GUI and command-line interfaces

Example Usage:
    >>> from perceptual_interdependence import AssetBinder
    >>> binder = AssetBinder()
    >>> binder.bind_textures("albedo.png", "normal.png", user_id=42)
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Core exports
from .core.asset_binder import AssetBinder
from .algorithms.cpu_math import CPUOptimizedMath
from .core.render_simulator import RenderSimulator
from .core.forensics import RGBForensics

# Convenience imports
from .utils.texture_processing import TextureProcessor
from .utils.validation import ValidationSuite

__all__ = [
    "AssetBinder",
    "CPUOptimizedMath", 
    "RenderSimulator",
    "RGBForensics",
    "TextureProcessor",
    "ValidationSuite",
]