"""
Mathematical algorithms for perceptual interdependence binding.

This package contains optimized implementations of core mathematical operations:
- CPU-optimized math with Numba JIT compilation
- Vectorized NumPy operations for maximum performance
- Poison generation, application, and antidote calculation algorithms
"""

from .cpu_math import CPUOptimizedMath, get_cpu_math

__all__ = ["CPUOptimizedMath", "get_cpu_math"]