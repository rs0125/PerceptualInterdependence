"""
Core modules for perceptual interdependence binding system.

This package contains the fundamental components:
- AssetBinderComplex: Geometry-aware texture binding
- RenderSimulator: PBR-lite rendering and quality assessment
- RGBForensics: Forensic analysis and traitor tracing
"""

from .asset_binder_complex import AssetBinderComplex
from .render_simulator import RenderSimulator
from .rgb_forensics import RGBForensics

__all__ = ['AssetBinderComplex', 'RenderSimulator', 'RGBForensics']