"""
Core modules for perceptual interdependence binding system.

This package contains the fundamental components:
- AssetBinder: Main binding orchestration
- RenderSimulator: Photometric rendering simulation  
- RGBForensics: Forensic analysis tools
"""

from .asset_binder import AssetBinder
from .render_simulator import RenderSimulator
from .forensics import RGBForensics

__all__ = ["AssetBinder", "RenderSimulator", "RGBForensics"]