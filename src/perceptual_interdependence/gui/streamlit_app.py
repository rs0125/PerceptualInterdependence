"""
Streamlit web application for perceptual interdependence binding system.
"""

from pathlib import Path


def get_app_path() -> Path:
    """Get the path to the Streamlit app file."""
    # Return path to the legacy streamlit_gui.py for now
    # TODO: Refactor streamlit_gui.py into this module
    return Path(__file__).parent.parent.parent.parent / "streamlit_gui.py"