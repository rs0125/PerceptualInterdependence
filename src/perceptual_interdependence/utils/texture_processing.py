"""
Texture processing utilities for loading, validation, and format conversion.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any


class TextureProcessor:
    """
    Utility class for texture processing operations.
    
    Provides methods for loading, validating, converting, and saving texture files
    with proper format handling for albedo and normal maps.
    """
    
    RGB_MAX = 255.0
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    @classmethod
    def load_albedo(cls, path: Union[str, Path]) -> np.ndarray:
        """
        Load albedo texture and convert to normalized float format.
        
        Args:
            path: Path to albedo texture file
            
        Returns:
            Albedo array in [0.0, 1.0] range with shape (H, W, 3)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(path)
        cls._validate_file(path)
        
        image = Image.open(path).convert('RGB')
        array = np.array(image, dtype=np.float32)
        
        return array / cls.RGB_MAX
    
    @classmethod
    def load_normal_map(cls, path: Union[str, Path]) -> np.ndarray:
        """
        Load normal map and convert to RGB format for processing.
        
        Args:
            path: Path to normal map file
            
        Returns:
            Normal map array in [0.0, 1.0] RGB format with shape (H, W, 3)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(path)
        cls._validate_file(path)
        
        image = Image.open(path).convert('RGB')
        array = np.array(image, dtype=np.float32)
        
        return array / cls.RGB_MAX
    
    @classmethod
    def save_albedo(
        cls, 
        albedo: np.ndarray, 
        path: Union[str, Path], 
        quality: int = 95
    ) -> None:
        """
        Save albedo texture to file.
        
        Args:
            albedo: Albedo array in [0.0, 1.0] range
            path: Output file path
            quality: JPEG quality (if saving as JPEG)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        albedo_uint8 = np.clip(albedo * cls.RGB_MAX, 0, 255).astype(np.uint8)
        
        # Save image
        image = Image.fromarray(albedo_uint8, mode='RGB')
        
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            image.save(path, format='JPEG', quality=quality, optimize=True)
        else:
            image.save(path, format='PNG', optimize=False)
    
    @classmethod
    def save_normal_map(
        cls, 
        normal: np.ndarray, 
        path: Union[str, Path]
    ) -> None:
        """
        Save normal map to file.
        
        Args:
            normal: Normal map array in [0.0, 1.0] RGB format
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        normal_uint8 = np.clip(normal * cls.RGB_MAX, 0, 255).astype(np.uint8)
        
        # Save as PNG (lossless for normal maps)
        image = Image.fromarray(normal_uint8, mode='RGB')
        image.save(path, format='PNG', optimize=False)
    
    @classmethod
    def get_texture_info(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a texture file.
        
        Args:
            path: Path to texture file
            
        Returns:
            Dictionary with texture information
        """
        path = Path(path)
        cls._validate_file(path)
        
        with Image.open(path) as image:
            return {
                'path': path,
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'file_size': path.stat().st_size,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
    
    @classmethod
    def validate_texture_pair(
        cls, 
        albedo_path: Union[str, Path], 
        normal_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Validate that albedo and normal textures are compatible.
        
        Args:
            albedo_path: Path to albedo texture
            normal_path: Path to normal map
            
        Returns:
            Validation results dictionary
            
        Raises:
            ValueError: If textures are incompatible
        """
        albedo_info = cls.get_texture_info(albedo_path)
        normal_info = cls.get_texture_info(normal_path)
        
        issues = []
        
        # Check size compatibility
        if albedo_info['size'] != normal_info['size']:
            issues.append(f"Size mismatch: albedo {albedo_info['size']} vs normal {normal_info['size']}")
        
        # Check if sizes are reasonable
        width, height = albedo_info['size']
        if width < 64 or height < 64:
            issues.append(f"Texture too small: {width}x{height} (minimum 64x64)")
        
        if width > 8192 or height > 8192:
            issues.append(f"Texture very large: {width}x{height} (may cause memory issues)")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'albedo_info': albedo_info,
            'normal_info': normal_info
        }
    
    @classmethod
    def _validate_file(cls, path: Path) -> None:
        """Validate that file exists and has supported format."""
        if not path.exists():
            raise FileNotFoundError(f"Texture file not found: {path}")
        
        if path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: {cls.SUPPORTED_FORMATS}")
    
    @staticmethod
    def convert_normal_vector_to_rgb(normal_vectors: np.ndarray) -> np.ndarray:
        """
        Convert normal vectors from [-1,1] range to RGB [0,1] format.
        
        Args:
            normal_vectors: Normal vectors in [-1,1] range
            
        Returns:
            Normal map in RGB [0,1] format
        """
        return (normal_vectors + 1.0) / 2.0
    
    @staticmethod
    def convert_rgb_to_normal_vector(normal_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB normal map to normal vectors in [-1,1] range.
        
        Args:
            normal_rgb: Normal map in RGB [0,1] format
            
        Returns:
            Normal vectors in [-1,1] range
        """
        return normal_rgb * 2.0 - 1.0