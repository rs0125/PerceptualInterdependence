"""
Geometry-Aware Photometric Binding System - Analytically Safe One-Way Binding

This module implements the AssetBinder class using strict algebraic cancellation
without calibration loops. The system guarantees mathematical quality restoration for
legitimate users while maintaining security against attackers.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

from ..algorithms.cpu_math import get_cpu_math


class AssetBinder:
    """
    Implements Analytically Safe One-Way Binding with strict algebraic cancellation.
    
    Core Concept:
    - Poison: Always brighten albedo (A_new = A * (1 + P))
    - Antidote: Always steepen normals (Z_new = Z_old / (1 + P))
    - Mathematical guarantee: A*(1+P) * Z/(1+P) = A*Z (perfect cancellation)
    
    No calibration loops needed - the algebra IS the calibration.
    
    Attributes:
        EPSILON: Division by zero protection constant
        BLOCK_SIZE: 4x4 pixel blocks for BC7 optimization
        RGB_MAX: Maximum RGB value for conversion
    """
    
    # Mathematical constants
    EPSILON = 1e-6      # Division by zero protection
    BLOCK_SIZE = 4      # 4x4 pixel blocks for BC7 optimization
    RGB_MAX = 255.0     # Maximum RGB value for conversion
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the AssetBinder class.
        
        Args:
            output_dir: Directory for output files (default: current directory)
        """
        self.cpu_math = get_cpu_math()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print acceleration status
        if self.cpu_math.use_numba:
            print("🚀 AssetBinder: CPU acceleration (Numba JIT) enabled")
        else:
            print("📊 AssetBinder: Vectorized NumPy mode")
    
    def bind_textures(
        self, 
        albedo_path: Union[str, Path], 
        normal_path: Union[str, Path], 
        user_id: int, 
        poison_strength: float = 0.2,
        output_prefix: str = "bound"
    ) -> Dict[str, Any]:
        """
        Main binding method using Analytically Safe One-Way Binding.
        
        This method implements strict algebraic cancellation without calibration loops:
        - Poison: Always brighten albedo (A_new = A * (1 + P))
        - Antidote: Always steepen normals (Z_new = Z_old / (1 + P))
        - Mathematical guarantee: A*(1+P) * Z/(1+P) = A*Z (perfect cancellation)
        
        Args:
            albedo_path: Path to the input albedo texture file
            normal_path: Path to the input normal map file
            user_id: Unique identifier for reproducible random number generation
            poison_strength: Strength of poison application [0.0, 1.0]
            output_prefix: Prefix for output filenames
            
        Returns:
            Dictionary containing binding results and statistics
            
        Raises:
            ValueError: If poison_strength is outside valid range [0.0, 1.0]
            FileNotFoundError: If input texture files do not exist
        """
        print(f"Starting Analytically Safe One-Way Binding (user_id: {user_id})...")
        
        # Validate inputs
        self._validate_poison_strength(poison_strength)
        
        # Step 1: Load and normalize inputs
        albedo, normal = self._load_and_validate_inputs(albedo_path, normal_path)
        print(f"  Loaded textures: {albedo.shape}")
        
        # Step 2: Generate target poison map (strictly positive values)
        texture_shape = (albedo.shape[0], albedo.shape[1])
        poison_target = self._generate_poison_map(texture_shape, user_id, poison_strength)
        print(f"  Generated poison map: range [0.0, {poison_strength}]")
        
        # Step 3: Apply poison to albedo (brightening attempt)
        albedo_poisoned, poison_effective = self._apply_poison_to_albedo_safe(albedo, poison_target)
        print(f"  Applied poison to albedo (with saturation handling)")
        
        # Step 4: Calculate analytical antidote for normal map
        normal_antidote = self._calculate_analytical_antidote(normal, poison_effective, user_id)
        print(f"  Calculated analytical antidote normal map")
        
        # Step 5: Save outputs
        output_paths = self._save_outputs(albedo_poisoned, normal_antidote, user_id, output_prefix)
        print(f"  Saved bound assets: {output_paths['albedo'].name}, {output_paths['normal'].name}")
        print(f"Analytically Safe Binding completed for user {user_id}")
        
        # Return comprehensive results
        return {
            'user_id': user_id,
            'poison_strength': poison_strength,
            'texture_shape': texture_shape,
            'output_paths': output_paths,
            'statistics': {
                'poison_target_mean': float(np.mean(poison_target)),
                'poison_effective_mean': float(np.mean(poison_effective)),
                'saturation_ratio': float(np.sum(albedo_poisoned >= 0.999) / albedo_poisoned.size)
            }
        }
    
    def _load_and_validate_inputs(
        self, 
        albedo_path: Union[str, Path], 
        normal_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate input textures, converting to appropriate formats.
        
        Args:
            albedo_path: Path to albedo texture file
            normal_path: Path to normal map file
            
        Returns:
            Tuple of processed albedo and normal arrays
            
        Raises:
            FileNotFoundError: If texture files do not exist
            ValueError: If image processing fails
        """
        albedo_path = Path(albedo_path)
        normal_path = Path(normal_path)
        
        # Validate file existence
        if not albedo_path.exists():
            raise FileNotFoundError(f"Albedo texture file not found: {albedo_path}")
        if not normal_path.exists():
            raise FileNotFoundError(f"Normal map file not found: {normal_path}")
        
        try:
            # Load albedo texture
            albedo_image = Image.open(albedo_path).convert('RGB')
            albedo_array = np.array(albedo_image, dtype=np.float32)
            
            # Convert albedo from [0, 255] to float [0.0, 1.0] range
            albedo_normalized = albedo_array / self.RGB_MAX
            
            # Load normal map
            normal_image = Image.open(normal_path).convert('RGB')
            normal_array = np.array(normal_image, dtype=np.float32)
            
            # Convert normal from RGB [0, 255] to vector [-1.0, 1.0] format
            normal_vectors = (normal_array / self.RGB_MAX) * 2.0 - 1.0
            
            # Normalize normal vectors to ensure unit length
            normal_normalized = self._normalize_normal_vectors(normal_vectors)
            
            # Convert back to RGB format for processing
            normal_rgb = (normal_normalized + 1.0) / 2.0
            
            return albedo_normalized, normal_rgb
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Failed to load or process image files: {str(e)}")
    
    def _normalize_normal_vectors(self, normal_array: np.ndarray) -> np.ndarray:
        """
        Normalize normal vectors to ensure unit length.
        
        Args:
            normal_array: Input normal vectors in [-1.0, 1.0] range
            
        Returns:
            Normalized normal vectors
        """
        # Calculate vector magnitudes
        magnitudes = np.sqrt(np.sum(normal_array ** 2, axis=2, keepdims=True))
        
        # Handle zero-length vectors by replacing with epsilon
        magnitudes = np.where(magnitudes < self.EPSILON, self.EPSILON, magnitudes)
        
        # Normalize vectors
        normalized_normals = normal_array / magnitudes
        
        return normalized_normals
    
    def _validate_poison_strength(self, poison_strength: float) -> None:
        """
        Validate poison_strength parameter range.
        
        Args:
            poison_strength: Poison strength value to validate
            
        Raises:
            ValueError: If poison_strength is outside valid range [0.0, 1.0]
        """
        if not isinstance(poison_strength, (int, float)):
            raise ValueError(f"poison_strength must be a number, got {type(poison_strength)}")
        
        if not (0.0 <= poison_strength <= 1.0):
            raise ValueError(f"poison_strength must be in range [0.0, 1.0], got {poison_strength}")
    
    def _generate_poison_map(
        self, 
        shape: Tuple[int, int], 
        seed: int, 
        poison_strength: float
    ) -> np.ndarray:
        """
        Generate target poison map with strictly positive values for brightening.
        
        Args:
            shape: Shape of the output poison array (height, width)
            seed: Random seed for deterministic generation
            poison_strength: Maximum poison strength [0.0, 1.0]
            
        Returns:
            Poison map with values in range [0.0, poison_strength]
        """
        return self.cpu_math.generate_poison_map(shape, seed, poison_strength)
    
    def _apply_poison_to_albedo_safe(
        self, 
        albedo: np.ndarray, 
        poison_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply poison to albedo with saturation safety check.
        
        This method handles the "White Pixel Trap" where bright pixels can't be brightened
        further due to clipping. We calculate the effective poison after clipping.
        
        Args:
            albedo: Input albedo texture [0.0, 1.0]
            poison_target: Target poison values [0.0, poison_strength]
            
        Returns:
            Tuple of poisoned albedo and effective poison arrays
        """
        poisoned_albedo, effective_poison = self.cpu_math.apply_poison(albedo, poison_target)
        
        # Statistics
        saturation_pixels = np.sum(poisoned_albedo >= 0.999)
        total_pixels = albedo.size
        avg_poison_target = np.mean(poison_target)
        avg_poison_effective = np.mean(effective_poison)
        
        print(f"    Poison application statistics:")
        print(f"      Target poison: {avg_poison_target:.4f} (range: [0.0, {np.max(poison_target):.4f}])")
        print(f"      Effective poison: {avg_poison_effective:.4f} (range: [0.0, {np.max(effective_poison):.4f}])")
        print(f"      Saturated pixels: {saturation_pixels} / {total_pixels} ({100*saturation_pixels/total_pixels:.1f}%)")
        
        return poisoned_albedo, effective_poison
    
    def _calculate_analytical_antidote(
        self, 
        normal: np.ndarray, 
        poison_effective: np.ndarray, 
        user_seed: int
    ) -> np.ndarray:
        """
        Calculate analytical antidote normal map using strict algebraic cancellation.
        
        This method implements the core mathematical guarantee:
        Z_new = Z_old / (1 + P_effective)
        
        Args:
            normal: Original normal map in RGB format [0.0, 1.0]
            poison_effective: Effective poison values [0.0, max_poison]
            user_seed: User seed for deterministic processing
            
        Returns:
            Antidote normal map with algebraic cancellation
        """
        print(f"    Calculating analytical antidote...")
        
        # Use CPU-optimized antidote calculation
        antidote_vectors = self.cpu_math.calculate_antidote(normal, poison_effective)
        
        # Convert to vector format for statistics
        normal_vec = normal * 2.0 - 1.0
        antidote_vec = antidote_vectors * 2.0 - 1.0
        original_z = normal_vec[:, :, 2]
        antidote_z = antidote_vec[:, :, 2]
        
        # Statistics
        flat_threshold = 0.01
        lateral_mag = np.sqrt(normal_vec[:, :, 0]**2 + normal_vec[:, :, 1]**2)
        flat_pixels = np.sum(lateral_mag < flat_threshold)
        avg_z_reduction = np.mean(original_z - antidote_z)
        max_z_reduction = np.max(original_z - antidote_z)
        
        # Geometric validation
        steepening_check = np.all(antidote_z <= original_z + self.EPSILON)
        
        print(f"      Antidote statistics:")
        print(f"        Flat pixels handled: {flat_pixels} / {normal.size//3} ({100*flat_pixels/(normal.size//3):.1f}%)")
        print(f"        Average Z reduction (steepening): {avg_z_reduction:.4f}")
        print(f"        Maximum Z reduction: {max_z_reduction:.4f}")
        print(f"        Z range: [{np.min(antidote_z):.4f}, {np.max(antidote_z):.4f}]")
        print(f"        Geometric validation: {'PASSED' if steepening_check else 'FAILED'}")
        
        return antidote_vectors
    
    def _save_outputs(
        self, 
        bound_albedo: np.ndarray, 
        bound_normal: np.ndarray, 
        user_id: int,
        prefix: str = "bound"
    ) -> Dict[str, Path]:
        """
        Save processed textures with appropriate format conversion and naming.
        
        Args:
            bound_albedo: Processed albedo texture [0.0, 1.0]
            bound_normal: Processed normal map [0.0, 1.0]
            user_id: User identifier for filename generation
            prefix: Filename prefix
            
        Returns:
            Dictionary with output file paths
            
        Raises:
            IOError: If file saving fails
        """
        try:
            # Pack float albedo values [0.0, 1.0] back to RGB format [0, 255]
            albedo_rgb = np.clip(bound_albedo * self.RGB_MAX, 0, 255).astype(np.uint8)
            
            # Pack normal vectors [0.0, 1.0] to RGB format [0, 255]
            normal_rgb = np.clip(bound_normal * self.RGB_MAX, 0, 255).astype(np.uint8)
            
            # Generate output filenames
            albedo_path = self.output_dir / f"{prefix}_albedo_{user_id}.png"
            normal_path = self.output_dir / f"{prefix}_normal_{user_id}.png"
            
            # Convert numpy arrays to PIL Images and save
            Image.fromarray(albedo_rgb, mode='RGB').save(albedo_path, format='PNG', optimize=False)
            Image.fromarray(normal_rgb, mode='RGB').save(normal_path, format='PNG', optimize=False)
            
            return {
                'albedo': albedo_path,
                'normal': normal_path
            }
            
        except Exception as e:
            raise IOError(f"Failed to save output files: {str(e)}")