"""
Geometry-Aware Photometric Binding System - Analytically Safe One-Way Binding

This module implements the AssetBinderComplex class using strict algebraic cancellation
without calibration loops. The system guarantees mathematical quality restoration for
legitimate users while maintaining security against attackers.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


class AssetBinderComplex:
    """
    Implements Analytically Safe One-Way Binding with strict algebraic cancellation.
    
    Core Concept:
    - Poison: Always brighten albedo (A_new = A * (1 + P))
    - Antidote: Always steepen normals (Z_new = Z_old / (1 + P))
    - Mathematical guarantee: A*(1+P) * Z/(1+P) = A*Z (perfect cancellation)
    
    No calibration loops needed - the algebra IS the calibration.
    """
    
    # Mathematical constants
    EPSILON = 1e-6      # Division by zero protection
    BLOCK_SIZE = 4      # 4x4 pixel blocks for BC7 optimization
    RGB_MAX = 255.0     # Maximum RGB value for conversion
    
    def __init__(self):
        """
        Initialize the AssetBinderComplex class.
        
        The class is designed to be instantiated without parameters and configured
        through the bind_textures method call.
        """
        pass
    
    def bind_textures(self, clean_albedo_path: str, original_normal_path: str, 
                     user_seed: int, poison_strength: float = 0.2) -> None:
        """
        Main binding method using Analytically Safe One-Way Binding.
        
        This method implements strict algebraic cancellation without calibration loops:
        - Poison: Always brighten albedo (A_new = A * (1 + P))
        - Antidote: Always steepen normals (Z_new = Z_old / (1 + P))
        - Mathematical guarantee: A*(1+P) * Z/(1+P) = A*Z (perfect cancellation)
        
        Args:
            clean_albedo_path (str): Path to the input albedo texture file
            original_normal_path (str): Path to the input normal map file
            user_seed (int): Seed for reproducible random number generation
            poison_strength (float): Strength of poison application [0.0, 1.0], default 0.2
            
        Raises:
            ValueError: If poison_strength is outside valid range [0.0, 1.0]
            FileNotFoundError: If input texture files do not exist
        """
        print(f"Starting Analytically Safe One-Way Binding (user_seed: {user_seed})...")
        
        # Validate poison_strength parameter
        self._validate_poison_strength(poison_strength)
        
        # Step 1: Load and normalize inputs
        albedo, normal = self._load_and_validate_inputs(clean_albedo_path, original_normal_path)
        print(f"  Loaded textures: {albedo.shape}")
        
        # Step 2: Generate target poison map (strictly positive values)
        texture_shape = (albedo.shape[0], albedo.shape[1])
        poison_target = self._generate_poison_map(texture_shape, user_seed, poison_strength)
        print(f"  Generated poison map: range [0.0, {poison_strength}]")
        
        # Step 3: Apply poison to albedo (brightening attempt)
        albedo_poisoned, poison_effective = self._apply_poison_to_albedo_safe(albedo, poison_target)
        print(f"  Applied poison to albedo (with saturation handling)")
        
        # Step 4: Calculate analytical antidote for normal map
        normal_antidote = self._calculate_analytical_antidote(normal, poison_effective, user_seed)
        print(f"  Calculated analytical antidote normal map")
        
        # Step 5: Save outputs (NO calibration - the algebra IS the calibration)
        self._save_outputs(albedo_poisoned, normal_antidote, user_seed)
        print(f"  Saved bound assets: bound_albedo_{user_seed}.png, bound_normal_{user_seed}.png")
        print(f"Analytically Safe Binding completed for user {user_seed}")
    
    def _load_and_validate_inputs(self, albedo_path: str, normal_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate input textures, converting to appropriate formats.
        
        Args:
            albedo_path (str): Path to albedo texture file
            normal_path (str): Path to normal map file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed albedo and normal arrays
            
        Raises:
            FileNotFoundError: If texture files do not exist
            PIL.UnidentifiedImageError: If image format is not supported
        """
        # Validate file existence
        import os
        if not os.path.exists(albedo_path):
            raise FileNotFoundError(f"Albedo texture file not found: {albedo_path}")
        if not os.path.exists(normal_path):
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
            
            # Unpack normal maps from [0, 255] to [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
            normal_unpacked = (normal_array / self.RGB_MAX) * 2.0 - 1.0
            
            # Normalize normal vectors to ensure unit length
            normal_normalized = self._normalize_normal_vectors(normal_unpacked)
            
            return albedo_normalized, normal_normalized
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Failed to load or process image files: {str(e)}")
    
    def _normalize_normal_vectors(self, normal_array: np.ndarray) -> np.ndarray:
        """
        Normalize normal vectors to ensure unit length.
        
        Args:
            normal_array (np.ndarray): Input normal vectors [-1.0, 1.0]
            
        Returns:
            np.ndarray: Normalized normal vectors
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
            poison_strength (float): Poison strength value to validate
            
        Raises:
            ValueError: If poison_strength is outside valid range [0.0, 1.0]
        """
        if not isinstance(poison_strength, (int, float)):
            raise ValueError(f"poison_strength must be a number, got {type(poison_strength)}")
        
        if not (0.0 <= poison_strength <= 1.0):
            raise ValueError(f"poison_strength must be in range [0.0, 1.0], got {poison_strength}")
    
    def _generate_poison_map(self, shape: Tuple[int, int], seed: int, poison_strength: float) -> np.ndarray:
        """
        Generate target poison map with strictly positive values for brightening.
        
        Args:
            shape (Tuple[int, int]): Shape of the output poison array (height, width)
            seed (int): Random seed for deterministic generation
            poison_strength (float): Maximum poison strength [0.0, 1.0]
            
        Returns:
            np.ndarray: Poison map with values in range [0.0, poison_strength]
        """
        height, width = shape
        
        # Initialize deterministic random generator
        rng = np.random.RandomState(seed)
        
        # Calculate 4x4 blocks for BC7 compression resistance
        blocks_height = (height + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        blocks_width = (width + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        
        # Generate random values in range [0.0, poison_strength] (strictly positive)
        block_values = rng.uniform(0.0, poison_strength, size=(blocks_height, blocks_width))
        
        # Expand blocks to full resolution using nearest-neighbor (compression resilient)
        poison_map = np.repeat(np.repeat(block_values, self.BLOCK_SIZE, axis=0), self.BLOCK_SIZE, axis=1)
        
        # Crop to exact dimensions
        poison_map = poison_map[:height, :width]
        
        return poison_map.astype(np.float32)
    
    def _apply_poison_to_albedo_safe(self, albedo: np.ndarray, poison_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply poison to albedo with saturation safety check.
        
        This method handles the "White Pixel Trap" where bright pixels can't be brightened
        further due to clipping. We calculate the effective poison after clipping.
        
        Args:
            albedo (np.ndarray): Input albedo texture [0.0, 1.0]
            poison_target (np.ndarray): Target poison values [0.0, poison_strength]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Poisoned albedo and effective poison arrays
        """
        # Expand poison to RGB channels
        if len(poison_target.shape) == 2:
            poison_expanded = np.expand_dims(poison_target, axis=2)
            poison_expanded = np.repeat(poison_expanded, 3, axis=2)
        else:
            poison_expanded = poison_target
        
        # Apply brightening: A_new = A * (1 + P_target)
        albedo_attempted = albedo * (1.0 + poison_expanded)
        
        # Clip to valid range [0.0, 1.0] (this may cause saturation)
        albedo_clipped = np.clip(albedo_attempted, 0.0, 1.0)
        
        # Calculate effective poison after clipping (handles White Pixel Trap)
        # P_effective = (A_clipped / (A_original + epsilon)) - 1.0
        albedo_protected = albedo + self.EPSILON
        poison_effective = (albedo_clipped / albedo_protected) - 1.0
        
        # Ensure poison_effective is non-negative (clipping can't make things darker)
        poison_effective = np.maximum(poison_effective, 0.0)
        
        # Statistics
        saturation_pixels = np.sum(albedo_attempted > 1.0)
        total_pixels = albedo_attempted.size
        avg_poison_target = np.mean(poison_target)
        avg_poison_effective = np.mean(poison_effective)
        
        print(f"    Poison application statistics:")
        print(f"      Target poison: {avg_poison_target:.4f} (range: [0.0, {np.max(poison_target):.4f}])")
        print(f"      Effective poison: {avg_poison_effective:.4f} (range: [0.0, {np.max(poison_effective):.4f}])")
        print(f"      Saturated pixels: {saturation_pixels} / {total_pixels} ({100*saturation_pixels/total_pixels:.1f}%)")
        
        return albedo_clipped, poison_effective
    
    def _calculate_analytical_antidote(self, normal: np.ndarray, poison_effective: np.ndarray, 
                                     user_seed: int) -> np.ndarray:
        """
        Calculate analytical antidote normal map using strict algebraic cancellation.
        
        This method implements the core mathematical guarantee:
        Z_new = Z_old / (1 + P_effective)
        
        Since P_effective >= 0, Z_new <= Z_old (always steeper/rougher normals).
        This is geometrically safe and requires no calibration.
        
        Args:
            normal (np.ndarray): Original normal map [-1.0, 1.0]
            poison_effective (np.ndarray): Effective poison values [0.0, max_poison]
            user_seed (int): User seed for deterministic flat surface handling
            
        Returns:
            np.ndarray: Antidote normal map with algebraic cancellation
        """
        print(f"    Calculating analytical antidote...")
        
        # Extract original normal components
        x_old = normal[:, :, 0]
        y_old = normal[:, :, 1]
        z_old = normal[:, :, 2]
        
        # Convert poison to scalar (average RGB channels if needed)
        if len(poison_effective.shape) == 3:
            poison_scalar = np.mean(poison_effective, axis=2)
        else:
            poison_scalar = poison_effective
        
        # Calculate new Z using algebraic cancellation: Z_new = Z_old / (1 + P)
        z_new = z_old / (1.0 + poison_scalar)
        
        # Geometric validation: Z_new should always be <= Z_old (steeper normals)
        steepening_check = np.all(z_new <= z_old + self.EPSILON)
        if not steepening_check:
            print(f"      Warning: Some normals became flatter (should not happen)")
        
        # Calculate required lateral magnitudes for unit vector constraint
        lat_old = np.sqrt(np.maximum(0.0, 1.0 - z_old**2))
        lat_new = np.sqrt(np.maximum(0.0, 1.0 - z_new**2))
        
        # Handle flat surfaces (lat_old ≈ 0) with deterministic random directions
        flat_threshold = 0.01
        is_flat = lat_old < flat_threshold
        
        # For flat surfaces: generate deterministic random directions based on user_seed
        rng = np.random.RandomState(user_seed)
        height, width = normal.shape[:2]
        
        # Create deterministic angles for flat pixels
        angles = np.zeros((height, width))
        flat_indices = np.where(is_flat)
        if len(flat_indices[0]) > 0:
            # Use pixel coordinates + user_seed for deterministic randomness
            pixel_seeds = flat_indices[0] * width + flat_indices[1] + user_seed
            for i, (row, col) in enumerate(zip(flat_indices[0], flat_indices[1])):
                local_rng = np.random.RandomState(int(pixel_seeds[i]) % (2**31))
                angles[row, col] = local_rng.uniform(0, 2 * np.pi)
        
        # Calculate new X and Y components
        x_new = np.where(
            is_flat,
            lat_new * np.cos(angles),  # Random direction for flat surfaces
            x_old * (lat_new / (lat_old + self.EPSILON))  # Scaled direction for curved surfaces
        )
        
        y_new = np.where(
            is_flat,
            lat_new * np.sin(angles),  # Random direction for flat surfaces
            y_old * (lat_new / (lat_old + self.EPSILON))  # Scaled direction for curved surfaces
        )
        
        # Reconstruct normal map
        normal_antidote = np.stack([x_new, y_new, z_new], axis=2)
        
        # Normalize to ensure unit vectors (safety check)
        normal_antidote = self._normalize_normal_vectors(normal_antidote)
        
        # Statistics
        flat_pixels = np.sum(is_flat)
        avg_z_reduction = np.mean(z_old - z_new)
        max_z_reduction = np.max(z_old - z_new)
        
        print(f"      Antidote statistics:")
        print(f"        Flat pixels handled: {flat_pixels} / {is_flat.size} ({100*flat_pixels/is_flat.size:.1f}%)")
        print(f"        Average Z reduction (steepening): {avg_z_reduction:.4f}")
        print(f"        Maximum Z reduction: {max_z_reduction:.4f}")
        print(f"        Z range: [{np.min(z_new):.4f}, {np.max(z_new):.4f}]")
        print(f"        Geometric validation: {'PASSED' if steepening_check else 'FAILED'}")
        
        return normal_antidote
    
    def _save_outputs(self, bound_albedo: np.ndarray, bound_normal: np.ndarray, seed: int) -> None:
        """
        Save processed textures with appropriate format conversion and naming.
        
        Args:
            bound_albedo (np.ndarray): Processed albedo texture [0.0, 1.0]
            bound_normal (np.ndarray): Processed normal map [-1.0, 1.0]
            seed (int): User seed for filename generation
            
        Raises:
            IOError: If file saving fails
        """
        try:
            # Pack float albedo values [0.0, 1.0] back to RGB format [0, 255]
            albedo_rgb = np.clip(bound_albedo * self.RGB_MAX, 0, 255).astype(np.uint8)
            
            # Pack normal vectors [-1.0, 1.0] to RGB using formula (Vector + 1.0) / 2.0 * 255.0
            normal_rgb = np.clip((bound_normal + 1.0) / 2.0 * self.RGB_MAX, 0, 255).astype(np.uint8)
            
            # Generate output filenames: bound_albedo_{seed}.png and bound_normal_{seed}.png
            albedo_filename = f"bound_albedo_{seed}.png"
            normal_filename = f"bound_normal_{seed}.png"
            
            # Convert numpy arrays to PIL Images
            albedo_image = Image.fromarray(albedo_rgb, mode='RGB')
            normal_image = Image.fromarray(normal_rgb, mode='RGB')
            
            # Save files in PNG format maintaining original dimensions and quality
            albedo_image.save(albedo_filename, format='PNG', optimize=False)
            normal_image.save(normal_filename, format='PNG', optimize=False)
            
        except Exception as e:
            raise IOError(f"Failed to save output files: {str(e)}")