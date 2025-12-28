#!/usr/bin/env python3
"""
CPU-Optimized Mathematical Operations for Asset Binding

This module provides highly optimized CPU implementations of core mathematical
operations using NumPy vectorization, advanced indexing, and efficient algorithms.
Replaces GPU acceleration with faster CPU-based approaches.
"""

import numpy as np
from typing import Tuple
import warnings

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("⚡ JIT compilation available (Numba)")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  JIT compilation not available")


# Numba-compiled functions for maximum performance
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _poison_map_kernel(height: int, width: int, poison_strength: float, 
                          block_noise: np.ndarray) -> np.ndarray:
        """Ultra-fast Numba-compiled poison map generation."""
        poison_map = np.zeros((height, width), dtype=np.float32)
        block_size = 4
        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size
        
        center_y, center_x = height // 2, width // 2
        max_distance = np.sqrt(center_y * center_y + center_x * center_x)
        
        for block_i in prange(blocks_h):
            for block_j in prange(blocks_w):
                block_value = block_noise[block_i, block_j]
                
                start_i = block_i * block_size
                start_j = block_j * block_size
                end_i = min(start_i + block_size, height)
                end_j = min(start_j + block_size, width)
                
                for i in range(start_i, end_i):
                    for j in range(start_j, end_j):
                        # Frequency weighting
                        distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                        freq_weight = 1.0 - (distance / max_distance) * 0.3
                        freq_weight = max(0.7, min(1.0, freq_weight))
                        
                        poison_map[i, j] = block_value * freq_weight
        
        return poison_map
    
    @jit(nopython=True, parallel=True, cache=True)
    def _poison_application_kernel(albedo: np.ndarray, poison_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast Numba-compiled poison application."""
        height, width, channels = albedo.shape
        poisoned_albedo = np.zeros_like(albedo)
        effective_poison = np.zeros((height, width), dtype=np.float32)
        
        for i in prange(height):
            for j in prange(width):
                poison_val = poison_map[i, j]
                max_effective = 0.0
                
                for c in range(channels):
                    original_val = albedo[i, j, c]
                    target_val = original_val * (1.0 + poison_val)
                    
                    if target_val > 1.0:
                        poisoned_albedo[i, j, c] = 1.0
                        if original_val > 0.001:
                            channel_effective = (1.0 / original_val) - 1.0
                        else:
                            channel_effective = poison_val
                    else:
                        poisoned_albedo[i, j, c] = target_val
                        channel_effective = poison_val
                    
                    max_effective = max(max_effective, channel_effective)
                
                effective_poison[i, j] = max_effective
        
        return poisoned_albedo, effective_poison
    
    @jit(nopython=True, parallel=True, cache=True)
    def _antidote_calculation_kernel(normal: np.ndarray, effective_poison: np.ndarray) -> np.ndarray:
        """Ultra-fast Numba-compiled antidote calculation using Analytically Safe logic."""
        height, width, _ = normal.shape
        antidote_normal = np.zeros_like(normal)
        
        for i in prange(height):
            for j in prange(width):
                # Convert RGB to normal vector
                nx = normal[i, j, 0] * 2.0 - 1.0
                ny = normal[i, j, 1] * 2.0 - 1.0
                nz = normal[i, j, 2] * 2.0 - 1.0
                
                # Get effective poison for this pixel
                p_effective = effective_poison[i, j]
                
                # ANALYTICALLY SAFE LOGIC: Z_new = Z_old / (1 + P_effective)
                nz_new = nz / (1.0 + p_effective)
                
                # Calculate lateral lengths for vector reconstruction
                lat_len_old = np.sqrt(max(0.0, 1.0 - nz*nz))
                lat_len_new = np.sqrt(max(0.0, 1.0 - nz_new*nz_new))
                
                # Reconstruct X/Y components
                if lat_len_old < 1e-4:
                    # Handle flat surface (original normal was [0,0,1])
                    # Generate pseudo-random direction based on coordinates
                    angle = (i * 12.9898 + j * 78.233) * 43758.5453
                    angle = angle - np.floor(angle)  # Get fractional part
                    angle = angle * 6.28318530718  # Convert to radians [0, 2π]
                    nx_new = lat_len_new * np.cos(angle)
                    ny_new = lat_len_new * np.sin(angle)
                else:
                    # Scale existing direction
                    ratio = lat_len_new / lat_len_old
                    nx_new = nx * ratio
                    ny_new = ny * ratio
                
                # Convert back to RGB format
                antidote_normal[i, j, 0] = (nx_new + 1.0) * 0.5
                antidote_normal[i, j, 1] = (ny_new + 1.0) * 0.5
                antidote_normal[i, j, 2] = (nz_new + 1.0) * 0.5
        
        return antidote_normal


class CPUOptimizedMath:
    """
    CPU-optimized mathematical operations for asset binding.
    
    Uses advanced NumPy vectorization, efficient algorithms, and Numba JIT
    compilation for maximum performance without GPU dependencies.
    """
    
    def __init__(self):
        """Initialize CPU-optimized math operations."""
        self.use_numba = NUMBA_AVAILABLE
        
        if self.use_numba:
            print("🚀 CPU Math: Numba JIT acceleration enabled")
        else:
            print("🖥️  CPU Math: Pure NumPy vectorization")
    
    def generate_poison_map(self, shape: Tuple[int, int], seed: int, 
                           poison_strength: float = 0.2) -> np.ndarray:
        """
        Generate poison map using optimized CPU algorithms.
        
        Uses block-based generation with vectorized frequency weighting
        for maximum performance and compression resistance.
        
        Args:
            shape: Output shape (height, width)
            seed: Random seed for reproducibility
            poison_strength: Maximum poison strength
            
        Returns:
            Generated poison map
        """
        np.random.seed(seed)
        height, width = shape
        
        if self.use_numba:
            return self._generate_poison_map_numba(height, width, poison_strength)
        else:
            return self._generate_poison_map_vectorized(height, width, poison_strength)
    
    def _generate_poison_map_numba(self, height: int, width: int, poison_strength: float) -> np.ndarray:
        """Numba-accelerated poison map generation."""
        block_size = 4
        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size
        
        # Generate block noise
        block_noise = np.random.uniform(0, poison_strength, (blocks_h, blocks_w)).astype(np.float32)
        
        return _poison_map_kernel(height, width, poison_strength, block_noise)
    
    def _generate_poison_map_vectorized(self, height: int, width: int, poison_strength: float) -> np.ndarray:
        """Highly optimized vectorized poison map generation."""
        block_size = 4
        
        # Generate block-based noise using advanced indexing
        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size
        
        # Create block noise
        block_noise = np.random.uniform(0, poison_strength, (blocks_h, blocks_w))
        
        # Efficient block expansion using repeat and crop
        poison_map = np.repeat(np.repeat(block_noise, block_size, axis=0), block_size, axis=1)
        poison_map = poison_map[:height, :width]
        
        # Vectorized frequency weighting
        y_coords, x_coords = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Distance calculation (vectorized)
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        
        # Frequency mask (vectorized)
        freq_mask = 1.0 - (distances / max_distance) * 0.3
        freq_mask = np.clip(freq_mask, 0.7, 1.0)
        
        return (poison_map * freq_mask).astype(np.float32)
    
    def apply_poison(self, albedo: np.ndarray, poison_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply poison to albedo texture using optimized CPU algorithms.
        
        Uses vectorized operations with efficient saturation handling.
        
        Args:
            albedo: Input albedo texture [0.0, 1.0]
            poison_map: Poison map [0.0, poison_strength]
            
        Returns:
            Tuple of (poisoned_albedo, effective_poison)
        """
        if self.use_numba:
            return _poison_application_kernel(albedo, poison_map)
        else:
            return self._apply_poison_vectorized(albedo, poison_map)
    
    def _apply_poison_vectorized(self, albedo: np.ndarray, poison_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Highly optimized vectorized poison application."""
        # Expand poison map to match albedo channels
        poison_expanded = poison_map[..., np.newaxis]
        
        # Vectorized poison application
        target_albedo = albedo * (1.0 + poison_expanded)
        
        # Efficient saturation handling
        saturated_mask = target_albedo > 1.0
        poisoned_albedo = np.where(saturated_mask, 1.0, target_albedo)
        
        # Calculate effective poison (vectorized)
        # For saturated pixels: effective = (1.0 / original) - 1.0
        # For normal pixels: effective = original poison
        safe_albedo = np.where(albedo < 1e-6, 1e-6, albedo)  # Avoid division by zero
        saturation_effective = (1.0 / safe_albedo) - 1.0
        
        # Use maximum effective poison across channels
        effective_poison = np.where(
            np.any(saturated_mask, axis=2, keepdims=True),
            np.max(saturation_effective, axis=2, keepdims=True),
            poison_expanded
        ).squeeze(axis=2)
        
        return poisoned_albedo, effective_poison
    
    def calculate_antidote(self, normal: np.ndarray, effective_poison: np.ndarray) -> np.ndarray:
        """
        Calculate antidote normal map using optimized CPU algorithms.
        
        Uses vectorized normal vector operations with geometric constraints.
        
        Args:
            normal: Input normal map in RGB format [0.0, 1.0]
            effective_poison: Effective poison map
            
        Returns:
            Antidote normal map in RGB format
        """
        if self.use_numba:
            return _antidote_calculation_kernel(normal, effective_poison)
        else:
            return self._calculate_antidote_vectorized(normal, effective_poison)
    
    def _calculate_antidote_vectorized(self, normal: np.ndarray, effective_poison: np.ndarray) -> np.ndarray:
        """Highly optimized vectorized antidote calculation using Analytically Safe logic."""
        # Convert RGB normal to vector format (vectorized)
        normal_vectors = normal * 2.0 - 1.0
        
        # Extract components
        nx, ny, nz = normal_vectors[..., 0], normal_vectors[..., 1], normal_vectors[..., 2]
        
        # ANALYTICALLY SAFE LOGIC: Z_new = Z_old / (1 + P_effective)
        nz_new = nz / (1.0 + effective_poison)
        
        # Calculate lateral lengths for vector reconstruction
        lat_len_old = np.sqrt(np.maximum(0.0, 1.0 - nz**2))
        lat_len_new = np.sqrt(np.maximum(0.0, 1.0 - nz_new**2))
        
        # Handle flat surfaces (where lat_len_old is very small)
        flat_mask = lat_len_old < 1e-4
        
        # For non-flat surfaces: scale existing direction
        ratio = np.where(lat_len_old > 1e-4, lat_len_new / lat_len_old, 1.0)
        nx_new = nx * ratio
        ny_new = ny * ratio
        
        # For flat surfaces: generate pseudo-random direction
        if np.any(flat_mask):
            y_coords, x_coords = np.ogrid[:normal.shape[0], :normal.shape[1]]
            angles = (x_coords * 12.9898 + y_coords * 78.233) * 43758.5453
            angles = (angles - np.floor(angles)) * 2 * np.pi  # Convert to [0, 2π]
            
            flat_nx = lat_len_new * np.cos(angles)
            flat_ny = lat_len_new * np.sin(angles)
            
            nx_new = np.where(flat_mask, flat_nx, nx_new)
            ny_new = np.where(flat_mask, flat_ny, ny_new)
        
        # Reconstruct normal vectors (vectorized)
        new_normal_vectors = np.stack([nx_new, ny_new, nz_new], axis=2)
        
        # Convert back to RGB format (vectorized)
        return (new_normal_vectors + 1.0) * 0.5
    
    def benchmark_performance(self, shape: Tuple[int, int] = (2048, 2048)) -> dict:
        """
        Benchmark performance of CPU implementations.
        
        Args:
            shape: Test image shape
            
        Returns:
            Performance results dictionary
        """
        import time
        
        print(f"🔬 Benchmarking CPU optimizations on {shape[0]}x{shape[1]} image...")
        
        results = {}
        
        # Test poison map generation
        start_time = time.time()
        poison_map = self.generate_poison_map(shape, 42, 0.2)
        poison_time = time.time() - start_time
        results['poison_map'] = poison_time
        print(f"  Poison Map Generation: {poison_time:.3f}s")
        
        # Create test albedo
        test_albedo = np.random.uniform(0.3, 0.9, (*shape, 3)).astype(np.float32)
        
        # Test poison application
        start_time = time.time()
        poisoned_albedo, effective_poison = self.apply_poison(test_albedo, poison_map)
        poison_apply_time = time.time() - start_time
        results['poison_application'] = poison_apply_time
        print(f"  Poison Application: {poison_apply_time:.3f}s")
        
        # Create test normal map
        test_normal = np.random.uniform(0.0, 1.0, (*shape, 3)).astype(np.float32)
        
        # Test antidote calculation
        start_time = time.time()
        antidote_normal = self.calculate_antidote(test_normal, effective_poison)
        antidote_time = time.time() - start_time
        results['antidote_calculation'] = antidote_time
        print(f"  Antidote Calculation: {antidote_time:.3f}s")
        
        # Total time
        total_time = poison_time + poison_apply_time + antidote_time
        results['total'] = total_time
        print(f"  Total Processing Time: {total_time:.3f}s")
        
        if self.use_numba:
            print(f"  🚀 Using Numba JIT acceleration")
        else:
            print(f"  📊 Using pure NumPy vectorization")
        
        return results


# Global instance for easy access
cpu_math = CPUOptimizedMath()


def get_cpu_math() -> CPUOptimizedMath:
    """
    Get CPU-optimized math instance.
    
    Returns:
        CPUOptimizedMath: Math operations instance
    """
    return cpu_math