"""
Validation utilities for perceptual interdependence binding system.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from ..core.asset_binder import AssetBinder
from ..algorithms.cpu_math import get_cpu_math


class ValidationSuite:
    """
    Comprehensive validation suite for binding operations and mathematical correctness.
    
    Provides methods to validate binding results, mathematical properties,
    and system performance.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.cpu_math = get_cpu_math()
        self.tolerance = 1e-4
    
    def validate_binding_result(
        self, 
        original_albedo: np.ndarray,
        original_normal: np.ndarray,
        bound_albedo: np.ndarray,
        bound_normal: np.ndarray,
        poison_strength: float
    ) -> Dict[str, Any]:
        """
        Validate the mathematical correctness of binding results.
        
        Args:
            original_albedo: Original albedo texture [0,1]
            original_normal: Original normal map [0,1] 
            bound_albedo: Bound albedo texture [0,1]
            bound_normal: Bound normal map [0,1]
            poison_strength: Applied poison strength
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'issues': [],
            'statistics': {},
            'mathematical_properties': {}
        }
        
        # Check array shapes
        if original_albedo.shape != bound_albedo.shape:
            results['issues'].append("Albedo shape mismatch")
            results['valid'] = False
        
        if original_normal.shape != bound_normal.shape:
            results['issues'].append("Normal shape mismatch") 
            results['valid'] = False
        
        # Check value ranges
        if not self._check_value_range(bound_albedo, 0.0, 1.0):
            results['issues'].append("Bound albedo values outside [0,1] range")
            results['valid'] = False
        
        if not self._check_value_range(bound_normal, 0.0, 1.0):
            results['issues'].append("Bound normal values outside [0,1] range")
            results['valid'] = False
        
        # Check poison application (albedo should be brighter or equal)
        albedo_diff = bound_albedo - original_albedo
        if np.any(albedo_diff < -self.tolerance):
            results['issues'].append("Bound albedo is darker than original (poison should brighten)")
            results['valid'] = False
        
        # Check normal steepening (Z component should be smaller or equal)
        original_normal_vec = original_normal * 2.0 - 1.0
        bound_normal_vec = bound_normal * 2.0 - 1.0
        
        z_diff = bound_normal_vec[:,:,2] - original_normal_vec[:,:,2]
        if np.any(z_diff > self.tolerance):
            results['issues'].append("Bound normals are flatter than original (antidote should steepen)")
            results['valid'] = False
        
        # Calculate statistics
        results['statistics'] = {
            'albedo_brightness_increase': float(np.mean(albedo_diff)),
            'normal_z_decrease': float(np.mean(-z_diff)),
            'saturation_ratio': float(np.sum(bound_albedo >= 0.999) / bound_albedo.size),
            'max_poison_strength': float(np.max(albedo_diff / np.maximum(original_albedo, 1e-6)))
        }
        
        # Mathematical properties
        results['mathematical_properties'] = {
            'poison_strength_target': poison_strength,
            'poison_strength_achieved': results['statistics']['max_poison_strength'],
            'geometric_consistency': self._check_normal_unit_vectors(bound_normal_vec),
            'algebraic_cancellation': self._estimate_cancellation_quality(
                original_albedo, original_normal, bound_albedo, bound_normal
            )
        }
        
        return results
    
    def validate_poison_map(
        self, 
        poison_map: np.ndarray, 
        expected_strength: float,
        seed: int
    ) -> Dict[str, Any]:
        """
        Validate poison map properties.
        
        Args:
            poison_map: Generated poison map
            expected_strength: Expected maximum poison strength
            seed: Random seed used for generation
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'properties': {}
        }
        
        # Check value range
        if not self._check_value_range(poison_map, 0.0, expected_strength):
            results['issues'].append(f"Poison values outside [0, {expected_strength}] range")
            results['valid'] = False
        
        # Check for negative values (poison should always be positive)
        if np.any(poison_map < 0):
            results['issues'].append("Poison map contains negative values")
            results['valid'] = False
        
        # Check determinism (regenerate with same seed)
        poison_map_2 = self.cpu_math.generate_poison_map(poison_map.shape, seed, expected_strength)
        if not np.allclose(poison_map, poison_map_2, atol=1e-6):
            results['issues'].append("Poison map generation is not deterministic")
            results['valid'] = False
        
        # Calculate properties
        results['properties'] = {
            'mean_strength': float(np.mean(poison_map)),
            'max_strength': float(np.max(poison_map)),
            'min_strength': float(np.min(poison_map)),
            'std_strength': float(np.std(poison_map)),
            'zero_ratio': float(np.sum(poison_map == 0) / poison_map.size),
            'block_structure': self._analyze_block_structure(poison_map)
        }
        
        return results
    
    def benchmark_performance(
        self, 
        image_sizes: List[Tuple[int, int]] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark system performance across different image sizes.
        
        Args:
            image_sizes: List of (width, height) tuples to test
            iterations: Number of iterations per size
            
        Returns:
            Performance benchmark results
        """
        if image_sizes is None:
            image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        
        results = {
            'cpu_math_available': self.cpu_math.use_numba,
            'benchmarks': {}
        }
        
        for size in image_sizes:
            size_key = f"{size[0]}x{size[1]}"
            times = []
            
            for _ in range(iterations):
                benchmark_result = self.cpu_math.benchmark_performance(size)
                times.append(benchmark_result['total'])
            
            results['benchmarks'][size_key] = {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput_mpixels_per_sec': float((size[0] * size[1]) / np.mean(times) / 1e6)
            }
        
        return results
    
    def validate_system_integrity(self) -> Dict[str, Any]:
        """
        Perform comprehensive system integrity validation.
        
        Returns:
            System integrity results
        """
        results = {
            'valid': True,
            'issues': [],
            'components': {}
        }
        
        # Test CPU math module
        try:
            test_shape = (64, 64)
            poison_map = self.cpu_math.generate_poison_map(test_shape, 42, 0.2)
            
            test_albedo = np.random.uniform(0.3, 0.9, (*test_shape, 3)).astype(np.float32)
            poisoned_albedo, effective_poison = self.cpu_math.apply_poison(test_albedo, poison_map)
            
            test_normal = np.random.uniform(0.0, 1.0, (*test_shape, 3)).astype(np.float32)
            antidote_normal = self.cpu_math.calculate_antidote(test_normal, effective_poison)
            
            results['components']['cpu_math'] = {
                'available': True,
                'numba_acceleration': self.cpu_math.use_numba,
                'test_passed': True
            }
            
        except Exception as e:
            results['issues'].append(f"CPU math module failed: {e}")
            results['valid'] = False
            results['components']['cpu_math'] = {
                'available': False,
                'error': str(e)
            }
        
        # Test AssetBinder
        try:
            binder = AssetBinder()
            results['components']['asset_binder'] = {
                'available': True,
                'test_passed': True
            }
        except Exception as e:
            results['issues'].append(f"AssetBinder failed: {e}")
            results['valid'] = False
            results['components']['asset_binder'] = {
                'available': False,
                'error': str(e)
            }
        
        return results
    
    def _check_value_range(
        self, 
        array: np.ndarray, 
        min_val: float, 
        max_val: float
    ) -> bool:
        """Check if all array values are within specified range."""
        return np.all((array >= min_val) & (array <= max_val))
    
    def _check_normal_unit_vectors(self, normal_vectors: np.ndarray) -> Dict[str, Any]:
        """Check if normal vectors have unit length."""
        magnitudes = np.linalg.norm(normal_vectors, axis=2)
        
        return {
            'mean_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'min_magnitude': float(np.min(magnitudes)),
            'max_magnitude': float(np.max(magnitudes)),
            'unit_vector_ratio': float(np.sum(np.abs(magnitudes - 1.0) < 0.01) / magnitudes.size)
        }
    
    def _estimate_cancellation_quality(
        self,
        original_albedo: np.ndarray,
        original_normal: np.ndarray, 
        bound_albedo: np.ndarray,
        bound_normal: np.ndarray
    ) -> Dict[str, Any]:
        """Estimate the quality of algebraic cancellation."""
        # Simulate rendering with original textures
        original_render = self._simulate_simple_render(original_albedo, original_normal)
        
        # Simulate rendering with bound textures  
        bound_render = self._simulate_simple_render(bound_albedo, bound_normal)
        
        # Calculate difference
        render_diff = np.abs(bound_render - original_render)
        
        return {
            'mean_render_difference': float(np.mean(render_diff)),
            'max_render_difference': float(np.max(render_diff)),
            'cancellation_quality': float(1.0 - np.mean(render_diff))
        }
    
    def _simulate_simple_render(
        self, 
        albedo: np.ndarray, 
        normal: np.ndarray
    ) -> np.ndarray:
        """Simulate simple Lambertian rendering."""
        # Convert normal to vector format
        normal_vec = normal * 2.0 - 1.0
        
        # Simple directional light from above
        light_dir = np.array([0, 0, 1])
        
        # Calculate dot product (Lambertian shading)
        dot_product = np.sum(normal_vec * light_dir, axis=2, keepdims=True)
        dot_product = np.maximum(dot_product, 0.0)  # Clamp to positive
        
        # Apply to albedo
        return albedo * dot_product
    
    def _analyze_block_structure(self, poison_map: np.ndarray) -> Dict[str, Any]:
        """Analyze block structure in poison map."""
        block_size = 4
        height, width = poison_map.shape
        
        # Check if dimensions are compatible with block structure
        blocks_h = height // block_size
        blocks_w = width // block_size
        
        if blocks_h * block_size != height or blocks_w * block_size != width:
            return {'block_structure_detected': False}
        
        # Analyze block consistency
        block_variances = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = poison_map[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_variances.append(np.var(block))
        
        return {
            'block_structure_detected': True,
            'mean_block_variance': float(np.mean(block_variances)),
            'block_consistency': float(np.sum(np.array(block_variances) < 1e-6) / len(block_variances))
        }