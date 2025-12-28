"""
Unit tests for CPU-optimized mathematical operations.
"""

import pytest
import numpy as np
from src.perceptual_interdependence.algorithms.cpu_math import CPUOptimizedMath, get_cpu_math


class TestCPUOptimizedMath:
    """Test suite for CPU-optimized math operations."""
    
    @pytest.fixture
    def cpu_math(self):
        """Create CPU math instance for testing."""
        return get_cpu_math()
    
    @pytest.fixture
    def test_shape(self):
        """Standard test shape."""
        return (64, 64)
    
    @pytest.fixture
    def test_albedo(self, test_shape):
        """Create test albedo texture."""
        return np.random.uniform(0.3, 0.9, (*test_shape, 3)).astype(np.float32)
    
    @pytest.fixture
    def test_normal(self, test_shape):
        """Create test normal map."""
        return np.random.uniform(0.0, 1.0, (*test_shape, 3)).astype(np.float32)
    
    def test_poison_map_generation(self, cpu_math, test_shape):
        """Test poison map generation."""
        poison_strength = 0.2
        seed = 42
        
        poison_map = cpu_math.generate_poison_map(test_shape, seed, poison_strength)
        
        # Check shape
        assert poison_map.shape == test_shape
        
        # Check data type
        assert poison_map.dtype == np.float32
        
        # Check value range
        assert np.all(poison_map >= 0)
        assert np.all(poison_map <= poison_strength)
        
        # Check determinism
        poison_map_2 = cpu_math.generate_poison_map(test_shape, seed, poison_strength)
        np.testing.assert_allclose(poison_map, poison_map_2, atol=1e-6)
    
    def test_poison_application(self, cpu_math, test_albedo):
        """Test poison application to albedo."""
        poison_map = np.full(test_albedo.shape[:2], 0.2, dtype=np.float32)
        
        poisoned_albedo, effective_poison = cpu_math.apply_poison(test_albedo, poison_map)
        
        # Check shapes
        assert poisoned_albedo.shape == test_albedo.shape
        assert effective_poison.shape == test_albedo.shape[:2]
        
        # Check value ranges
        assert np.all(poisoned_albedo >= 0)
        assert np.all(poisoned_albedo <= 1.0)
        assert np.all(effective_poison >= 0)
        
        # Check that albedo is brightened (or stays same due to saturation)
        assert np.all(poisoned_albedo >= test_albedo - 1e-6)
    
    def test_antidote_calculation(self, cpu_math, test_normal):
        """Test antidote calculation for normal maps."""
        effective_poison = np.full(test_normal.shape[:2], 0.2, dtype=np.float32)
        
        antidote_normal = cpu_math.calculate_antidote(test_normal, effective_poison)
        
        # Check shape
        assert antidote_normal.shape == test_normal.shape
        
        # Check value range
        assert np.all(antidote_normal >= 0)
        assert np.all(antidote_normal <= 1.0)
        
        # Convert to vector format and check that normals are still valid
        original_vec = test_normal * 2.0 - 1.0
        antidote_vec = antidote_normal * 2.0 - 1.0
        
        # Check that normal vectors are still unit vectors (approximately)
        magnitudes = np.linalg.norm(antidote_vec, axis=2)
        assert np.allclose(magnitudes, 1.0, atol=0.1)  # Allow some tolerance for normalization
    
    def test_mathematical_correctness(self, cpu_math):
        """Test mathematical correctness of operations."""
        # Simple test case with known values
        shape = (4, 4)
        
        # Test albedo: uniform gray
        test_albedo = np.full((4, 4, 3), 0.5, dtype=np.float32)
        
        # Test poison: uniform 20%
        poison_uniform = np.full((4, 4), 0.2, dtype=np.float32)
        
        poisoned_albedo, effective_poison = cpu_math.apply_poison(test_albedo, poison_uniform)
        
        # Should brighten by 20%: 0.5 * 1.2 = 0.6
        expected_albedo = 0.6
        np.testing.assert_allclose(poisoned_albedo, expected_albedo, atol=0.01)
        
        # Effective poison should match target (no saturation)
        np.testing.assert_allclose(effective_poison, 0.2, atol=0.01)
    
    def test_performance_benchmark(self, cpu_math):
        """Test performance benchmarking."""
        shape = (256, 256)
        
        results = cpu_math.benchmark_performance(shape)
        
        # Check that all required keys are present
        required_keys = ['poison_map', 'poison_application', 'antidote_calculation', 'total']
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], float)
            assert results[key] > 0  # Should take some time
    
    def test_edge_cases(self, cpu_math):
        """Test edge cases and error conditions."""
        # Test with very small images
        small_shape = (2, 2)
        poison_map = cpu_math.generate_poison_map(small_shape, 42, 0.1)
        assert poison_map.shape == small_shape
        
        # Test with zero poison strength
        zero_poison = cpu_math.generate_poison_map((4, 4), 42, 0.0)
        assert np.all(zero_poison == 0.0)
        
        # Test with maximum poison strength
        max_poison = cpu_math.generate_poison_map((4, 4), 42, 1.0)
        assert np.all(max_poison <= 1.0)
    
    def test_numba_availability(self, cpu_math):
        """Test Numba availability detection."""
        # Should have use_numba attribute
        assert hasattr(cpu_math, 'use_numba')
        assert isinstance(cpu_math.use_numba, bool)
        
        # If Numba is available, should be faster
        if cpu_math.use_numba:
            print("✅ Numba JIT acceleration available")
        else:
            print("⚠️  Using pure NumPy (Numba not available)")


if __name__ == '__main__':
    pytest.main([__file__])