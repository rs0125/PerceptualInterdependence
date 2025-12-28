"""
Unit tests for AssetBinder class.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.perceptual_interdependence.core.asset_binder import AssetBinder


class TestAssetBinder:
    """Test suite for AssetBinder class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_textures(self, temp_dir):
        """Create test texture files."""
        # Create test albedo (256x256 RGB)
        albedo_array = np.random.uniform(0.3, 0.9, (256, 256, 3))
        albedo_image = Image.fromarray((albedo_array * 255).astype(np.uint8), mode='RGB')
        albedo_path = temp_dir / "test_albedo.png"
        albedo_image.save(albedo_path)
        
        # Create test normal map (256x256 RGB)
        normal_array = np.random.uniform(0.0, 1.0, (256, 256, 3))
        normal_image = Image.fromarray((normal_array * 255).astype(np.uint8), mode='RGB')
        normal_path = temp_dir / "test_normal.png"
        normal_image.save(normal_path)
        
        return {
            'albedo_path': albedo_path,
            'normal_path': normal_path,
            'albedo_array': albedo_array,
            'normal_array': normal_array
        }
    
    @pytest.fixture
    def asset_binder(self, temp_dir):
        """Create AssetBinder instance with temp output directory."""
        return AssetBinder(output_dir=temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test AssetBinder initialization."""
        binder = AssetBinder(output_dir=temp_dir)
        
        assert binder.output_dir == temp_dir
        assert hasattr(binder, 'cpu_math')
        assert binder.EPSILON == 1e-6
        assert binder.RGB_MAX == 255.0
    
    def test_bind_textures_success(self, asset_binder, test_textures, temp_dir):
        """Test successful texture binding."""
        user_id = 42
        poison_strength = 0.2
        
        results = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=user_id,
            poison_strength=poison_strength
        )
        
        # Check return value structure
        assert isinstance(results, dict)
        assert 'user_id' in results
        assert 'poison_strength' in results
        assert 'texture_shape' in results
        assert 'output_paths' in results
        assert 'statistics' in results
        
        assert results['user_id'] == user_id
        assert results['poison_strength'] == poison_strength
        
        # Check output files exist
        albedo_output = results['output_paths']['albedo']
        normal_output = results['output_paths']['normal']
        
        assert albedo_output.exists()
        assert normal_output.exists()
        assert albedo_output.suffix == '.png'
        assert normal_output.suffix == '.png'
        
        # Check file naming
        assert f"bound_albedo_{user_id}.png" in str(albedo_output)
        assert f"bound_normal_{user_id}.png" in str(normal_output)
    
    def test_bind_textures_custom_prefix(self, asset_binder, test_textures):
        """Test texture binding with custom output prefix."""
        user_id = 123
        custom_prefix = "custom"
        
        results = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=user_id,
            output_prefix=custom_prefix
        )
        
        # Check custom prefix in filenames
        albedo_output = results['output_paths']['albedo']
        normal_output = results['output_paths']['normal']
        
        assert f"{custom_prefix}_albedo_{user_id}.png" in str(albedo_output)
        assert f"{custom_prefix}_normal_{user_id}.png" in str(normal_output)
    
    def test_invalid_poison_strength(self, asset_binder, test_textures):
        """Test validation of poison strength parameter."""
        # Test negative poison strength
        with pytest.raises(ValueError, match="poison_strength must be in range"):
            asset_binder.bind_textures(
                albedo_path=test_textures['albedo_path'],
                normal_path=test_textures['normal_path'],
                user_id=42,
                poison_strength=-0.1
            )
        
        # Test poison strength > 1.0
        with pytest.raises(ValueError, match="poison_strength must be in range"):
            asset_binder.bind_textures(
                albedo_path=test_textures['albedo_path'],
                normal_path=test_textures['normal_path'],
                user_id=42,
                poison_strength=1.5
            )
        
        # Test non-numeric poison strength
        with pytest.raises(ValueError, match="poison_strength must be a number"):
            asset_binder.bind_textures(
                albedo_path=test_textures['albedo_path'],
                normal_path=test_textures['normal_path'],
                user_id=42,
                poison_strength="invalid"
            )
    
    def test_missing_input_files(self, asset_binder, temp_dir):
        """Test handling of missing input files."""
        # Test missing albedo file
        with pytest.raises(FileNotFoundError, match="Albedo texture file not found"):
            asset_binder.bind_textures(
                albedo_path=temp_dir / "missing_albedo.png",
                normal_path=temp_dir / "missing_normal.png",
                user_id=42
            )
    
    def test_load_and_validate_inputs(self, asset_binder, test_textures):
        """Test input loading and validation."""
        albedo, normal = asset_binder._load_and_validate_inputs(
            test_textures['albedo_path'],
            test_textures['normal_path']
        )
        
        # Check shapes
        assert albedo.shape == (256, 256, 3)
        assert normal.shape == (256, 256, 3)
        
        # Check value ranges
        assert np.all(albedo >= 0.0)
        assert np.all(albedo <= 1.0)
        assert np.all(normal >= 0.0)
        assert np.all(normal <= 1.0)
        
        # Check data types
        assert albedo.dtype == np.float32
        assert normal.dtype == np.float32
    
    def test_normalize_normal_vectors(self, asset_binder):
        """Test normal vector normalization."""
        # Create test normal vectors (not normalized)
        test_vectors = np.array([
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]
        ], dtype=np.float32)
        
        normalized = asset_binder._normalize_normal_vectors(test_vectors)
        
        # Check that vectors have unit length
        magnitudes = np.linalg.norm(normalized, axis=2)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-6)
    
    def test_deterministic_output(self, asset_binder, test_textures):
        """Test that binding produces deterministic results."""
        user_id = 42
        poison_strength = 0.2
        
        # Bind textures twice with same parameters
        results1 = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=user_id,
            poison_strength=poison_strength,
            output_prefix="test1"
        )
        
        results2 = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=user_id,
            poison_strength=poison_strength,
            output_prefix="test2"
        )
        
        # Load and compare output images
        albedo1 = np.array(Image.open(results1['output_paths']['albedo']))
        albedo2 = np.array(Image.open(results2['output_paths']['albedo']))
        
        normal1 = np.array(Image.open(results1['output_paths']['normal']))
        normal2 = np.array(Image.open(results2['output_paths']['normal']))
        
        # Should be identical (deterministic)
        np.testing.assert_array_equal(albedo1, albedo2)
        np.testing.assert_array_equal(normal1, normal2)
    
    def test_different_user_ids(self, asset_binder, test_textures):
        """Test that different user IDs produce different results."""
        poison_strength = 0.2
        
        # Bind with different user IDs
        results1 = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=42,
            poison_strength=poison_strength,
            output_prefix="user42"
        )
        
        results2 = asset_binder.bind_textures(
            albedo_path=test_textures['albedo_path'],
            normal_path=test_textures['normal_path'],
            user_id=99,
            poison_strength=poison_strength,
            output_prefix="user99"
        )
        
        # Load output images
        albedo1 = np.array(Image.open(results1['output_paths']['albedo']))
        albedo2 = np.array(Image.open(results2['output_paths']['albedo']))
        
        # Should be different (different random seeds)
        assert not np.array_equal(albedo1, albedo2)


if __name__ == '__main__':
    pytest.main([__file__])