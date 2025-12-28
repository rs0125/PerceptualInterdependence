#!/usr/bin/env python3
"""
Direct test of the asset binder to see exact calculations
"""

import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.asset_binder_complex import AssetBinderComplex

def direct_binder_test():
    """Test the asset binder directly with debug output"""
    
    print("Testing asset binder directly...")
    
    # Create a simple test texture
    test_size = 8
    test_albedo = np.full((test_size, test_size, 3), 0.5, dtype=np.float32)
    test_normal = np.zeros((test_size, test_size, 3), dtype=np.float32)
    test_normal[:, :, 2] = 1.0  # Flat normal pointing up
    
    # Save test textures
    test_albedo_img = Image.fromarray((test_albedo * 255).astype(np.uint8))
    test_normal_packed = (test_normal + 1.0) / 2.0
    test_normal_img = Image.fromarray((test_normal_packed * 255).astype(np.uint8))
    
    test_albedo_img.save("test_albedo.png")
    test_normal_img.save("test_normal.png")
    
    print(f"Created test textures: {test_size}x{test_size}")
    print(f"Test albedo value: {test_albedo[0, 0]}")
    print(f"Test normal value: {test_normal[0, 0]}")
    
    # Create binder and bind
    binder = AssetBinderComplex()
    
    # Monkey patch the methods to add debug output
    original_generate_antidote = binder._generate_antidote_normal
    
    def debug_generate_antidote(normal, s_effective):
        print(f"\\n=== ANTIDOTE GENERATION DEBUG ===")
        print(f"Input normal shape: {normal.shape}")
        print(f"Input s_effective shape: {s_effective.shape}")
        
        # Sample pixel
        y, x = 0, 0
        print(f"\\nPixel ({y}, {x}) analysis:")
        print(f"Original normal: {normal[y, x]}")
        print(f"S_effective: {s_effective[y, x] if len(s_effective.shape) == 3 else s_effective[y, x]}")
        
        # Call original method
        result = original_generate_antidote(normal, s_effective)
        
        print(f"Result normal: {result[y, x]}")
        print(f"Z change: {normal[y, x, 2]} -> {result[y, x, 2]} (factor: {result[y, x, 2]/normal[y, x, 2]:.6f})")
        
        return result
    
    binder._generate_antidote_normal = debug_generate_antidote
    
    # Bind with debug
    print(f"\\nBinding with user seed 42, poison strength 0.3...")
    binder.bind_textures("test_albedo.png", "test_normal.png", user_seed=42, poison_strength=0.3)
    
    # Load and check results
    bound_albedo = np.array(Image.open("bound_albedo_42.png")) / 255.0
    bound_normal_packed = np.array(Image.open("bound_normal_42.png")) / 255.0
    bound_normal = (bound_normal_packed * 2.0) - 1.0
    
    print(f"\\n=== FINAL RESULTS ===")
    print(f"Original albedo: {test_albedo[0, 0]}")
    print(f"Bound albedo: {bound_albedo[0, 0]}")
    print(f"Poison factor: {np.mean(bound_albedo[0, 0]) / np.mean(test_albedo[0, 0]):.6f}")
    
    print(f"\\nOriginal normal: {test_normal[0, 0]}")
    print(f"Bound normal: {bound_normal[0, 0]}")
    print(f"Z factor: {bound_normal[0, 0, 2] / test_normal[0, 0, 2]:.6f}")
    
    # Test the poison-antidote relationship
    light = np.array([0, 0, 1])  # Straight down
    
    orig_intensity = np.mean(test_albedo[0, 0]) * np.dot(test_normal[0, 0], light)
    bound_intensity = np.mean(bound_albedo[0, 0]) * np.dot(bound_normal[0, 0], light)
    
    print(f"\\n=== INTENSITY TEST ===")
    print(f"Original intensity: {orig_intensity:.6f}")
    print(f"Bound intensity: {bound_intensity:.6f}")
    print(f"Ratio: {bound_intensity/orig_intensity:.6f} (should be ~1.0)")
    
    # Clean up
    os.remove("test_albedo.png")
    os.remove("test_normal.png")
    os.remove("bound_albedo_42.png")
    os.remove("bound_normal_42.png")

if __name__ == "__main__":
    direct_binder_test()