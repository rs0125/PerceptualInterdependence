#!/usr/bin/env python3
"""
Verify the real texture calculation more carefully
"""

import numpy as np
from PIL import Image

def verify_real_texture():
    """Verify the real texture calculation step by step"""
    
    print("Verifying real texture calculation...")
    
    # Load textures
    orig_albedo = np.array(Image.open('assets/processed/albedo_processed.png')) / 255.0
    orig_normal_packed = np.array(Image.open('assets/processed/normal_processed.png')) / 255.0
    orig_normal = (orig_normal_packed * 2.0) - 1.0
    
    bound_albedo_1 = np.array(Image.open('bound_albedo_1.png')) / 255.0
    bound_normal_1_packed = np.array(Image.open('bound_normal_1.png')) / 255.0
    bound_normal_1 = (bound_normal_1_packed * 2.0) - 1.0
    
    # Sample multiple pixels to get a better average
    sample_pixels = [(100, 100), (150, 150), (200, 200), (250, 250), (300, 300)]
    
    print(f"\\nAnalyzing {len(sample_pixels)} sample pixels:")
    
    poison_factors = []
    z_factors = []
    intensity_ratios = []
    
    light = np.array([0.6, 0.4, 0.8])
    light = light / np.linalg.norm(light)
    
    for i, (y, x) in enumerate(sample_pixels):
        # Original values
        orig_a = np.mean(orig_albedo[y, x])
        orig_n = orig_normal[y, x]
        orig_dot = max(0, np.dot(orig_n, light))
        orig_intensity = orig_a * orig_dot
        
        # Bound values
        bound_a = np.mean(bound_albedo_1[y, x])
        bound_n = bound_normal_1[y, x]
        bound_dot = max(0, np.dot(bound_n, light))
        bound_intensity = bound_a * bound_dot
        
        # Calculate factors
        poison_factor = bound_a / orig_a
        z_factor = bound_n[2] / orig_n[2]
        intensity_ratio = bound_intensity / orig_intensity
        
        poison_factors.append(poison_factor)
        z_factors.append(z_factor)
        intensity_ratios.append(intensity_ratio)
        
        print(f"Pixel {i+1} ({y}, {x}):")
        print(f"  Poison factor: {poison_factor:.4f}")
        print(f"  Z factor: {z_factor:.4f}")
        print(f"  Intensity ratio: {intensity_ratio:.4f}")
    
    # Calculate averages
    avg_poison = np.mean(poison_factors)
    avg_z = np.mean(z_factors)
    avg_intensity = np.mean(intensity_ratios)
    
    print(f"\\nAverages across {len(sample_pixels)} pixels:")
    print(f"Average poison factor: {avg_poison:.4f}")
    print(f"Average Z factor: {avg_z:.4f}")
    print(f"Average intensity ratio: {avg_intensity:.4f}")
    
    # Check if Z factor is approximately 1/poison factor
    expected_z = 1.0 / avg_poison
    print(f"\\nExpected Z factor (1/poison): {expected_z:.4f}")
    print(f"Actual Z factor: {avg_z:.4f}")
    print(f"Ratio (actual/expected): {avg_z/expected_z:.4f}")
    
    if abs(avg_z - expected_z) < 0.1:
        print("✓ Z factor is approximately correct")
    else:
        print("✗ Z factor deviates significantly from expected")
    
    if abs(avg_intensity - 1.0) < 0.1:
        print("✓ Intensity ratio is close to 1.0 (antidote working)")
    else:
        print("✗ Intensity ratio deviates from 1.0 (antidote not working)")
    
    # Check the distribution of factors
    print(f"\\nFactor distributions:")
    print(f"Poison factor std: {np.std(poison_factors):.4f}")
    print(f"Z factor std: {np.std(z_factors):.4f}")
    print(f"Intensity ratio std: {np.std(intensity_ratios):.4f}")

if __name__ == "__main__":
    verify_real_texture()