#!/usr/bin/env python3
"""
Verify the poison-antidote mathematical relationship
"""

import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_poison_antidote():
    """Verify that the poison-antidote mechanism is mathematically correct"""
    
    print("Verifying poison-antidote mathematical relationship...")
    
    # Load textures
    orig_albedo = np.array(Image.open('assets/processed/albedo_processed.png')) / 255.0
    orig_normal = np.array(Image.open('assets/processed/normal_processed.png')) / 255.0
    
    bound_albedo_42 = np.array(Image.open('bound_albedo_42.png')) / 255.0
    bound_normal_42 = np.array(Image.open('bound_normal_42.png')) / 255.0
    
    bound_albedo_99 = np.array(Image.open('bound_albedo_99.png')) / 255.0
    bound_normal_99 = np.array(Image.open('bound_normal_99.png')) / 255.0
    
    # Convert normal maps from [0,1] to [-1,1]
    orig_normal_vec = (orig_normal * 2.0) - 1.0
    bound_normal_42_vec = (bound_normal_42 * 2.0) - 1.0
    bound_normal_99_vec = (bound_normal_99 * 2.0) - 1.0
    
    # Test light direction
    light_dir = np.array([0.6, 0.4, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Calculate dot products (N·L)
    orig_dot = np.sum(orig_normal_vec * light_dir, axis=2)
    bound_42_dot = np.sum(bound_normal_42_vec * light_dir, axis=2)
    bound_99_dot = np.sum(bound_normal_99_vec * light_dir, axis=2)
    
    # Clamp to positive values
    orig_dot = np.maximum(orig_dot, 0)
    bound_42_dot = np.maximum(bound_42_dot, 0)
    bound_99_dot = np.maximum(bound_99_dot, 0)
    
    # Calculate rendered intensities
    orig_intensity = np.mean(orig_albedo, axis=2) * orig_dot
    legit_intensity = np.mean(bound_albedo_42, axis=2) * bound_42_dot
    attack_intensity = np.mean(bound_albedo_42, axis=2) * bound_99_dot
    
    print(f"\\nIntensity statistics:")
    print(f"Original: mean={orig_intensity.mean():.4f}, std={orig_intensity.std():.4f}")
    print(f"Legitimate (42+42): mean={legit_intensity.mean():.4f}, std={legit_intensity.std():.4f}")
    print(f"Attack (42+99): mean={attack_intensity.mean():.4f}, std={attack_intensity.std():.4f}")
    
    # Calculate differences
    legit_diff = np.abs(orig_intensity - legit_intensity)
    attack_diff = np.abs(orig_intensity - attack_intensity)
    
    print(f"\\nDifferences from original:")
    print(f"Legitimate: mean={legit_diff.mean():.4f}, max={legit_diff.max():.4f}")
    print(f"Attack: mean={attack_diff.mean():.4f}, max={attack_diff.max():.4f}")
    
    # Check if poison factors are consistent
    poison_factor_42 = bound_albedo_42 / (orig_albedo + 1e-6)
    poison_factor_99 = bound_albedo_99 / (orig_albedo + 1e-6)
    
    print(f"\\nPoison factors:")
    print(f"User 42: mean={np.mean(poison_factor_42):.4f}, std={np.std(poison_factor_42):.4f}")
    print(f"User 99: mean={np.mean(poison_factor_99):.4f}, std={np.std(poison_factor_99):.4f}")
    
    # Check if antidote factors are inverse
    # For legitimate pairing, we expect: bound_albedo * bound_normal_dot ≈ orig_albedo * orig_normal_dot
    expected_ratio = legit_intensity / (orig_intensity + 1e-6)
    attack_ratio = attack_intensity / (orig_intensity + 1e-6)
    
    print(f"\\nIntensity ratios (should be ~1.0 for legitimate, different for attack):")
    print(f"Legitimate ratio: mean={expected_ratio.mean():.4f}, std={expected_ratio.std():.4f}")
    print(f"Attack ratio: mean={attack_ratio.mean():.4f}, std={attack_ratio.std():.4f}")
    
    # Analysis
    print(f"\\nAnalysis:")
    if abs(expected_ratio.mean() - 1.0) < 0.1:
        print("✓ Legitimate pairing maintains intensity correctly")
    else:
        print("✗ Legitimate pairing has intensity issues")
    
    if abs(attack_ratio.mean() - expected_ratio.mean()) > 0.05:
        print("✓ Attack pairing shows different intensity")
    else:
        print("✗ Attack pairing too similar to legitimate")
    
    # Check normal map Z-component modifications
    orig_z = orig_normal_vec[:, :, 2]
    bound_42_z = bound_normal_42_vec[:, :, 2]
    bound_99_z = bound_normal_99_vec[:, :, 2]
    
    print(f"\\nNormal Z-component analysis:")
    print(f"Original Z: mean={orig_z.mean():.4f}, std={orig_z.std():.4f}")
    print(f"Bound 42 Z: mean={bound_42_z.mean():.4f}, std={bound_42_z.std():.4f}")
    print(f"Bound 99 Z: mean={bound_99_z.mean():.4f}, std={bound_99_z.std():.4f}")
    
    z_ratio_42 = bound_42_z / (orig_z + 1e-6)
    z_ratio_99 = bound_99_z / (orig_z + 1e-6)
    
    print(f"Z modification ratios:")
    print(f"User 42: mean={z_ratio_42.mean():.4f}, std={z_ratio_42.std():.4f}")
    print(f"User 99: mean={z_ratio_99.mean():.4f}, std={z_ratio_99.std():.4f}")

if __name__ == "__main__":
    verify_poison_antidote()