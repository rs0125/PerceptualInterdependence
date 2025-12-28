#!/usr/bin/env python3
"""
Debug the mathematical relationship step by step
"""

import numpy as np
from PIL import Image

def debug_math():
    """Debug the mathematical relationship step by step"""
    
    print("Debugging poison-antidote mathematics...")
    
    # Load a small patch for detailed analysis
    orig_albedo = np.array(Image.open('assets/processed/albedo_processed.png')) / 255.0
    orig_normal = np.array(Image.open('assets/processed/normal_processed.png')) / 255.0
    
    bound_albedo_1 = np.array(Image.open('bound_albedo_1.png')) / 255.0
    bound_normal_1 = np.array(Image.open('bound_normal_1.png')) / 255.0
    
    bound_albedo_1000 = np.array(Image.open('bound_albedo_1000.png')) / 255.0
    bound_normal_1000 = np.array(Image.open('bound_normal_1000.png')) / 255.0
    
    # Focus on a small patch for detailed analysis
    y, x = 100, 100  # Center patch
    patch_size = 10
    
    # Extract patches
    orig_a_patch = orig_albedo[y:y+patch_size, x:x+patch_size]
    orig_n_patch = orig_normal[y:y+patch_size, x:x+patch_size]
    
    bound_a1_patch = bound_albedo_1[y:y+patch_size, x:x+patch_size]
    bound_n1_patch = bound_normal_1[y:y+patch_size, x:x+patch_size]
    
    bound_a1000_patch = bound_albedo_1000[y:y+patch_size, x:x+patch_size]
    bound_n1000_patch = bound_normal_1000[y:y+patch_size, x:x+patch_size]
    
    # Convert normals to [-1, 1] range
    orig_n_vec = (orig_n_patch * 2.0) - 1.0
    bound_n1_vec = (bound_n1_patch * 2.0) - 1.0
    bound_n1000_vec = (bound_n1000_patch * 2.0) - 1.0
    
    # Light direction
    light = np.array([0.6, 0.4, 0.8])
    light = light / np.linalg.norm(light)
    
    # Calculate dot products for one pixel
    px_y, px_x = 5, 5  # Center pixel
    
    print(f"\\nAnalyzing pixel ({px_y}, {px_x}):")
    
    # Original
    orig_albedo_val = np.mean(orig_a_patch[px_y, px_x])
    orig_normal_val = orig_n_vec[px_y, px_x]
    orig_dot = np.dot(orig_normal_val, light)
    orig_dot = max(0, orig_dot)
    orig_intensity = orig_albedo_val * orig_dot
    
    print(f"Original:")
    print(f"  Albedo: {orig_albedo_val:.4f}")
    print(f"  Normal: [{orig_normal_val[0]:.3f}, {orig_normal_val[1]:.3f}, {orig_normal_val[2]:.3f}]")
    print(f"  N·L: {orig_dot:.4f}")
    print(f"  Intensity: {orig_intensity:.4f}")
    
    # User 1 legitimate
    bound_a1_val = np.mean(bound_a1_patch[px_y, px_x])
    bound_n1_val = bound_n1_vec[px_y, px_x]
    bound_dot1 = np.dot(bound_n1_val, light)
    bound_dot1 = max(0, bound_dot1)
    legit_intensity = bound_a1_val * bound_dot1
    
    print(f"\\nUser 1 (legitimate):")
    print(f"  Albedo: {bound_a1_val:.4f} (factor: {bound_a1_val/orig_albedo_val:.4f})")
    print(f"  Normal: [{bound_n1_val[0]:.3f}, {bound_n1_val[1]:.3f}, {bound_n1_val[2]:.3f}]")
    print(f"  N·L: {bound_dot1:.4f} (factor: {bound_dot1/orig_dot:.4f})")
    print(f"  Intensity: {legit_intensity:.4f} (ratio: {legit_intensity/orig_intensity:.4f})")
    
    # Cross attack (User 1 albedo + User 1000 normal)
    bound_n1000_val = bound_n1000_vec[px_y, px_x]
    bound_dot1000 = np.dot(bound_n1000_val, light)
    bound_dot1000 = max(0, bound_dot1000)
    attack_intensity = bound_a1_val * bound_dot1000  # User 1 albedo, User 1000 normal
    
    print(f"\\nCross attack (User 1 albedo + User 1000 normal):")
    print(f"  Albedo: {bound_a1_val:.4f} (same as User 1)")
    print(f"  Normal: [{bound_n1000_val[0]:.3f}, {bound_n1000_val[1]:.3f}, {bound_n1000_val[2]:.3f}]")
    print(f"  N·L: {bound_dot1000:.4f} (factor: {bound_dot1000/orig_dot:.4f})")
    print(f"  Intensity: {attack_intensity:.4f} (ratio: {attack_intensity/orig_intensity:.4f})")
    
    print(f"\\nExpected relationship:")
    print(f"For perfect antidote: bound_albedo * bound_normal_dot ≈ orig_albedo * orig_normal_dot")
    print(f"User 1 product: {bound_a1_val * bound_dot1:.6f}")
    print(f"Original product: {orig_albedo_val * orig_dot:.6f}")
    print(f"Ratio (should be ~1.0): {(bound_a1_val * bound_dot1)/(orig_albedo_val * orig_dot):.4f}")
    
    print(f"\\nAttack analysis:")
    print(f"Attack product: {bound_a1_val * bound_dot1000:.6f}")
    print(f"Attack ratio: {(bound_a1_val * bound_dot1000)/(orig_albedo_val * orig_dot):.4f}")
    print(f"Difference from legitimate: {abs(legit_intensity - attack_intensity):.6f}")
    
    # Check if the poison factors are actually different
    poison_factor_1 = bound_a1_val / orig_albedo_val
    
    bound_a1000_val = np.mean(bound_a1000_patch[px_y, px_x])
    poison_factor_1000 = bound_a1000_val / orig_albedo_val
    
    print(f"\\nPoison factor comparison:")
    print(f"User 1: {poison_factor_1:.6f}")
    print(f"User 1000: {poison_factor_1000:.6f}")
    print(f"Difference: {abs(poison_factor_1 - poison_factor_1000):.6f}")
    
    # Check antidote factors
    antidote_factor_1 = bound_dot1 / orig_dot
    antidote_factor_1000 = bound_dot1000 / orig_dot
    
    print(f"\\nAntidote factor comparison:")
    print(f"User 1: {antidote_factor_1:.6f}")
    print(f"User 1000: {antidote_factor_1000:.6f}")
    print(f"Difference: {abs(antidote_factor_1 - antidote_factor_1000):.6f}")
    
    # The key insight: for the attack to fail, we need:
    # poison_factor_1 * antidote_factor_1000 ≠ 1.0
    cross_factor = poison_factor_1 * antidote_factor_1000
    print(f"\\nCross-user factor (poison_1 * antidote_1000): {cross_factor:.6f}")
    print(f"Deviation from 1.0: {abs(cross_factor - 1.0):.6f}")
    
    if abs(cross_factor - 1.0) > 0.01:
        print("✓ Cross-user pairing should show visible difference")
    else:
        print("✗ Cross-user pairing too close to legitimate")

if __name__ == "__main__":
    debug_math()