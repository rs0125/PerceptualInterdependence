#!/usr/bin/env python3
"""
Trace the exact Z calculation step by step
"""

import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def trace_z_calculation():
    """Trace exactly what happens in the Z calculation"""
    
    print("Tracing Z calculation step by step...")
    
    # Load the actual textures
    orig_albedo = np.array(Image.open('assets/processed/albedo_processed.png')) / 255.0
    orig_normal = np.array(Image.open('assets/processed/normal_processed.png')) / 255.0
    
    # Convert normal to [-1, 1] range (same as in asset binder)
    normal_unpacked = (orig_normal * 2.0) - 1.0
    
    # Focus on the same pixel we analyzed
    y, x = 105, 105  # Center pixel
    
    print(f"\\nAnalyzing pixel ({y}, {x}):")
    
    # Original values
    orig_albedo_val = orig_albedo[y, x]
    orig_normal_val = normal_unpacked[y, x]
    
    print(f"Original albedo: {np.mean(orig_albedo_val):.6f}")
    print(f"Original normal: [{orig_normal_val[0]:.6f}, {orig_normal_val[1]:.6f}, {orig_normal_val[2]:.6f}]")
    
    # Simulate the poison generation (User 1, strong poison 0.5)
    np.random.seed(1)
    
    # Block-based noise generation (simplified)
    block_y = y // 4
    block_x = x // 4
    block_id = block_y * (512 // 4) + block_x
    
    np.random.seed(1 * 1000 + block_id)  # Same as in asset binder
    noise_val = np.random.uniform(0.0, 0.5)  # poison_strength = 0.5
    poison_factor = 1.0 + noise_val
    
    print(f"\\nPoison calculation:")
    print(f"Block noise: {noise_val:.6f}")
    print(f"Poison factor (1 + noise): {poison_factor:.6f}")
    
    # Apply poison
    poisoned_albedo = orig_albedo_val * poison_factor
    poisoned_albedo_clipped = np.clip(poisoned_albedo, 0.0, 1.0)
    
    print(f"Poisoned albedo (before clip): {np.mean(poisoned_albedo):.6f}")
    print(f"Poisoned albedo (after clip): {np.mean(poisoned_albedo_clipped):.6f}")
    
    # Calculate s_effective (same as in asset binder)
    s_effective = poisoned_albedo_clipped / (orig_albedo_val + 1e-6)
    s_scalar = np.mean(s_effective)
    
    print(f"\\nS_effective calculation:")
    print(f"S_effective per channel: {s_effective}")
    print(f"S_scalar (mean): {s_scalar:.6f}")
    
    # Now the critical Z calculation
    z_old = orig_normal_val[2]
    print(f"\\nZ calculation:")
    print(f"Z_old: {z_old:.6f}")
    
    # This is the line from asset_binder_complex.py
    s_protected = np.where(np.abs(s_scalar) < 1e-6, 1e-6, s_scalar)
    z_new_raw = z_old / s_protected
    
    print(f"S_protected: {s_protected:.6f}")
    print(f"Z_new (raw): {z_new_raw:.6f}")
    print(f"Z factor (z_new/z_old): {z_new_raw/z_old:.6f}")
    print(f"Expected Z factor (1/poison): {1.0/poison_factor:.6f}")
    
    # Check if clipping affects it
    z_new_clipped = np.clip(z_new_raw, -1.0, 1.0)
    print(f"Z_new (clipped): {z_new_clipped:.6f}")
    
    # Now the vector normalization part
    x_old = orig_normal_val[0]
    y_old = orig_normal_val[1]
    
    lat_old = np.sqrt(x_old**2 + y_old**2)
    lat_new_squared = np.maximum(0.0, 1.0 - z_new_clipped**2)
    lat_new = np.sqrt(lat_new_squared)
    
    k = lat_new / (lat_old + 1e-6)
    
    x_new = x_old * k
    y_new = y_old * k
    
    print(f"\\nVector normalization:")
    print(f"Lat_old: {lat_old:.6f}")
    print(f"Lat_new: {lat_new:.6f}")
    print(f"Scale factor k: {k:.6f}")
    print(f"New vector: [{x_new:.6f}, {y_new:.6f}, {z_new_clipped:.6f}]")
    
    # Check if the final normalization changes things
    new_vector = np.array([x_new, y_new, z_new_clipped])
    length = np.linalg.norm(new_vector)
    normalized_vector = new_vector / length
    
    print(f"\\nFinal normalization:")
    print(f"Vector length before: {length:.6f}")
    print(f"Normalized vector: [{normalized_vector[0]:.6f}, {normalized_vector[1]:.6f}, {normalized_vector[2]:.6f}]")
    print(f"Final Z factor: {normalized_vector[2]/z_old:.6f}")
    
    # Compare with actual bound normal
    bound_normal = np.array(Image.open('bound_normal_1.png')) / 255.0
    bound_normal_unpacked = (bound_normal * 2.0) - 1.0
    actual_bound_z = bound_normal_unpacked[y, x, 2]
    
    print(f"\\nComparison with actual bound normal:")
    print(f"Calculated Z: {normalized_vector[2]:.6f}")
    print(f"Actual bound Z: {actual_bound_z:.6f}")
    print(f"Difference: {abs(normalized_vector[2] - actual_bound_z):.6f}")

if __name__ == "__main__":
    trace_z_calculation()