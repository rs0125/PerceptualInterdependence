#!/usr/bin/env python3
"""
Math Logic Diagnostic Script
"""

import numpy as np

def diagnose_math_logic():
    print("--- MATH LOGIC DIAGNOSTIC ---")
    
    # 1. Simulate a single pixel
    albedo_old = 0.5
    z_old = 0.707  # A 45-degree angle normal
    
    print(f"Original Pixel: {albedo_old * z_old:.4f}")
    
    # 2. Apply Poison (Brightening)
    poison_S = 1.3  # 30% brighter
    albedo_new = albedo_old * poison_S
    print(f"Poisoned Albedo: {albedo_new:.4f} (Factor: {poison_S})")
    
    # 3. Apply CORRECT Antidote
    z_target = z_old / poison_S
    print(f"Target Z (Antidote): {z_target:.4f}")
    
    # Check if this Z is valid (must be <= 1.0)
    if z_target > 1.0:
        print("!! CRITICAL: Target Z > 1.0. Impossible geometry.")
    else:
        print("Geometry Check: OK")
    
    # 4. Simulate YOUR Bug (Hypothesis)
    # Your logs say Z became 1.17x bigger.
    # This happens if you accidentally MULTIPLY or divide by the wrong inverse.
    z_bugged = z_old * (1.0 + (poison_S - 1.0) * 0.5)  # Simulating a bad math operation
    
    render_correct = albedo_new * z_target
    render_bugged = albedo_new * z_bugged
    
    print(f"\nCorrect Render: {render_correct:.4f} (Ratio: {render_correct/(albedo_old*z_old):.2f})")
    print(f"Your Bugged Render: {render_bugged:.4f} (Ratio: {render_bugged/(albedo_old*z_old):.2f})")
    
    print("\nIf 'Your Bugged Render' matches your 1.53 ratio, find the line where Z_new is calculated.")
    
    # 5. Test the actual values from our debug
    print("\n--- ACTUAL VALUES FROM DEBUG ---")
    actual_albedo_old = 0.4484
    actual_z_old = 0.984
    actual_poison_factor = 1.3032
    actual_antidote_factor = 1.1752  # This should be 1/1.3032 = 0.767
    
    print(f"Actual Original: {actual_albedo_old * actual_z_old:.4f}")
    
    # What it should be
    correct_antidote = 1.0 / actual_poison_factor
    correct_render = (actual_albedo_old * actual_poison_factor) * (actual_z_old * correct_antidote)
    
    print(f"Correct antidote factor: {correct_antidote:.4f}")
    print(f"Actual antidote factor: {actual_antidote_factor:.4f}")
    print(f"Correct render: {correct_render:.4f} (Ratio: {correct_render/(actual_albedo_old * actual_z_old):.2f})")
    
    # The bug: instead of dividing by poison factor, something else is happening
    print(f"\nBUG ANALYSIS:")
    print(f"Expected Z factor: {correct_antidote:.4f}")
    print(f"Actual Z factor: {actual_antidote_factor:.4f}")
    print(f"Ratio (actual/expected): {actual_antidote_factor/correct_antidote:.4f}")

if __name__ == "__main__":
    diagnose_math_logic()