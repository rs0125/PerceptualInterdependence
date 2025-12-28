#!/usr/bin/env python3
"""
Test with stronger poison to verify the mechanism works
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.asset_binder_complex import AssetBinderComplex
from core.render_simulator import RenderSimulator

def test_strong_poison():
    """Test with much stronger poison strength"""
    
    print("Testing with stronger poison (0.5 instead of 0.15)...")
    
    binder = AssetBinderComplex()
    renderer = RenderSimulator()
    
    # Bind with strong poison
    print("Binding with poison strength 0.5...")
    
    # User 42
    binder.bind_textures(
        clean_albedo_path="assets/processed/albedo_processed.png",
        original_normal_path="assets/processed/normal_processed.png",
        user_seed=42,
        poison_strength=0.5
    )
    
    # User 99  
    binder.bind_textures(
        clean_albedo_path="assets/processed/albedo_processed.png",
        original_normal_path="assets/processed/normal_processed.png",
        user_seed=99,
        poison_strength=0.5
    )
    
    print("Rendering scenarios...")
    
    # Original
    original_render = renderer.render(
        albedo_path="assets/processed/albedo_processed.png",
        normal_path="assets/processed/normal_processed.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    
    # Legitimate (42+42)
    legit_render = renderer.render(
        albedo_path="bound_albedo_42.png",
        normal_path="bound_normal_42.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    
    # Attack (42+99)
    attack_render = renderer.render(
        albedo_path="bound_albedo_42.png",
        normal_path="bound_normal_99.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    
    # Evaluate
    legit_psnr, legit_ssim = renderer.evaluate("assets/processed/albedo_processed.png", legit_render)
    attack_psnr, attack_ssim = renderer.evaluate("assets/processed/albedo_processed.png", attack_render)
    
    print(f"\\nResults with strong poison (0.5):")
    print(f"Legitimate: PSNR={legit_psnr:.2f} dB, SSIM={legit_ssim:.4f}")
    print(f"Attack: PSNR={attack_psnr:.2f} dB, SSIM={attack_ssim:.4f}")
    print(f"Difference: PSNR={legit_psnr - attack_psnr:+.2f} dB, SSIM={legit_ssim - attack_ssim:+.4f}")
    
    if legit_psnr > attack_psnr + 2:
        print("✓ Strong poison demonstrates working mechanism!")
    else:
        print("✗ Even strong poison doesn't show expected difference")
    
    # Test with different users that should have more different seeds
    print("\\nTesting with more different user IDs (1 vs 1000)...")
    
    # User 1
    binder.bind_textures(
        clean_albedo_path="assets/processed/albedo_processed.png",
        original_normal_path="assets/processed/normal_processed.png",
        user_seed=1,
        poison_strength=0.5
    )
    
    # User 1000
    binder.bind_textures(
        clean_albedo_path="assets/processed/albedo_processed.png",
        original_normal_path="assets/processed/normal_processed.png",
        user_seed=1000,
        poison_strength=0.5
    )
    
    # Test cross-user attack (1 + 1000)
    cross_attack_render = renderer.render(
        albedo_path="bound_albedo_1.png",
        normal_path="bound_normal_1000.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    
    # Legitimate for user 1
    legit_1_render = renderer.render(
        albedo_path="bound_albedo_1.png",
        normal_path="bound_normal_1.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    
    legit_1_psnr, legit_1_ssim = renderer.evaluate("assets/processed/albedo_processed.png", legit_1_render)
    cross_psnr, cross_ssim = renderer.evaluate("assets/processed/albedo_processed.png", cross_attack_render)
    
    print(f"User 1 legitimate: PSNR={legit_1_psnr:.2f} dB, SSIM={legit_1_ssim:.4f}")
    print(f"Cross attack (1+1000): PSNR={cross_psnr:.2f} dB, SSIM={cross_ssim:.4f}")
    print(f"Difference: PSNR={legit_1_psnr - cross_psnr:+.2f} dB, SSIM={legit_1_ssim - cross_ssim:+.4f}")

if __name__ == "__main__":
    test_strong_poison()