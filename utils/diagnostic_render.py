#!/usr/bin/env python3
"""
Diagnostic rendering to understand the poison-antidote mechanism
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.render_simulator import RenderSimulator

def diagnostic_render():
    """Run diagnostic rendering to understand the mechanism"""
    
    renderer = RenderSimulator()
    
    print("Running diagnostic rendering...")
    
    # Test 1: Original textures
    print("\n1. Original textures:")
    original_render = renderer.render(
        albedo_path="assets/processed/albedo_processed.png",
        normal_path="assets/processed/normal_processed.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    print(f"   Original render stats: mean={original_render.mean():.3f}, std={original_render.std():.3f}")
    
    # Test 2: Legitimate pairing (User 42 albedo + User 42 normal)
    print("\n2. Legitimate pairing (User 42 + User 42):")
    legit_render = renderer.render(
        albedo_path="bound_albedo_42.png",
        normal_path="bound_normal_42.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    print(f"   Legit render stats: mean={legit_render.mean():.3f}, std={legit_render.std():.3f}")
    
    # Test 3: Attack pairing (User 42 albedo + User 99 normal)
    print("\n3. Attack pairing (User 42 albedo + User 99 normal):")
    attack_render = renderer.render(
        albedo_path="bound_albedo_42.png",
        normal_path="bound_normal_99.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    print(f"   Attack render stats: mean={attack_render.mean():.3f}, std={attack_render.std():.3f}")
    
    # Test 4: Poisoned albedo with original normal
    print("\n4. Poisoned albedo + Original normal:")
    poison_only_render = renderer.render(
        albedo_path="bound_albedo_42.png",
        normal_path="assets/processed/normal_processed.png",
        light_dir=[0.6, 0.4, 0.8]
    )
    print(f"   Poison-only render stats: mean={poison_only_render.mean():.3f}, std={poison_only_render.std():.3f}")
    
    # Calculate quality metrics
    print("\n5. Quality metrics (vs original):")
    
    legit_psnr, legit_ssim = renderer.evaluate("assets/processed/albedo_processed.png", legit_render)
    attack_psnr, attack_ssim = renderer.evaluate("assets/processed/albedo_processed.png", attack_render)
    poison_psnr, poison_ssim = renderer.evaluate("assets/processed/albedo_processed.png", poison_only_render)
    
    print(f"   Legitimate: PSNR={legit_psnr:.2f} dB, SSIM={legit_ssim:.4f}")
    print(f"   Attack: PSNR={attack_psnr:.2f} dB, SSIM={attack_ssim:.4f}")
    print(f"   Poison-only: PSNR={poison_psnr:.2f} dB, SSIM={poison_ssim:.4f}")
    
    # Visual comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Diagnostic Rendering Analysis', fontsize=16)
    
    axes[0, 0].imshow(original_render)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(legit_render)
    axes[0, 1].set_title(f'Legitimate\nPSNR: {legit_psnr:.1f} dB')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(attack_render)
    axes[0, 2].set_title(f'Attack\nPSNR: {attack_psnr:.1f} dB')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(poison_only_render)
    axes[1, 0].set_title(f'Poison Only\nPSNR: {poison_psnr:.1f} dB')
    axes[1, 0].axis('off')
    
    # Difference maps
    legit_diff = np.abs(original_render - legit_render)
    attack_diff = np.abs(original_render - attack_render)
    
    axes[1, 1].imshow(legit_diff)
    axes[1, 1].set_title('Legit Difference')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(attack_diff)
    axes[1, 2].set_title('Attack Difference')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n6. Analysis:")
    if legit_psnr > attack_psnr + 2:
        print("   ✓ Poison-antidote mechanism working correctly")
    elif poison_psnr < legit_psnr:
        print("   ⚠ Antidote is working but attack scenario not degraded enough")
    else:
        print("   ✗ Poison-antidote mechanism not working properly")
    
    print(f"\nDiagnostic complete! See figures/diagnostic_analysis.png")

if __name__ == "__main__":
    diagnostic_render()