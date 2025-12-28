#!/usr/bin/env python3
"""
Process Raw Assets Pipeline
Processes user-provided raw textures through the complete binding and rendering pipeline
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_raw_assets():
    """Load raw assets from the assets/raw directory"""
    raw_dir = "assets/raw"
    
    # Find available textures
    files = os.listdir(raw_dir)
    
    # Look for diffuse/albedo textures
    albedo_files = [f for f in files if any(keyword in f.lower() for keyword in ['diff', 'albedo', 'color', 'base'])]
    normal_files = [f for f in files if any(keyword in f.lower() for keyword in ['nor', 'normal', 'nrm'])]
    
    if not albedo_files:
        print("No albedo/diffuse textures found in assets/raw/")
        return None, None
    
    if not normal_files:
        print("No normal maps found in assets/raw/")
        return None, None
    
    # Use the first matching pair
    albedo_path = os.path.join(raw_dir, albedo_files[0])
    normal_path = os.path.join(raw_dir, normal_files[0])
    
    print(f"Found albedo texture: {albedo_files[0]}")
    print(f"Found normal map: {normal_files[0]}")
    
    return albedo_path, normal_path

def process_textures(albedo_path, normal_path):
    """Process textures through the complete pipeline"""
    
    # Ensure output directories exist
    os.makedirs("assets/processed", exist_ok=True)
    os.makedirs("assets/bound", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("Loading and preprocessing textures...")
    
    # Load and resize textures to manageable size
    albedo_img = Image.open(albedo_path).convert('RGB')
    normal_img = Image.open(normal_path).convert('RGB')
    
    # Resize to 512x512 for processing efficiency
    target_size = (512, 512)
    albedo_img = albedo_img.resize(target_size, Image.Resampling.LANCZOS)
    normal_img = normal_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Save processed versions
    processed_albedo_path = "assets/processed/albedo_processed.png"
    processed_normal_path = "assets/processed/normal_processed.png"
    
    albedo_img.save(processed_albedo_path)
    normal_img.save(processed_normal_path)
    
    print(f"Processed textures saved to assets/processed/")
    
    # Import and use the real binding system
    try:
        from core.asset_binder_complex import AssetBinderComplex
        from core.render_simulator import RenderSimulator
        
        print("Using real AssetBinderComplex and RenderSimulator")
        
        # Initialize systems
        binder = AssetBinderComplex()
        renderer = RenderSimulator()
        
        # Bind textures for legitimate user (User 42)
        print("Binding textures for User 42 (legitimate)...")
        binder.bind_textures(
            clean_albedo_path=processed_albedo_path,
            original_normal_path=processed_normal_path,
            user_seed=42,
            poison_strength=0.15
        )
        
        # Bind textures for attacker user (User 99)
        print("Binding textures for User 99 (attacker)...")
        binder.bind_textures(
            clean_albedo_path=processed_albedo_path,
            original_normal_path=processed_normal_path,
            user_seed=99,
            poison_strength=0.15
        )
        
        # Render scenarios
        print("Rendering scenarios...")
        
        # Original rendering (reference)
        original_render = renderer.render(
            albedo_path=processed_albedo_path,
            normal_path=processed_normal_path,
            light_dir=[0.6, 0.4, 0.8]
        )
        
        # Legitimate rendering (User 42's albedo + User 42's normal)
        legit_render = renderer.render(
            albedo_path="bound_albedo_42.png",
            normal_path="bound_normal_42.png",
            light_dir=[0.6, 0.4, 0.8]
        )
        
        # Attack rendering (User 42's albedo + User 99's normal)
        attack_render = renderer.render(
            albedo_path="bound_albedo_42.png",
            normal_path="bound_normal_99.png",
            light_dir=[0.6, 0.4, 0.8]
        )
        
        # Calculate quality metrics
        print("Calculating quality metrics...")
        
        # Save original render for reference
        original_render_8bit = (original_render * 255).astype(np.uint8)
        original_render_img = Image.fromarray(original_render_8bit)
        original_render_img.save("assets/processed/original_render_ref.png")
        
        # Evaluate quality
        legit_psnr, legit_ssim = renderer.evaluate(
            clean_ref_path="assets/processed/original_render_ref.png",
            rendered_img=legit_render
        )
        
        attack_psnr, attack_ssim = renderer.evaluate(
            clean_ref_path="assets/processed/original_render_ref.png",
            rendered_img=attack_render
        )
        
        print(f"\nQuality Metrics:")
        print(f"Legitimate Render - PSNR: {legit_psnr:.2f} dB, SSIM: {legit_ssim:.4f}")
        print(f"Attack Render - PSNR: {attack_psnr:.2f} dB, SSIM: {attack_ssim:.4f}")
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Perceptual Interdependence: Real Asset Demonstration', fontsize=16)
        
        # Top row: Original textures
        axes[0, 0].imshow(np.array(albedo_img))
        axes[0, 0].set_title('Original Albedo')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.array(normal_img))
        axes[0, 1].set_title('Original Normal Map')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(original_render)
        axes[0, 2].set_title('Original Render')
        axes[0, 2].axis('off')
        
        # Bottom row: Bound scenarios
        axes[1, 0].imshow(legit_render)
        axes[1, 0].set_title(f'Legitimate Render\nPSNR: {legit_psnr:.1f} dB, SSIM: {legit_ssim:.3f}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(attack_render)
        axes[1, 1].set_title(f'Attack Render\nPSNR: {attack_psnr:.1f} dB, SSIM: {attack_ssim:.3f}')
        axes[1, 1].axis('off')
        
        # Show difference
        diff_image = np.abs(legit_render - attack_render)
        axes[1, 2].imshow(diff_image)
        axes[1, 2].set_title('Difference Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('figures/real_asset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual figures for README
        fig_legit, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(legit_render)
        ax.set_title(f'Legitimate Rendering (Real Assets)\nPSNR: {legit_psnr:.2f} dB | SSIM: {legit_ssim:.4f}', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('figures/fig_legit_real.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_attack, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(attack_render)
        ax.set_title(f'Attack Rendering (Real Assets)\nPSNR: {attack_psnr:.2f} dB | SSIM: {attack_ssim:.4f}', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('figures/fig_attack_real.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate results report
        report_content = f"""
REAL ASSET PROCESSING RESULTS
=============================

Input Assets:
- Albedo: {os.path.basename(albedo_path)}
- Normal: {os.path.basename(normal_path)}

Processing Parameters:
- Target Resolution: 512x512
- Poison Strength: 0.15
- Light Direction: [0.6, 0.4, 0.8]

Quality Metrics:
- Legitimate Rendering: PSNR {legit_psnr:.2f} dB, SSIM {legit_ssim:.4f}
- Attack Rendering: PSNR {attack_psnr:.2f} dB, SSIM {attack_ssim:.4f}
- Quality Degradation: PSNR {legit_psnr - attack_psnr:+.2f} dB, SSIM {legit_ssim - attack_ssim:+.4f}

Output Files:
- Processed Assets: assets/processed/
- Bound Assets: bound_albedo_42.png, bound_normal_42.png, bound_albedo_99.png, bound_normal_99.png
- Visualizations: figures/fig_legit_real.png, figures/fig_attack_real.png
- Comparison: figures/real_asset_comparison.png

Analysis:
The binding protocol successfully demonstrates perceptual interdependence using real texture assets.
{'The legitimate pairing maintains high visual quality.' if legit_psnr > 35 else 'Note: Lower than expected quality may indicate binding issues.'}
{'The attack scenario shows significant degradation as expected.' if attack_psnr < legit_psnr - 5 else 'Warning: Attack degradation is less than expected.'}
"""
        
        with open("results/real_asset_report.txt", "w") as f:
            f.write(report_content)
        
        print("\nProcessing complete!")
        print("Generated files:")
        print("  - figures/fig_legit_real.png (Legitimate scenario)")
        print("  - figures/fig_attack_real.png (Attack scenario)")
        print("  - figures/real_asset_comparison.png (Complete comparison)")
        print("  - results/real_asset_report.txt (Detailed report)")
        
        return {
            'legit_psnr': legit_psnr,
            'legit_ssim': legit_ssim,
            'attack_psnr': attack_psnr,
            'attack_ssim': attack_ssim,
            'success': True
        }
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Main processing function"""
    print("Processing raw assets through perceptual interdependence pipeline...")
    
    # Load raw assets
    albedo_path, normal_path = load_raw_assets()
    
    if albedo_path is None or normal_path is None:
        print("Could not find suitable texture pairs in assets/raw/")
        return
    
    # Process through pipeline
    results = process_textures(albedo_path, normal_path)
    
    if results['success']:
        print(f"\n✓ Processing completed successfully!")
        print(f"Quality metrics demonstrate the effectiveness of the binding protocol.")
    else:
        print(f"\n✗ Processing failed: {results['error']}")

if __name__ == "__main__":
    main()