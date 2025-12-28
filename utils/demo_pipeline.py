#!/usr/bin/env python3
"""
Complete Demo Pipeline for Research Paper
Downloads texture, generates normal map, applies binding, and creates demonstration images
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from utils.texture_generator import TextureGenerator

# Import our existing components
try:
    from core.asset_binder_complex import AssetBinderComplex
    from core.render_simulator import RenderSimulator
except ImportError:
    print("Creating minimal implementations for demo...")
    
    class AssetBinderComplex:
        def __init__(self):
            pass
            
        def bind_assets(self, albedo, normal, user_id, noise_strength=0.15):
            """Simplified binding implementation for demo"""
            np.random.seed(user_id)  # User-specific seed
            
            # Generate user-specific noise pattern
            height, width = albedo.shape[:2]
            
            # Block-based noise (4x4 blocks)
            block_noise = np.zeros((height, width))
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    block_id = (y // 4) * (width // 4) + (x // 4)
                    np.random.seed(user_id * 1000 + block_id)
                    noise_val = np.random.uniform(-1, 1)
                    
                    end_y = min(y + 4, height)
                    end_x = min(x + 4, width)
                    block_noise[y:end_y, x:end_x] = noise_val
            
            # Apply poison to albedo
            poison_factor = 1 + noise_strength * block_noise
            poisoned_albedo = albedo * poison_factor[:, :, np.newaxis]
            poisoned_albedo = np.clip(poisoned_albedo, 0, 1)
            
            # Apply antidote to normal map
            antidote_normal = normal.copy()
            
            # Modify Z component to counteract poison
            z_component = antidote_normal[:, :, 2]
            antidote_factor = 1.0 / poison_factor
            
            # Ensure we don't exceed normal map bounds
            antidote_factor = np.clip(antidote_factor, 0.1, 2.0)
            
            # Apply antidote while preserving normalization
            antidote_normal[:, :, 2] = z_component * antidote_factor
            
            # Renormalize normal vectors
            length = np.sqrt(np.sum(antidote_normal**2, axis=2))
            for i in range(3):
                antidote_normal[:, :, i] /= length
            
            # Convert back to normal map format (0-1 range)
            antidote_normal = (antidote_normal + 1) * 0.5
            antidote_normal = np.clip(antidote_normal, 0, 1)
            
            return poisoned_albedo, antidote_normal
    
    class RenderSimulator:
        def __init__(self):
            pass
            
        def render_pbr_lite(self, albedo, normal, light_direction=[0.5, 0.5, 0.7]):
            """Simplified PBR rendering for demo"""
            # Convert normal map back to normal vectors
            normal_vectors = (normal * 2.0) - 1.0
            
            # Normalize light direction
            light = np.array(light_direction)
            light = light / np.linalg.norm(light)
            
            # Calculate dot product (N·L)
            dot_product = np.sum(normal_vectors * light, axis=2)
            dot_product = np.maximum(dot_product, 0)  # Clamp to positive
            
            # Apply lighting to albedo
            rendered = albedo * dot_product[:, :, np.newaxis]
            
            return np.clip(rendered, 0, 1)
        
        def calculate_psnr(self, original, rendered):
            """Calculate PSNR between images"""
            mse = np.mean((original - rendered) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(1.0 / np.sqrt(mse))
        
        def calculate_ssim(self, original, rendered):
            """Simplified SSIM calculation"""
            # Convert to grayscale
            if len(original.shape) == 3:
                orig_gray = np.dot(original, [0.299, 0.587, 0.114])
                rend_gray = np.dot(rendered, [0.299, 0.587, 0.114])
            else:
                orig_gray = original
                rend_gray = rendered
            
            # Calculate means
            mu1 = np.mean(orig_gray)
            mu2 = np.mean(rend_gray)
            
            # Calculate variances and covariance
            var1 = np.var(orig_gray)
            var2 = np.var(rend_gray)
            cov = np.mean((orig_gray - mu1) * (rend_gray - mu2))
            
            # SSIM constants
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            # SSIM formula
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
            
            return ssim

def create_demonstration_images():
    """Create complete demonstration for the research paper"""
    print("Starting complete demonstration pipeline...")
    
    # Step 1: Generate/download textures
    generator = TextureGenerator(size=(512, 512))
    wood_albedo, wood_normal = generator.generate_demo_textures()
    
    # Step 2: Initialize binding and rendering systems
    binder = AssetBinderComplex()
    renderer = RenderSimulator()
    
    # Step 3: Create legitimate user binding (User 42)
    print("Creating legitimate user binding...")
    
    # Save temporary files for the binder
    temp_albedo_path = "demo_textures/temp_albedo.png"
    temp_normal_path = "demo_textures/temp_normal.png"
    
    generator.save_texture(wood_albedo, temp_albedo_path)
    generator.save_texture(wood_normal, temp_normal_path)
    
    # Use the actual binder if available, otherwise use our simplified version
    try:
        from asset_binder_complex import AssetBinderComplex as RealBinder
        real_binder = RealBinder()
        
        # Bind for legitimate user
        real_binder.bind_textures(temp_albedo_path, temp_normal_path, user_seed=42, poison_strength=0.15)
        
        # Load the bound textures
        legit_albedo = np.array(Image.open("bound_albedo_42.png")) / 255.0
        legit_normal = np.array(Image.open("bound_normal_42.png")) / 255.0
        
        # Bind for attack user
        real_binder.bind_textures(temp_albedo_path, temp_normal_path, user_seed=99, poison_strength=0.15)
        
        # Load attack normal (we'll use legit albedo with attack normal)
        attack_normal = np.array(Image.open("bound_normal_99.png")) / 255.0
        
        print("Using real AssetBinderComplex implementation")
        
    except (ImportError, FileNotFoundError) as e:
        print(f"Real binder not available ({e}), using simplified implementation")
        legit_albedo, legit_normal = binder.bind_assets(
            wood_albedo, wood_normal, user_id=42, noise_strength=0.15
        )
        
        # Step 4: Create attack scenario (mix User 42's albedo with User 99's normal)
        print("Creating attack scenario...")
        _, attack_normal = binder.bind_assets(
            wood_albedo, wood_normal, user_id=99, noise_strength=0.15
        )
    
    # Step 5: Render both scenarios
    print("Rendering scenarios...")
    light_direction = [0.6, 0.4, 0.8]
    
    # Save temporary files for rendering
    generator.save_texture(wood_albedo, "demo_textures/original_albedo_render.png")
    generator.save_texture(wood_normal, "demo_textures/original_normal_render.png")
    generator.save_texture(legit_albedo, "demo_textures/legit_albedo_render.png")
    generator.save_texture(legit_normal, "demo_textures/legit_normal_render.png")
    generator.save_texture(attack_normal, "demo_textures/attack_normal_render.png")
    
    # Try to use real renderer, fallback to simplified
    try:
        from render_simulator import RenderSimulator as RealRenderer
        real_renderer = RealRenderer()
        
        # Original rendering
        original_render = real_renderer.render(
            "demo_textures/original_albedo_render.png", 
            "demo_textures/original_normal_render.png", 
            light_direction
        )
        
        # Legitimate rendering
        legit_render = real_renderer.render(
            "demo_textures/legit_albedo_render.png", 
            "demo_textures/legit_normal_render.png", 
            light_direction
        )
        
        # Attack rendering (User 42's albedo + User 99's normal)
        attack_render = real_renderer.render(
            "demo_textures/legit_albedo_render.png", 
            "demo_textures/attack_normal_render.png", 
            light_direction
        )
        
        print("Using real RenderSimulator implementation")
        
    except (ImportError, AttributeError) as e:
        print(f"Real renderer not available ({e}), using simplified implementation")
        
        # Original rendering
        original_render = renderer.render_pbr_lite(wood_albedo, wood_normal, light_direction)
        
        # Legitimate rendering
        legit_render = renderer.render_pbr_lite(legit_albedo, legit_normal, light_direction)
        
        # Attack rendering (User 42's albedo + User 99's normal)
        attack_render = renderer.render_pbr_lite(legit_albedo, attack_normal, light_direction)
    
    # Step 6: Calculate quality metrics
    print("Calculating quality metrics...")
    
    # Save original render for reference
    generator.save_texture(original_render, "demo_textures/original_render_ref.png")
    
    # Try to use real quality evaluation, fallback to simplified
    try:
        # Use the real renderer's evaluate method
        legit_psnr, legit_ssim = real_renderer.evaluate(
            "demo_textures/original_render_ref.png", 
            legit_render
        )
        
        attack_psnr, attack_ssim = real_renderer.evaluate(
            "demo_textures/original_render_ref.png", 
            attack_render
        )
        
        print("Using real RenderSimulator quality evaluation")
        
    except (NameError, AttributeError) as e:
        print(f"Real quality evaluation not available ({e}), using simplified implementation")
        legit_psnr = renderer.calculate_psnr(original_render, legit_render)
        legit_ssim = renderer.calculate_ssim(original_render, legit_render)
        
        attack_psnr = renderer.calculate_psnr(original_render, attack_render)
        attack_ssim = renderer.calculate_ssim(original_render, attack_render)
    
    print(f"Legitimate Render - PSNR: {legit_psnr:.2f} dB, SSIM: {legit_ssim:.4f}")
    print(f"Attack Render - PSNR: {attack_psnr:.2f} dB, SSIM: {attack_ssim:.4f}")
    
    # Step 7: Create visualization
    print("Creating demonstration figures...")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Perceptual Interdependence: Legitimate vs Attack Scenarios', fontsize=16)
    
    # Top row: Textures
    axes[0, 0].imshow(wood_albedo)
    axes[0, 0].set_title('Original Albedo')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(legit_albedo)
    axes[0, 1].set_title('Poisoned Albedo (User 42)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(wood_normal)
    axes[0, 2].set_title('Original Normal Map')
    axes[0, 2].axis('off')
    
    # Bottom row: Rendered results
    axes[1, 0].imshow(original_render)
    axes[1, 0].set_title('Original Render')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(legit_render)
    axes[1, 1].set_title(f'Legitimate Render\nPSNR: {legit_psnr:.1f} dB, SSIM: {legit_ssim:.3f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(attack_render)
    axes[1, 2].set_title(f'Attack Render\nPSNR: {attack_psnr:.1f} dB, SSIM: {attack_ssim:.3f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_textures/fig_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate legitimate and attack figures for README
    fig_legit, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(legit_render)
    ax.set_title(f'Legitimate Rendering\nPSNR: {legit_psnr:.2f} dB | SSIM: {legit_ssim:.4f}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('fig_legit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_attack, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(attack_render)
    ax.set_title(f'Attack Rendering\nPSNR: {attack_psnr:.2f} dB | SSIM: {attack_ssim:.4f}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('fig_attack.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Demonstration complete!")
    print(f"Generated files:")
    print(f"  - fig_legit.png (Legitimate scenario)")
    print(f"  - fig_attack.png (Attack scenario)")
    print(f"  - demo_textures/fig_comparison.png (Complete comparison)")
    
    return {
        'legit_psnr': legit_psnr,
        'legit_ssim': legit_ssim,
        'attack_psnr': attack_psnr,
        'attack_ssim': attack_ssim
    }

if __name__ == "__main__":
    results = create_demonstration_images()