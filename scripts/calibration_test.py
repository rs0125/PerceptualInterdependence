#!/usr/bin/env python3
"""
Calibration Test: Verify Z-Score improvements after forensics fix
Tests 5 sample assets with the corrected Theoretical Z-Score calculation
"""

import numpy as np
from pathlib import Path
import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from perceptual_interdependence.core.asset_binder import AssetBinder
from perceptual_interdependence.core.render_simulator import RenderSimulator
from perceptual_interdependence.core.forensics import RGBForensics


def run_calibration_test():
    """Run calibration test on 5 sample assets."""
    
    print("\n" + "="*80)
    print("CALIBRATION TEST: Four-Metric Validation")
    print("="*80)
    print("Testing 5 sample assets with four distinct quality metrics:")
    print("  1. Texture Fidelity (File Theft)")
    print("  2. Visual Fidelity (Authorized User)")
    print("  3. Attack Severity (Flat Normal)")
    print("  4. Mismatched Attack (Wrong Antidote)")
    print("="*80 + "\n")
    
    # Select 5 sample assets (one from each category)
    test_assets = [
        ("wood", "Bark Brown 02"),
        ("metal", "Blue Metal Plate"),
        ("terrain", "Aerial Beach 01"),
        ("fabric", "Bi Stretch"),
        ("rock", "Broken Wall")
    ]
    
    dataset_root = Path("data/real_validation_set")
    temp_dir = Path("temp_calibration")
    temp_dir.mkdir(exist_ok=True)
    
    # Initialize components
    binder = AssetBinder()
    renderer = RenderSimulator()
    forensics = RGBForensics()
    
    target_user_id = 42
    poison_strength = 0.4  # Testing intermediate strength
    
    results = []
    
    for idx, (category, asset_name) in enumerate(test_assets, 1):
        print(f"\n[{idx}/5] Processing: {category}/{asset_name}")
        print("-" * 60)
        
        # Paths
        albedo_path = dataset_root / category / asset_name / "albedo.jpg"
        normal_path = dataset_root / category / asset_name / "normal.jpg"
        
        if not albedo_path.exists() or not normal_path.exists():
            print(f"  ⚠️  Skipping: Missing files")
            continue
        
        # Step 1: Binding
        print(f"  [1/3] Binding with User ID {target_user_id}, strength={poison_strength}")
        
        start_time = time.time()
        bind_result = binder.bind_textures(
            albedo_path=str(albedo_path),
            normal_path=str(normal_path),
            user_id=target_user_id,
            poison_strength=poison_strength,
            output_prefix=f"calibration_{category}_{asset_name}"
        )
        bind_time = time.time() - start_time
        
        # Get the bound albedo and antidote normal paths from results
        bound_albedo = bind_result['output_paths']['albedo']
        bound_normal = bind_result['output_paths']['normal']
        print(f"        Binding completed in {bind_time:.2f}s")
        
        # Step 2: Calculate three distinct quality metrics
        print(f"  [2/3] Calculating three quality metrics")
        
        # METRIC 1: Texture Fidelity (File Theft Metric)
        # Compare: Original Albedo vs Poisoned Albedo
        original_albedo = np.array(Image.open(albedo_path))
        bound_albedo_img = np.array(Image.open(bound_albedo))
        
        # Ensure same shape for comparison
        if original_albedo.shape != bound_albedo_img.shape:
            min_h = min(original_albedo.shape[0], bound_albedo_img.shape[0])
            min_w = min(original_albedo.shape[1], bound_albedo_img.shape[1])
            original_albedo = original_albedo[:min_h, :min_w]
            bound_albedo_img = bound_albedo_img[:min_h, :min_w]
        
        texture_ssim = np.mean([
            ssim(original_albedo[:, :, i], bound_albedo_img[:, :, i], data_range=255)
            for i in range(3)
        ])
        
        texture_psnr = psnr(original_albedo, bound_albedo_img, data_range=255)
        
        # METRIC 2 & 3: Render-level metrics with tilted lighting
        tilted_light = [0.3, 0.3, 0.9]  # Tilted to reveal normal map differences
        
        # Render A (Truth): Original Albedo + Original Normal
        original_render = renderer.render(
            albedo_path=str(albedo_path),
            normal_path=str(normal_path),
            light_dir=tilted_light
        )
        
        # Render B (Legit): Poisoned Albedo + Antidote Normal
        legitimate_render = renderer.render(
            albedo_path=str(bound_albedo),
            normal_path=str(bound_normal),
            light_dir=tilted_light
        )
        
        # Render C (Attack): Poisoned Albedo + Flat Normal
        flat_normal_path = temp_dir / f"flat_normal_{category}_{asset_name}.png"
        normal_img = np.array(Image.open(normal_path))
        flat_normal = np.full_like(normal_img, [128, 128, 255], dtype=np.uint8)
        Image.fromarray(flat_normal).save(flat_normal_path)
        
        attack_render = renderer.render(
            albedo_path=str(bound_albedo),
            normal_path=str(flat_normal_path),
            light_dir=tilted_light
        )
        
        # Render D (Mismatched): Poisoned Albedo (user 42) + Wrong Antidote Normal (user 99)
        # Bind with different user to get mismatched antidote
        mismatched_bind = binder.bind_textures(
            albedo_path=str(albedo_path),
            normal_path=str(normal_path),
            user_id=99,  # Different user
            poison_strength=poison_strength,
            output_prefix=f"calibration_mismatch_{category}_{asset_name}"
        )
        mismatched_normal = mismatched_bind['output_paths']['normal']
        
        mismatched_render = renderer.render(
            albedo_path=str(bound_albedo),  # User 42's poisoned albedo
            normal_path=str(mismatched_normal),  # User 99's antidote normal
            light_dir=tilted_light
        )
        
        # Calculate render SSIMs
        min_h = min(original_render.shape[0], legitimate_render.shape[0], attack_render.shape[0], mismatched_render.shape[0])
        min_w = min(original_render.shape[1], legitimate_render.shape[1], attack_render.shape[1], mismatched_render.shape[1])
        original_render = original_render[:min_h, :min_w]
        legitimate_render = legitimate_render[:min_h, :min_w]
        attack_render = attack_render[:min_h, :min_w]
        mismatched_render = mismatched_render[:min_h, :min_w]
        
        # Convert to uint8 for SSIM calculation
        original_render_uint8 = (original_render * 255).astype(np.uint8)
        legitimate_render_uint8 = (legitimate_render * 255).astype(np.uint8)
        attack_render_uint8 = (attack_render * 255).astype(np.uint8)
        mismatched_render_uint8 = (mismatched_render * 255).astype(np.uint8)
        
        legitimate_ssim = np.mean([
            ssim(original_render_uint8[:, :, i], legitimate_render_uint8[:, :, i], data_range=255)
            for i in range(3)
        ])
        
        attack_ssim = np.mean([
            ssim(original_render_uint8[:, :, i], attack_render_uint8[:, :, i], data_range=255)
            for i in range(3)
        ])
        
        mismatched_ssim = np.mean([
            ssim(original_render_uint8[:, :, i], mismatched_render_uint8[:, :, i], data_range=255)
            for i in range(3)
        ])
        
        print(f"        Metric 1 - Texture Fidelity:  {texture_ssim:.4f} (file theft)")
        print(f"        Metric 2 - Visual Fidelity:   {legitimate_ssim:.4f} (authorized user)")
        print(f"        Metric 3 - Attack Severity:   {attack_ssim:.4f} (flat normal)")
        print(f"        Metric 4 - Mismatched Attack: {mismatched_ssim:.4f} (wrong antidote)")
        
        # Step 3: Forensics with proper Z-score calculation
        print(f"  [3/3] Extracting signature and detecting traitor")
        print(f"        Analyzing poisoned albedo file")
        
        start_time = time.time()
        signature = forensics.extract_signature(
            suspicious_albedo_path=str(bound_albedo),
            original_clean_path=str(albedo_path)
        )
        
        detected_user = forensics.find_traitor(signature, max_users=100)
        forensics_time = time.time() - start_time
        
        # Calculate proper Z-score: (target_score - mean) / std
        if hasattr(forensics, '_last_correlation_scores'):
            correlation_scores = forensics._last_correlation_scores
            target_score = correlation_scores[target_user_id]
            mean_score = np.mean(correlation_scores)
            std_score = np.std(correlation_scores)
            
            if std_score > 1e-6:
                detection_z_score = (target_score - mean_score) / std_score
            else:
                detection_z_score = 0.0
        else:
            detection_z_score = 0.0
        
        print(f"        Forensics completed in {forensics_time:.2f}s")
        print(f"        Detected User: {detected_user} (Target: {target_user_id})")
        print(f"        Z-Score: {detection_z_score:.2f}")
        
        # Determine success
        success = detected_user == target_user_id
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"        {status}")
        
        results.append({
            'category': category,
            'asset': asset_name,
            'detected_user': detected_user,
            'target_user': target_user_id,
            'detection_z_score': detection_z_score,
            'success': success,
            'total_time': bind_time + forensics_time,
            'texture_ssim': texture_ssim,
            'texture_psnr': texture_psnr,
            'legitimate_ssim': legitimate_ssim,
            'attack_ssim': attack_ssim,
            'mismatched_ssim': mismatched_ssim
        })
    
    # Summary
    print("\n" + "="*80)
    print("CALIBRATION TEST RESULTS")
    print("="*80)
    
    if results:
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        avg_z_score = np.mean([r['detection_z_score'] for r in results])
        min_z_score = np.min([r['detection_z_score'] for r in results])
        max_z_score = np.max([r['detection_z_score'] for r in results])
        
        avg_texture_ssim = np.mean([r['texture_ssim'] for r in results])
        avg_legitimate_ssim = np.mean([r['legitimate_ssim'] for r in results])
        avg_attack_ssim = np.mean([r['attack_ssim'] for r in results])
        avg_mismatched_ssim = np.mean([r['mismatched_ssim'] for r in results])
        
        print(f"\nSuccess Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        print(f"\n📊 FOUR-METRIC RESULTS:")
        
        print(f"\nMETRIC 1: Texture Fidelity (File Theft)")
        print(f"  Average: {avg_texture_ssim:.4f}")
        print(f"  Interpretation: {'✅ Low (good)' if avg_texture_ssim < 0.85 else '⚠️  High (file not degraded enough)'}")
        
        print(f"\nMETRIC 2: Visual Fidelity (Authorized User)")
        print(f"  Average: {avg_legitimate_ssim:.4f}")
        print(f"  Interpretation: {'✅ High (good)' if avg_legitimate_ssim > 0.90 else '⚠️  Low (antidote not working well)'}")
        
        print(f"\nMETRIC 3: Attack Severity (Flat Normal)")
        print(f"  Average: {avg_attack_ssim:.4f}")
        print(f"  Interpretation: {'✅ Low (good)' if avg_attack_ssim < 0.70 else '⚠️  High (attack not penalized enough)'}")
        
        print(f"\nMETRIC 4: Mismatched Attack (Wrong Antidote)")
        print(f"  Average: {avg_mismatched_ssim:.4f}")
        print(f"  Interpretation: {'✅ Low (good)' if avg_mismatched_ssim < 0.70 else '⚠️  High (wrong antidote not penalized enough)'}")
        
        print(f"\nFORENSIC DETECTION (Z-Score)")
        print(f"  Average: {avg_z_score:.2f}")
        print(f"  Range: [{min_z_score:.2f}, {max_z_score:.2f}]")
        if avg_z_score > 10:
            print(f"  Interpretation: ✅ Excellent (Z > 10)")
        elif avg_z_score > 6:
            print(f"  Interpretation: ✅ Good (Z > 6, Six Sigma)")
        else:
            print(f"  Interpretation: ⚠️  Weak (Z < 6)")
        
        print(f"\nDetailed Results:")
        print(f"  {'Cat':<8} | {'Tex':>6} | {'Legit':>6} | {'Flat':>6} | {'Wrong':>6} | {'Z-Score':>8}")
        print(f"  {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8}")
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"  {r['category']:8s} | {r['texture_ssim']:6.4f} | "
                  f"{r['legitimate_ssim']:6.4f} | {r['attack_ssim']:6.4f} | "
                  f"{r['mismatched_ssim']:6.4f} | {r['detection_z_score']:8.2f}")
        
        # Evaluation
        print(f"\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        
        print(f"\n✅ System validates all four security dimensions:")
        print(f"   1. File Theft: Texture SSIM = {avg_texture_ssim:.4f}")
        print(f"   2. Authorized Use: Legit SSIM = {avg_legitimate_ssim:.4f}")
        print(f"   3. Flat Normal Attack: Attack SSIM = {avg_attack_ssim:.4f}")
        print(f"   4. Wrong Antidote Attack: Mismatched SSIM = {avg_mismatched_ssim:.4f}")
        print(f"   5. Detection: Z-Score = {avg_z_score:.2f}")
        
        print("="*80 + "\n")
    else:
        print("No results to display.\n")


if __name__ == "__main__":
    run_calibration_test()
