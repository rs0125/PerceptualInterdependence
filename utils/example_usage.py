#!/usr/bin/env python3
"""
Example Usage Script for Render Validation System

This script demonstrates how to use the RenderSimulator class for:
1. Basic texture rendering with PBR-lite pipeline
2. Quality assessment using PSNR and SSIM metrics
3. Binding validation experiments (legitimate vs attack scenarios)

Requirements:
- numpy
- PIL (Pillow)
- scikit-image

Example texture files needed:
- albedo_texture.png (RGB albedo texture)
- normal_map.png (RGB normal map)
- clean_reference.png (clean reference image for comparison)
- bound_albedo_a.png (bound albedo texture A)
- bound_normal_a.png (bound normal texture A - legitimate pair)
- bound_normal_b.png (bound normal texture B - attack pair)
"""

import numpy as np
from PIL import Image
import os
from render_simulator import RenderSimulator


def create_sample_textures():
    """
    Create sample texture files for demonstration purposes.
    
    This function generates simple synthetic textures that can be used
    to test the RenderSimulator functionality when real texture files
    are not available.
    """
    print("Creating sample texture files...")
    
    # Create a simple albedo texture (64x64 RGB)
    albedo = np.random.rand(64, 64, 3) * 255
    albedo_img = Image.fromarray(albedo.astype(np.uint8), 'RGB')
    albedo_img.save('sample_albedo.png')
    
    # Create a simple normal map (64x64 RGB, representing normalized vectors)
    # Normal vectors in [-1, 1] range, packed to [0, 255]
    normal_vectors = np.random.rand(64, 64, 3) * 2.0 - 1.0  # [-1, 1] range
    normal_packed = ((normal_vectors + 1.0) / 2.0) * 255.0  # Pack to [0, 255]
    normal_img = Image.fromarray(normal_packed.astype(np.uint8), 'RGB')
    normal_img.save('sample_normal.png')
    
    # Create a clean reference image
    reference = np.random.rand(64, 64, 3) * 255
    reference_img = Image.fromarray(reference.astype(np.uint8), 'RGB')
    reference_img.save('sample_reference.png')
    
    # Create bound textures for experiment
    bound_albedo = np.random.rand(64, 64, 3) * 255
    bound_albedo_img = Image.fromarray(bound_albedo.astype(np.uint8), 'RGB')
    bound_albedo_img.save('sample_bound_albedo.png')
    
    bound_normal_a = np.random.rand(64, 64, 3) * 2.0 - 1.0
    bound_normal_a_packed = ((bound_normal_a + 1.0) / 2.0) * 255.0
    bound_normal_a_img = Image.fromarray(bound_normal_a_packed.astype(np.uint8), 'RGB')
    bound_normal_a_img.save('sample_bound_normal_a.png')
    
    bound_normal_b = np.random.rand(64, 64, 3) * 2.0 - 1.0
    bound_normal_b_packed = ((bound_normal_b + 1.0) / 2.0) * 255.0
    bound_normal_b_img = Image.fromarray(bound_normal_b_packed.astype(np.uint8), 'RGB')
    bound_normal_b_img.save('sample_bound_normal_b.png')
    
    print("Sample texture files created successfully!")


def example_basic_rendering():
    """
    Demonstrate basic texture rendering using the RenderSimulator.
    
    Shows how to:
    - Initialize the RenderSimulator
    - Render textures with default and custom light directions
    - Handle the rendered output
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Texture Rendering")
    print("="*60)
    
    # Initialize the RenderSimulator
    simulator = RenderSimulator()
    print("✓ RenderSimulator initialized")
    
    # Example 1a: Render with default light direction [0, 0, 1]
    print("\n1a. Rendering with default light direction [0, 0, 1]:")
    try:
        rendered_image = simulator.render(
            albedo_path='sample_albedo.png',
            normal_path='sample_normal.png'
        )
        print(f"✓ Rendered image shape: {rendered_image.shape}")
        print(f"✓ Rendered image value range: [{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
        print(f"✓ Rendered image dtype: {rendered_image.dtype}")
        
        # Save rendered output for inspection
        rendered_output = (rendered_image * 255).astype(np.uint8)
        Image.fromarray(rendered_output, 'RGB').save('rendered_default_light.png')
        print("✓ Rendered image saved as 'rendered_default_light.png'")
        
    except Exception as e:
        print(f"✗ Error during rendering: {e}")
    
    # Example 1b: Render with custom light direction
    print("\n1b. Rendering with custom light direction [1, 1, 1]:")
    try:
        custom_light = [1, 1, 1]  # Will be normalized to unit vector
        rendered_image_custom = simulator.render(
            albedo_path='sample_albedo.png',
            normal_path='sample_normal.png',
            light_dir=custom_light
        )
        print(f"✓ Rendered image shape: {rendered_image_custom.shape}")
        print(f"✓ Rendered image value range: [{rendered_image_custom.min():.3f}, {rendered_image_custom.max():.3f}]")
        
        # Save rendered output for comparison
        rendered_output_custom = (rendered_image_custom * 255).astype(np.uint8)
        Image.fromarray(rendered_output_custom, 'RGB').save('rendered_custom_light.png')
        print("✓ Rendered image saved as 'rendered_custom_light.png'")
        
    except Exception as e:
        print(f"✗ Error during custom light rendering: {e}")
    
    # Example 1c: Demonstrate different light directions
    print("\n1c. Comparing different light directions:")
    light_directions = [
        [0, 0, 1],    # Front light
        [1, 0, 0],    # Side light
        [0, 1, 0],    # Top light
        [-1, -1, 1]   # Angled light
    ]
    
    for i, light_dir in enumerate(light_directions):
        try:
            rendered = simulator.render(
                albedo_path='sample_albedo.png',
                normal_path='sample_normal.png',
                light_dir=light_dir
            )
            avg_brightness = np.mean(rendered)
            print(f"  Light {light_dir}: Average brightness = {avg_brightness:.3f}")
        except Exception as e:
            print(f"  Light {light_dir}: Error = {e}")


def example_quality_assessment():
    """
    Demonstrate quality assessment using PSNR and SSIM metrics.
    
    Shows how to:
    - Calculate quality metrics between images
    - Interpret PSNR and SSIM values
    - Handle different image comparison scenarios
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Quality Assessment with PSNR and SSIM")
    print("="*60)
    
    simulator = RenderSimulator()
    
    # Example 2a: Compare rendered image with reference
    print("\n2a. Comparing rendered image with clean reference:")
    try:
        # First render an image
        rendered_image = simulator.render(
            albedo_path='sample_albedo.png',
            normal_path='sample_normal.png'
        )
        
        # Calculate quality metrics against reference
        psnr, ssim = simulator.evaluate(
            clean_ref_path='sample_reference.png',
            rendered_img=rendered_image
        )
        
        print(f"✓ PSNR: {psnr:.2f} dB")
        print(f"✓ SSIM: {ssim:.4f}")
        
        # Interpret the results
        print("\nInterpretation:")
        if psnr > 30:
            print(f"  PSNR ({psnr:.1f} dB): High quality (>30 dB is generally good)")
        elif psnr > 20:
            print(f"  PSNR ({psnr:.1f} dB): Moderate quality (20-30 dB)")
        else:
            print(f"  PSNR ({psnr:.1f} dB): Low quality (<20 dB)")
        
        if ssim > 0.9:
            print(f"  SSIM ({ssim:.3f}): Very similar (>0.9 is excellent)")
        elif ssim > 0.7:
            print(f"  SSIM ({ssim:.3f}): Moderately similar (0.7-0.9)")
        else:
            print(f"  SSIM ({ssim:.3f}): Low similarity (<0.7)")
            
    except Exception as e:
        print(f"✗ Error during quality assessment: {e}")
    
    # Example 2b: Compare identical images (should give perfect scores)
    print("\n2b. Comparing identical images (perfect case):")
    try:
        # Load the same image twice
        psnr_perfect, ssim_perfect = simulator.evaluate(
            clean_ref_path='sample_reference.png',
            rendered_img=np.array(Image.open('sample_reference.png'), dtype=np.float32) / 255.0
        )
        
        print(f"✓ PSNR (identical): {psnr_perfect}")
        print(f"✓ SSIM (identical): {ssim_perfect:.6f}")
        print("  Expected: PSNR = inf, SSIM = 1.0 for identical images")
        
    except Exception as e:
        print(f"✗ Error during perfect case assessment: {e}")
    
    # Example 2c: Compare very different images
    print("\n2c. Comparing different rendered images:")
    try:
        # Render with two different light directions
        rendered_1 = simulator.render(
            albedo_path='sample_albedo.png',
            normal_path='sample_normal.png',
            light_dir=[0, 0, 1]
        )
        
        rendered_2 = simulator.render(
            albedo_path='sample_albedo.png',
            normal_path='sample_normal.png',
            light_dir=[1, 0, 0]
        )
        
        # Compare the two rendered images
        # Note: We need to save one as reference to use evaluate method
        rendered_1_uint8 = (rendered_1 * 255).astype(np.uint8)
        Image.fromarray(rendered_1_uint8, 'RGB').save('temp_reference.png')
        
        psnr_diff, ssim_diff = simulator.evaluate(
            clean_ref_path='temp_reference.png',
            rendered_img=rendered_2
        )
        
        print(f"✓ PSNR (different lighting): {psnr_diff:.2f} dB")
        print(f"✓ SSIM (different lighting): {ssim_diff:.4f}")
        print("  This shows how lighting changes affect quality metrics")
        
        # Clean up temporary file
        os.remove('temp_reference.png')
        
    except Exception as e:
        print(f"✗ Error during different images assessment: {e}")


def example_binding_experiment():
    """
    Demonstrate binding validation experiments.
    
    Shows how to:
    - Run legitimate vs attack binding tests
    - Interpret experiment results
    - Detect binding effectiveness
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Binding Validation Experiment")
    print("="*60)
    
    simulator = RenderSimulator()
    
    print("\n3a. Running complete binding validation experiment:")
    try:
        # Run the binding experiment
        results = simulator.run_binding_experiment(
            clean_ref_path='sample_reference.png',
            bound_albedo_a='sample_bound_albedo.png',
            bound_normal_a='sample_bound_normal_a.png',
            bound_normal_b='sample_bound_normal_b.png',
            light_dir=[0, 0, 1]
        )
        
        print("✓ Experiment completed successfully!")
        print("\nResults:")
        print(f"  Legitimate Test (A+A):")
        print(f"    PSNR: {results['legitimate_psnr']:.2f} dB")
        print(f"    SSIM: {results['legitimate_ssim']:.4f}")
        
        print(f"  Attack Test (A+B):")
        print(f"    PSNR: {results['attack_psnr']:.2f} dB")
        print(f"    SSIM: {results['attack_ssim']:.4f}")
        
        print(f"  Deltas (Legitimate - Attack):")
        print(f"    PSNR Delta: {results['psnr_delta']:.2f} dB")
        print(f"    SSIM Delta: {results['ssim_delta']:.4f}")
        
        # Interpret binding effectiveness
        print("\nBinding Effectiveness Analysis:")
        if results['psnr_delta'] > 5.0:
            print(f"  ✓ Strong PSNR protection (Δ = {results['psnr_delta']:.1f} dB > 5 dB)")
        elif results['psnr_delta'] > 2.0:
            print(f"  ~ Moderate PSNR protection (Δ = {results['psnr_delta']:.1f} dB)")
        else:
            print(f"  ✗ Weak PSNR protection (Δ = {results['psnr_delta']:.1f} dB < 2 dB)")
        
        if results['ssim_delta'] > 0.1:
            print(f"  ✓ Strong SSIM protection (Δ = {results['ssim_delta']:.3f} > 0.1)")
        elif results['ssim_delta'] > 0.05:
            print(f"  ~ Moderate SSIM protection (Δ = {results['ssim_delta']:.3f})")
        else:
            print(f"  ✗ Weak SSIM protection (Δ = {results['ssim_delta']:.3f} < 0.05)")
        
        # Overall assessment
        if results['psnr_delta'] > 2.0 and results['ssim_delta'] > 0.05:
            print(f"\n  🎯 Overall: Binding appears EFFECTIVE")
        else:
            print(f"\n  ⚠️  Overall: Binding may be VULNERABLE to attacks")
            
    except Exception as e:
        print(f"✗ Error during binding experiment: {e}")
    
    # Example 3b: Experiment with different light directions
    print("\n3b. Testing binding under different lighting conditions:")
    light_conditions = [
        ([0, 0, 1], "Front lighting"),
        ([1, 1, 1], "Diagonal lighting"),
        ([0, 1, 0], "Top lighting")
    ]
    
    for light_dir, description in light_conditions:
        try:
            results = simulator.run_binding_experiment(
                clean_ref_path='sample_reference.png',
                bound_albedo_a='sample_bound_albedo.png',
                bound_normal_a='sample_bound_normal_a.png',
                bound_normal_b='sample_bound_normal_b.png',
                light_dir=light_dir
            )
            
            print(f"  {description}:")
            print(f"    PSNR Δ: {results['psnr_delta']:.2f} dB, SSIM Δ: {results['ssim_delta']:.4f}")
            
        except Exception as e:
            print(f"  {description}: Error = {e}")


def example_error_handling():
    """
    Demonstrate error handling and edge cases.
    
    Shows how the RenderSimulator handles:
    - Missing files
    - Invalid parameters
    - Dimension mismatches
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling and Edge Cases")
    print("="*60)
    
    simulator = RenderSimulator()
    
    # Example 4a: Missing file handling
    print("\n4a. Testing missing file handling:")
    try:
        simulator.render('nonexistent_albedo.png', 'sample_normal.png')
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ Correctly caught missing file: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Example 4b: Invalid light direction
    print("\n4b. Testing invalid light direction:")
    try:
        simulator.render(
            'sample_albedo.png', 
            'sample_normal.png', 
            light_dir=[0, 0, 0]  # Zero-length vector
        )
        print("✗ Should have raised ValueError for zero-length vector")
    except ValueError as e:
        print(f"✓ Correctly caught invalid light direction: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Example 4c: Invalid parameter types
    print("\n4c. Testing invalid parameter types:")
    try:
        simulator.render(123, 'sample_normal.png')  # Non-string path
        print("✗ Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Correctly caught type error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def cleanup_sample_files():
    """Clean up generated sample files."""
    sample_files = [
        'sample_albedo.png', 'sample_normal.png', 'sample_reference.png',
        'sample_bound_albedo.png', 'sample_bound_normal_a.png', 'sample_bound_normal_b.png',
        'rendered_default_light.png', 'rendered_custom_light.png'
    ]
    
    print(f"\nCleaning up sample files...")
    for filename in sample_files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"  ✓ Removed {filename}")
        except Exception as e:
            print(f"  ✗ Failed to remove {filename}: {e}")


def main():
    """
    Main function demonstrating complete RenderSimulator usage.
    
    This function runs through all examples showing:
    1. Basic rendering functionality
    2. Quality assessment capabilities  
    3. Binding validation experiments
    4. Error handling and edge cases
    """
    print("Render Validation System - Example Usage")
    print("=" * 60)
    print("This script demonstrates the RenderSimulator class capabilities.")
    print("It will create sample texture files and run various examples.")
    
    try:
        # Create sample textures for demonstration
        create_sample_textures()
        
        # Run all examples
        example_basic_rendering()
        example_quality_assessment()
        example_binding_experiment()
        example_error_handling()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. RenderSimulator provides PBR-lite rendering for texture validation")
        print("2. PSNR and SSIM metrics quantify visual quality differences")
        print("3. Binding experiments detect texture binding effectiveness")
        print("4. The system handles errors gracefully with informative messages")
        print("\nExpected outputs:")
        print("- High PSNR (>30 dB) and SSIM (>0.9) indicate good binding quality")
        print("- Large deltas between legitimate and attack tests show effective binding")
        print("- Different lighting conditions may affect binding effectiveness")
        
    except Exception as e:
        print(f"\n✗ Error during example execution: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy pillow scikit-image")
    
    finally:
        # Clean up sample files
        cleanup_sample_files()


if __name__ == "__main__":
    main()