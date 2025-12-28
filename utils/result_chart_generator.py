#!/usr/bin/env python3
"""
Result Chart Generator for Perceptual Interdependence Experiments

Creates comprehensive 6-panel visualization charts showing:
- Original assets (albedo, normal, rendered)
- Experimental results (legitimate, attack, difference)
- Quality metrics and analysis
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Optional
import os


class ResultChartGenerator:
    """
    Generates comprehensive result charts for experimental validation.
    
    Creates publication-ready 6-panel visualizations showing complete
    experimental pipeline results including original assets, rendered
    scenarios, and quality analysis.
    """
    
    def __init__(self):
        """Initialize the result chart generator."""
        self.figure_size = (16, 10)  # Figure size in inches
        self.dpi = 150  # High resolution for publication
        self.panel_size = (512, 512)  # Standard panel size for display
        
    def create_comprehensive_chart(self, 
                                 original_albedo_path: str,
                                 original_normal_path: str,
                                 legitimate_render_path: str,
                                 attack_render_path: str,
                                 texture_name: str,
                                 quality_metrics: Dict,
                                 output_path: str = None) -> str:
        """
        Create a comprehensive 6-panel result chart.
        
        Args:
            original_albedo_path (str): Path to original albedo texture
            original_normal_path (str): Path to original normal map
            legitimate_render_path (str): Path to legitimate rendering
            attack_render_path (str): Path to attack rendering
            texture_name (str): Name of texture being tested
            quality_metrics (Dict): Quality metrics (PSNR, SSIM values)
            output_path (str, optional): Custom output path
            
        Returns:
            str: Path to generated chart
        """
        try:
            # Load all images
            original_albedo = self._load_and_resize_image(original_albedo_path)
            original_normal = self._load_and_resize_image(original_normal_path)
            legitimate_render = self._load_and_resize_image(legitimate_render_path)
            attack_render = self._load_and_resize_image(attack_render_path)
            
            # Create original render (simulate clean rendering)
            original_render = self._create_original_render(original_albedo, original_normal)
            
            # Create difference map
            difference_map = self._create_difference_map(legitimate_render, attack_render)
            
            # Create the comprehensive chart
            chart_path = self._generate_chart(
                original_albedo, original_normal, original_render,
                legitimate_render, attack_render, difference_map,
                texture_name, quality_metrics, output_path
            )
            
            return chart_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to create comprehensive chart: {str(e)}")
    
    def _load_and_resize_image(self, image_path: str) -> np.ndarray:
        """Load and resize image to standard panel size."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.panel_size, Image.Resampling.LANCZOS)
            return np.array(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    def _create_original_render(self, albedo: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Create a simulated original render from albedo and normal."""
        # Simple lighting simulation for visualization
        # Convert normal map from RGB to normal vectors
        normal_vectors = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # Simple directional lighting
        light_dir = np.array([0.0, 0.0, 1.0])  # Top-down lighting
        
        # Calculate lighting intensity
        lighting = np.maximum(0.0, np.dot(normal_vectors, light_dir))
        lighting = np.expand_dims(lighting, axis=2)
        
        # Apply lighting to albedo
        rendered = (albedo.astype(np.float32) / 255.0) * lighting
        rendered = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
        
        return rendered
    
    def _create_difference_map(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Create a difference map between two images."""
        # Calculate absolute difference
        diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        
        # Enhance visibility by scaling
        diff = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
        
        # Convert to grayscale for better visibility
        diff_gray = np.mean(diff, axis=2, keepdims=True)
        diff_colored = np.repeat(diff_gray, 3, axis=2)
        
        return diff_colored.astype(np.uint8)
    
    def _generate_chart(self, 
                       original_albedo: np.ndarray,
                       original_normal: np.ndarray, 
                       original_render: np.ndarray,
                       legitimate_render: np.ndarray,
                       attack_render: np.ndarray,
                       difference_map: np.ndarray,
                       texture_name: str,
                       quality_metrics: Dict,
                       output_path: str = None) -> str:
        """Generate the final 6-panel chart."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(f'Perceptual Interdependence: Real Asset Demonstration\n{texture_name.replace("_", " ").title()} Texture', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Panel titles and images
        panels = [
            ("Original Albedo", original_albedo),
            ("Original Normal Map", original_normal),
            ("Original Render", original_render),
            ("Legitimate Render", legitimate_render),
            ("Attack Render", attack_render),
            ("Difference Map", difference_map)
        ]
        
        # Quality metrics for display
        legit_psnr = quality_metrics.get('legitimate_psnr', 0)
        legit_ssim = quality_metrics.get('legitimate_ssim', 0)
        attack_psnr = quality_metrics.get('attack_psnr', 0)
        attack_ssim = quality_metrics.get('attack_ssim', 0)
        
        # Add quality metrics to panel titles
        panel_subtitles = [
            "",  # Original Albedo
            "",  # Original Normal Map  
            "",  # Original Render
            f"PSNR: {legit_psnr:.1f} dB, SSIM: {legit_ssim:.3f}",  # Legitimate
            f"PSNR: {attack_psnr:.1f} dB, SSIM: {attack_ssim:.3f}",  # Attack
            f"Quality Δ: {legit_psnr - attack_psnr:.1f} dB"  # Difference
        ]
        
        # Plot each panel
        for i, ((title, image), subtitle) in enumerate(zip(panels, panel_subtitles)):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Display image
            ax.imshow(image)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            
            # Add subtitle with metrics
            if subtitle:
                ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Add experimental summary
        summary_text = f"""
Experimental Results Summary:
• Texture: {texture_name.replace('_', ' ').title()}
• Legitimate Quality: PSNR {legit_psnr:.1f} dB, SSIM {legit_ssim:.3f}
• Attack Quality: PSNR {attack_psnr:.1f} dB, SSIM {attack_ssim:.3f}
• Quality Degradation: {legit_psnr - attack_psnr:.1f} dB ({((legit_psnr - attack_psnr)/attack_psnr*100):.1f}%)
• Detection Status: {'PASSED' if legit_psnr > attack_psnr else 'FAILED'}
        """.strip()
        
        fig.text(0.02, 0.02, summary_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                verticalalignment='bottom')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95)
        
        # Save the chart
        if output_path is None:
            output_path = f"figures/{texture_name}_comprehensive_results.png"
        
        # Ensure figures directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path


def create_result_chart(original_albedo_path: str,
                       original_normal_path: str, 
                       legitimate_render_path: str,
                       attack_render_path: str,
                       texture_name: str,
                       quality_metrics: Dict,
                       output_path: str = None) -> str:
    """
    Convenience function to create a result chart.
    
    Args:
        original_albedo_path (str): Path to original albedo texture
        original_normal_path (str): Path to original normal map
        legitimate_render_path (str): Path to legitimate rendering
        attack_render_path (str): Path to attack rendering
        texture_name (str): Name of texture being tested
        quality_metrics (Dict): Quality metrics dictionary
        output_path (str, optional): Custom output path
        
    Returns:
        str: Path to generated chart
    """
    generator = ResultChartGenerator()
    return generator.create_comprehensive_chart(
        original_albedo_path, original_normal_path,
        legitimate_render_path, attack_render_path,
        texture_name, quality_metrics, output_path
    )