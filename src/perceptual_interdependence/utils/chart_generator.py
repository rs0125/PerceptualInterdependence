"""
Demonstration Chart Generator - Comprehensive Visualization System

This module implements the ChartGenerator class for creating demonstration charts
that visualize the effects of perceptual interdependence operations on texture pairs.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ..core.asset_binder import AssetBinder
from ..core.render_simulator import RenderSimulator


@dataclass
class AssetBundle:
    """Container for processed asset data."""
    original_albedo: np.ndarray
    original_normal: np.ndarray
    victim_albedo: np.ndarray
    victim_normal: np.ndarray
    attacker_albedo: np.ndarray
    attacker_normal: np.ndarray


@dataclass
class RenderResults:
    """Container for rendered image results."""
    original_render: np.ndarray
    legitimate_render: np.ndarray
    attack_render: np.ndarray


@dataclass
class QualityMetrics:
    """Container for quality assessment metrics."""
    legitimate_psnr: float
    legitimate_ssim: float
    attack_psnr: float
    attack_ssim: float
    psnr_delta: float
    ssim_delta: float


class ChartGenerator:
    """
    Demonstration chart generator for perceptual interdependence visualizations.
    
    Creates comprehensive charts showing original assets, processed results,
    quality metrics, and difference maps in a structured 2x3 layout.
    """
    
    def __init__(self):
        """Initialize ChartGenerator with core components."""
        # Create temporary directory for AssetBinder operations
        import tempfile
        import os
        temp_dir = Path(tempfile.mkdtemp(prefix="chart_gen_"))
        
        self.asset_binder = AssetBinder(output_dir=temp_dir)
        self.render_simulator = RenderSimulator()
        self.temp_dir = temp_dir  # Store for cleanup
        
        # Chart configuration
        self.figure_size = (15, 10)  # inches for high resolution
        self.dpi = 300  # High DPI for publication quality
        
    def generate_demonstration_chart(
        self, 
        albedo_path: str, 
        normal_path: str,
        victim_id: int,
        attacker_id: int,
        output_path: str
    ) -> str:
        """
        Main chart generation method.
        
        Args:
            albedo_path: Path to original albedo texture
            normal_path: Path to original normal map
            victim_id: User ID for legitimate binding scenario
            attacker_id: User ID for attack scenario
            output_path: Output path for generated chart
            
        Returns:
            Path to saved chart file
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If processing fails
        """
        try:
            # Step 1: Create chart layout
            fig, axes = self._create_chart_layout()
            
            # Step 2: Process assets through binding pipeline
            assets = self._process_assets(albedo_path, normal_path, victim_id, attacker_id)
            
            # Step 3: Generate renders for all scenarios
            renders = self._render_scenarios(assets)
            
            # Step 4: Calculate quality metrics
            metrics = self._calculate_metrics(renders)
            
            # Step 5: Compose chart with images and overlays
            self._compose_chart(fig, axes, assets, renders, metrics)
            
            # Step 6: Save chart and cleanup
            saved_path = self._save_chart(fig, output_path)
            
            # Cleanup temporary AssetBinder files
            self._cleanup_asset_binder_temp()
            
            return saved_path
            
        except Exception as e:
            # Ensure matplotlib cleanup on error
            plt.close('all')
            # Cleanup temporary AssetBinder files
            self._cleanup_asset_binder_temp()
            raise ValueError(f"Chart generation failed: {str(e)}")
    
    def _create_chart_layout(self) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create 2x3 matplotlib figure and axes grid.
        
        Returns:
            Tuple of (figure, axes_array) for chart composition
        """
        # Create figure with specified size and DPI
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Create 2x3 subplot grid
        axes = fig.subplots(2, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
        
        # Set main title
        fig.suptitle(
            'Perceptual Interdependence: Real Asset Demonstration',
            fontsize=16,
            fontweight='bold',
            y=0.95
        )
        
        # Configure individual panel titles
        panel_titles = [
            'Original Albedo',
            'Original Normal Map', 
            'Original Render',
            'Legitimate Render',
            'Attack Render',
            'Difference Map'
        ]
        
        # Apply titles to each subplot
        for i, (ax, title) in enumerate(zip(axes.flat, panel_titles)):
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        return fig, axes
    
    def _process_assets(
        self, 
        albedo_path: str, 
        normal_path: str,
        victim_id: int, 
        attacker_id: int
    ) -> AssetBundle:
        """
        Process assets through binding pipeline for both victim and attacker scenarios.
        
        Args:
            albedo_path: Path to original albedo texture
            normal_path: Path to original normal map
            victim_id: User ID for legitimate binding
            attacker_id: User ID for attack scenario
            
        Returns:
            AssetBundle containing all processed assets
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If binding operations fail
        """
        # Validate input files exist
        if not os.path.exists(albedo_path):
            raise FileNotFoundError(f"Albedo texture not found: {albedo_path}")
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal map not found: {normal_path}")
        
        try:
            # Load original assets for reference
            original_albedo = self._load_texture_as_array(albedo_path)
            original_normal = self._load_texture_as_array(normal_path)
            
            # Process victim scenario (legitimate binding)
            print(f"Processing legitimate binding scenario (victim_id: {victim_id})...")
            victim_result = self.asset_binder.bind_textures(
                albedo_path=albedo_path,
                normal_path=normal_path,
                user_id=victim_id,
                poison_strength=0.3,  # Increased from 0.2 for stronger effect
                output_prefix="victim"
            )
            
            # Load victim bound assets
            victim_albedo = self._load_texture_as_array(victim_result['output_paths']['albedo'])
            victim_normal = self._load_texture_as_array(victim_result['output_paths']['normal'])
            
            # Process attacker scenario (attack binding)
            print(f"Processing attack binding scenario (attacker_id: {attacker_id})...")
            attacker_result = self.asset_binder.bind_textures(
                albedo_path=albedo_path,
                normal_path=normal_path,
                user_id=attacker_id,
                poison_strength=0.3,  # Increased from 0.2 for stronger effect
                output_prefix="attacker"
            )
            
            # Load attacker bound assets
            attacker_albedo = self._load_texture_as_array(attacker_result['output_paths']['albedo'])
            attacker_normal = self._load_texture_as_array(attacker_result['output_paths']['normal'])
            
            # Create and return asset bundle
            return AssetBundle(
                original_albedo=original_albedo,
                original_normal=original_normal,
                victim_albedo=victim_albedo,
                victim_normal=victim_normal,
                attacker_albedo=attacker_albedo,
                attacker_normal=attacker_normal
            )
            
        except Exception as e:
            raise ValueError(f"Asset processing failed: {str(e)}")
    
    def _load_texture_as_array(self, texture_path: str) -> np.ndarray:
        """
        Load texture file as normalized numpy array.
        
        Args:
            texture_path: Path to texture file
            
        Returns:
            Normalized texture array [0.0, 1.0]
            
        Raises:
            FileNotFoundError: If texture file doesn't exist
            ValueError: If texture loading fails
        """
        texture_path = Path(texture_path)
        
        if not texture_path.exists():
            raise FileNotFoundError(f"Texture file not found: {texture_path}")
        
        try:
            # Load image and convert to RGB
            image = Image.open(texture_path).convert('RGB')
            array = np.array(image, dtype=np.float32)
            
            # Normalize to [0.0, 1.0] range
            if array.max() > 1.0:
                array = array / 255.0
            
            return array
            
        except Exception as e:
            raise ValueError(f"Failed to load texture {texture_path}: {str(e)}")
    
    def _render_scenarios(self, assets: AssetBundle) -> RenderResults:
        """
        Generate renders for original, legitimate, and attack scenarios.
        
        Args:
            assets: AssetBundle containing all processed textures
            
        Returns:
            RenderResults containing rendered images
            
        Raises:
            ValueError: If rendering operations fail
        """
        try:
            # Create temporary files for rendering
            temp_dir = Path("temp_render")
            temp_dir.mkdir(exist_ok=True)
            
            # Save original assets for rendering
            original_albedo_path = temp_dir / "original_albedo.png"
            original_normal_path = temp_dir / "original_normal.png"
            self._save_array_as_image(assets.original_albedo, original_albedo_path)
            self._save_array_as_image(assets.original_normal, original_normal_path)
            
            # Save victim assets for rendering
            victim_albedo_path = temp_dir / "victim_albedo.png"
            victim_normal_path = temp_dir / "victim_normal.png"
            self._save_array_as_image(assets.victim_albedo, victim_albedo_path)
            self._save_array_as_image(assets.victim_normal, victim_normal_path)
            
            # Save attacker assets for rendering (using victim albedo with attacker normal)
            attacker_normal_path = temp_dir / "attacker_normal.png"
            self._save_array_as_image(assets.attacker_normal, attacker_normal_path)
            
            # Render original scenario
            print("Rendering original scenario...")
            original_render = self.render_simulator.render(
                str(original_albedo_path),
                str(original_normal_path),
                light_dir=[0, 0, 1]
            )
            
            # Render legitimate scenario (victim albedo + victim normal)
            print("Rendering legitimate scenario...")
            legitimate_render = self.render_simulator.render(
                str(victim_albedo_path),
                str(victim_normal_path),
                light_dir=[0, 0, 1]
            )
            
            # Render attack scenario (victim albedo + attacker normal)
            print("Rendering attack scenario...")
            attack_render = self.render_simulator.render(
                str(victim_albedo_path),
                str(attacker_normal_path),
                light_dir=[0, 0, 1]
            )
            
            # Cleanup temporary files
            self._cleanup_temp_files(temp_dir)
            
            return RenderResults(
                original_render=original_render,
                legitimate_render=legitimate_render,
                attack_render=attack_render
            )
            
        except Exception as e:
            # Ensure cleanup on error
            if 'temp_dir' in locals():
                self._cleanup_temp_files(temp_dir)
            raise ValueError(f"Rendering failed: {str(e)}")
    
    def _save_array_as_image(self, array: np.ndarray, path: Path) -> None:
        """
        Save numpy array as image file.
        
        Args:
            array: Numpy array in [0.0, 1.0] range
            path: Output file path
        """
        # Convert to [0, 255] range and uint8
        image_array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        
        # Save as PNG
        image = Image.fromarray(image_array, mode='RGB')
        image.save(path, format='PNG')
    
    def _cleanup_temp_files(self, temp_dir: Path) -> None:
        """
        Clean up temporary rendering files.
        
        Args:
            temp_dir: Directory containing temporary files
        """
        try:
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    file.unlink()
                temp_dir.rmdir()
        except Exception:
            # Ignore cleanup errors
            pass
    
    def _calculate_metrics(self, renders: RenderResults) -> QualityMetrics:
        """
        Calculate PSNR and SSIM quality metrics for legitimate and attack scenarios.
        
        Args:
            renders: RenderResults containing all rendered images
            
        Returns:
            QualityMetrics with calculated values and deltas
            
        Raises:
            ValueError: If metric calculations fail
        """
        try:
            # Create temporary reference file for evaluation
            temp_dir = Path("temp_metrics")
            temp_dir.mkdir(exist_ok=True)
            
            original_ref_path = temp_dir / "original_reference.png"
            self._save_array_as_image(renders.original_render, original_ref_path)
            
            # Calculate metrics for legitimate scenario
            print("Calculating quality metrics for legitimate scenario...")
            legitimate_psnr, legitimate_ssim = self.render_simulator.evaluate(
                str(original_ref_path),
                renders.legitimate_render
            )
            
            # Calculate metrics for attack scenario
            print("Calculating quality metrics for attack scenario...")
            attack_psnr, attack_ssim = self.render_simulator.evaluate(
                str(original_ref_path),
                renders.attack_render
            )
            
            # Calculate deltas (legitimate - attack)
            psnr_delta = legitimate_psnr - attack_psnr
            ssim_delta = legitimate_ssim - attack_ssim
            
            # Cleanup temporary files
            self._cleanup_temp_files(temp_dir)
            
            print(f"Quality metrics calculated:")
            print(f"  Legitimate: PSNR={legitimate_psnr:.2f}dB, SSIM={legitimate_ssim:.4f}")
            print(f"  Attack: PSNR={attack_psnr:.2f}dB, SSIM={attack_ssim:.4f}")
            print(f"  Deltas: PSNR={psnr_delta:.2f}dB, SSIM={ssim_delta:.4f}")
            
            return QualityMetrics(
                legitimate_psnr=legitimate_psnr,
                legitimate_ssim=legitimate_ssim,
                attack_psnr=attack_psnr,
                attack_ssim=attack_ssim,
                psnr_delta=psnr_delta,
                ssim_delta=ssim_delta
            )
            
        except Exception as e:
            # Ensure cleanup on error
            if 'temp_dir' in locals():
                self._cleanup_temp_files(temp_dir)
            raise ValueError(f"Quality metrics calculation failed: {str(e)}")
    
    def _compose_chart(
        self, 
        fig: plt.Figure, 
        axes: np.ndarray, 
        assets: AssetBundle, 
        renders: RenderResults, 
        metrics: QualityMetrics
    ) -> None:
        """
        Populate chart panels with images, metrics, and difference maps.
        
        Args:
            fig: Matplotlib figure
            axes: 2x3 array of subplot axes
            assets: AssetBundle with all processed textures
            renders: RenderResults with rendered images
            metrics: QualityMetrics with calculated values
        """
        # Panel layout: 
        # [0,0] Original Albedo    [0,1] Original Normal    [0,2] Original Render
        # [1,0] Legitimate Render  [1,1] Attack Render      [1,2] Difference Map
        
        # Top row - Original assets and render
        axes[0,0].imshow(assets.original_albedo)
        axes[0,0].set_title('Original Albedo', fontsize=12, fontweight='bold')
        
        # Convert normal map for display (from [-1,1] to [0,1] if needed)
        normal_display = self._prepare_normal_for_display(assets.original_normal)
        axes[0,1].imshow(normal_display)
        axes[0,1].set_title('Original Normal Map', fontsize=12, fontweight='bold')
        
        axes[0,2].imshow(renders.original_render)
        axes[0,2].set_title('Original Render', fontsize=12, fontweight='bold')
        
        # Bottom row - Legitimate render with metrics
        axes[1,0].imshow(renders.legitimate_render)
        axes[1,0].set_title('Legitimate Render', fontsize=12, fontweight='bold')
        
        # Add metrics overlay to legitimate render
        self._add_metrics_overlay(
            axes[1,0], 
            f"PSNR: {metrics.legitimate_psnr:.1f}dB\nSSIM: {metrics.legitimate_ssim:.3f}",
            color='white'
        )
        
        # Attack render with metrics
        axes[1,1].imshow(renders.attack_render)
        axes[1,1].set_title('Attack Render', fontsize=12, fontweight='bold')
        
        # Add metrics overlay to attack render
        self._add_metrics_overlay(
            axes[1,1], 
            f"PSNR: {metrics.attack_psnr:.1f}dB\nSSIM: {metrics.attack_ssim:.3f}",
            color='white'
        )
        
        # Generate and display enhanced difference map
        difference_map = self._generate_difference_map(
            renders.original_render, 
            renders.attack_render
        )
        
        # Display the RGB difference map (no colormap needed since it's already RGB)
        axes[1,2].imshow(difference_map)
        axes[1,2].set_title('Difference Map', fontsize=12, fontweight='bold')
        
        # Add delta metrics overlay to difference map
        self._add_metrics_overlay(
            axes[1,2], 
            f"Δ PSNR: {metrics.psnr_delta:.1f}dB\nΔ SSIM: {metrics.ssim_delta:.3f}",
            color='cyan'
        )
        
        # Remove axis ticks and labels for all panels
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    def _prepare_normal_for_display(self, normal_array: np.ndarray) -> np.ndarray:
        """
        Prepare normal map for display by ensuring [0,1] range.
        
        Args:
            normal_array: Normal map array
            
        Returns:
            Display-ready normal map
        """
        # Check if normal map is in [-1,1] range and convert to [0,1]
        if normal_array.min() < 0:
            return (normal_array + 1.0) / 2.0
        else:
            return normal_array
    
    def _add_metrics_overlay(self, ax: plt.Axes, text: str, color: str = 'white') -> None:
        """
        Add text overlay with metrics to subplot.
        
        Args:
            ax: Matplotlib axes to add overlay to
            text: Text to display
            color: Text color
        """
        # Add text with black outline for visibility
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            color=color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    def _calculate_difference_map(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray,
        method: str = 'l2'
    ) -> np.ndarray:
        """
        Calculate pixel-wise differences between two images with proper scaling and normalization.
        
        Args:
            reference: Reference image array [0.0, 1.0]
            comparison: Image to compare against reference [0.0, 1.0]
            method: Difference calculation method ('l1', 'l2', 'perceptual')
            
        Returns:
            Normalized difference map [0.0, 1.0] with enhanced visualization
            
        Raises:
            ValueError: If images have incompatible shapes or invalid method
        """
        # Validate input shapes
        if reference.shape != comparison.shape:
            raise ValueError(
                f"Image shape mismatch: reference {reference.shape} vs comparison {comparison.shape}"
            )
        
        # Ensure inputs are in valid range [0.0, 1.0]
        reference = np.clip(reference, 0.0, 1.0)
        comparison = np.clip(comparison, 0.0, 1.0)
        
        # Calculate difference based on method
        if method == 'l1':
            # L1 (Manhattan) distance
            diff = np.abs(reference - comparison)
        elif method == 'l2':
            # L2 (Euclidean) distance
            diff = np.square(reference - comparison)
        elif method == 'perceptual':
            # Perceptual difference using luminance weighting
            if len(reference.shape) == 3 and reference.shape[2] == 3:
                # RGB to luminance weights (ITU-R BT.709)
                luma_weights = np.array([0.2126, 0.7152, 0.0722])
                ref_luma = np.sum(reference * luma_weights, axis=2)
                comp_luma = np.sum(comparison * luma_weights, axis=2)
                diff = np.abs(ref_luma - comp_luma)
                # Expand back to 3 channels for consistency
                diff = np.stack([diff, diff, diff], axis=2)
            else:
                diff = np.abs(reference - comparison)
        else:
            raise ValueError(f"Unknown difference method: {method}")
        
        # Convert to grayscale if multi-channel
        if len(diff.shape) == 3:
            if diff.shape[2] == 3:
                # Use luminance conversion for RGB
                diff_gray = np.sum(diff * np.array([0.2126, 0.7152, 0.0722]), axis=2)
            else:
                # Average across channels
                diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        
        # Handle edge cases for identical or very similar images
        max_diff = diff_gray.max()
        min_diff = diff_gray.min()
        
        if max_diff == 0:
            # Images are identical - return zero difference map
            return np.zeros_like(diff_gray)
        
        if max_diff - min_diff < 1e-6:
            # Very similar images - apply minimal enhancement
            diff_normalized = np.full_like(diff_gray, 0.1)
        else:
            # Apply scaling and normalization
            # First normalize to [0, 1] range
            diff_normalized = (diff_gray - min_diff) / (max_diff - min_diff)
            
            # Apply gamma correction to enhance small differences
            gamma = 0.5  # Enhance visibility of small differences
            diff_normalized = np.power(diff_normalized, gamma)
            
            # Apply adaptive scaling based on difference magnitude
            if max_diff < 0.1:  # Very small differences
                # Amplify small differences more aggressively
                diff_normalized = np.power(diff_normalized, 0.3)
            elif max_diff > 0.5:  # Large differences
                # Compress large differences to prevent saturation
                diff_normalized = np.power(diff_normalized, 0.7)
        
        # Final clipping to ensure valid range
        diff_normalized = np.clip(diff_normalized, 0.0, 1.0)
        
        return diff_normalized
    
    def _enhance_difference_visualization(
        self, 
        difference_map: np.ndarray,
        threshold: float = 0.1,
        colormap: str = 'hot'
    ) -> Tuple[np.ndarray, str]:
        """
        Apply visual enhancements to difference map with color mapping and thresholding.
        
        Args:
            difference_map: Normalized difference map [0.0, 1.0]
            threshold: Threshold for highlighting significant changes [0.0, 1.0]
            colormap: Matplotlib colormap name for visualization
            
        Returns:
            Tuple of (enhanced_difference_map, colormap_name)
        """
        # Create enhanced difference map
        enhanced_map = difference_map.copy()
        
        # Apply threshold-based highlighting
        significant_mask = difference_map > threshold
        
        if np.any(significant_mask):
            # Boost significant differences for better visibility
            enhanced_map[significant_mask] = np.clip(
                enhanced_map[significant_mask] * 1.5, 0.0, 1.0
            )
            
            # Apply non-linear enhancement to significant regions
            enhanced_map[significant_mask] = np.power(
                enhanced_map[significant_mask], 0.8
            )
        
        # Apply adaptive contrast enhancement
        # Calculate histogram-based enhancement
        hist, bins = np.histogram(enhanced_map.flatten(), bins=256, range=(0, 1))
        
        # Find the 95th percentile for contrast stretching
        cumsum = np.cumsum(hist)
        total_pixels = cumsum[-1]
        percentile_95 = bins[np.searchsorted(cumsum, 0.95 * total_pixels)]
        
        if percentile_95 > 0.1:  # Only apply if there's meaningful contrast
            # Stretch contrast to use full dynamic range
            enhanced_map = np.clip(enhanced_map / percentile_95, 0.0, 1.0)
        
        # Apply smoothing to reduce noise while preserving edges
        try:
            from scipy import ndimage
            # Light gaussian smoothing
            enhanced_map = ndimage.gaussian_filter(enhanced_map, sigma=0.5)
        except ImportError:
            # Fallback if scipy not available - skip smoothing
            pass
        
        return enhanced_map, colormap
    
    def _create_difference_colormap(self, difference_map: np.ndarray) -> np.ndarray:
        """
        Create RGB colormap visualization for difference map.
        
        Args:
            difference_map: Normalized difference map [0.0, 1.0]
            
        Returns:
            RGB image array for difference visualization
        """
        # Create custom colormap: black -> red -> yellow -> white
        # This provides good contrast for difference visualization
        
        # Normalize input
        diff_norm = np.clip(difference_map, 0.0, 1.0)
        
        # Create RGB channels
        height, width = diff_norm.shape
        rgb_map = np.zeros((height, width, 3))
        
        # Black to red transition (0.0 to 0.33)
        mask1 = diff_norm <= 0.33
        rgb_map[mask1, 0] = diff_norm[mask1] * 3.0  # Red channel
        rgb_map[mask1, 1] = 0.0  # Green channel
        rgb_map[mask1, 2] = 0.0  # Blue channel
        
        # Red to yellow transition (0.33 to 0.66)
        mask2 = (diff_norm > 0.33) & (diff_norm <= 0.66)
        rgb_map[mask2, 0] = 1.0  # Red channel (full)
        rgb_map[mask2, 1] = (diff_norm[mask2] - 0.33) * 3.0  # Green channel
        rgb_map[mask2, 2] = 0.0  # Blue channel
        
        # Yellow to white transition (0.66 to 1.0)
        mask3 = diff_norm > 0.66
        rgb_map[mask3, 0] = 1.0  # Red channel (full)
        rgb_map[mask3, 1] = 1.0  # Green channel (full)
        rgb_map[mask3, 2] = (diff_norm[mask3] - 0.66) * 3.0  # Blue channel
        
        # Ensure valid range
        rgb_map = np.clip(rgb_map, 0.0, 1.0)
        
        return rgb_map
    
    def _generate_difference_map(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray
    ) -> np.ndarray:
        """
        Generate enhanced difference map between two images.
        
        Args:
            reference: Reference image
            comparison: Image to compare against reference
            
        Returns:
            Enhanced RGB difference map for visualization
        """
        # Calculate base difference map
        diff_map = self._calculate_difference_map(reference, comparison, method='l2')
        
        # Apply visualization enhancements
        enhanced_map, _ = self._enhance_difference_visualization(
            diff_map, 
            threshold=0.1,
            colormap='hot'
        )
        
        # Create RGB colormap visualization
        rgb_diff_map = self._create_difference_colormap(enhanced_map)
        
        return rgb_diff_map
    
    def _save_chart(self, fig: plt.Figure, output_path: str) -> str:
        """
        Save chart with high-resolution PNG output and proper cleanup.
        
        Args:
            fig: Matplotlib figure to save
            output_path: Output file path
            
        Returns:
            Absolute path to saved chart file
            
        Raises:
            IOError: If file saving fails
        """
        try:
            # Ensure output path is a Path object
            output_path = Path(output_path)
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure .png extension
            if output_path.suffix.lower() != '.png':
                output_path = output_path.with_suffix('.png')
            
            # Save with high quality settings
            fig.savefig(
                output_path,
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png'
            )
            
            # Clean up matplotlib figure
            plt.close(fig)
            
            # Verify file was created and get absolute path
            if not output_path.exists():
                raise IOError(f"Chart file was not created: {output_path}")
            
            absolute_path = output_path.resolve()
            print(f"Chart saved successfully: {absolute_path}")
            
            return str(absolute_path)
            
        except Exception as e:
            # Ensure figure cleanup on error
            plt.close(fig)
            # Cleanup temporary AssetBinder files
            self._cleanup_asset_binder_temp()
            raise IOError(f"Failed to save chart: {str(e)}")
    
    def generate_zoomed_demonstration_chart(
        self, 
        albedo_path: str, 
        normal_path: str,
        victim_id: int,
        attacker_id: int,
        output_path: str,
        zoom_factor: float = 10.0,
        zoom_region: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Generate a zoomed version of the demonstration chart to show noise patterns.
        
        Args:
            albedo_path: Path to original albedo texture
            normal_path: Path to original normal map
            victim_id: User ID for legitimate binding scenario
            attacker_id: User ID for attack scenario
            output_path: Output path for generated chart
            zoom_factor: Magnification factor for zoom (default 10x)
            zoom_region: Optional (x, y, width, height) region to zoom into
            
        Returns:
            Path to saved zoomed chart file
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If processing fails
        """
        try:
            # Step 1: Create chart layout with zoom title
            fig, axes = self._create_zoomed_chart_layout(zoom_factor)
            
            # Step 2: Process assets through binding pipeline (optimized for zoom)
            assets = self._process_assets_for_zoom(albedo_path, normal_path, victim_id, attacker_id, zoom_region)
            
            # Step 3: Generate renders for all scenarios (on cropped regions)
            renders = self._render_scenarios_cropped(assets)
            
            # Step 4: Calculate quality metrics (on full images for accuracy)
            metrics = self._calculate_metrics_fast(renders)
            
            # Step 5: Compose chart with zoomed images and overlays
            self._compose_zoomed_chart_optimized(fig, axes, assets, renders, metrics, zoom_factor)
            
            # Step 6: Save chart and cleanup
            saved_path = self._save_chart(fig, output_path)
            
            # Cleanup temporary AssetBinder files
            self._cleanup_asset_binder_temp()
            
            return saved_path
            
        except Exception as e:
            # Ensure matplotlib cleanup on error
            plt.close('all')
            # Cleanup temporary AssetBinder files
            self._cleanup_asset_binder_temp()
            raise ValueError(f"Zoomed chart generation failed: {str(e)}")
    
    def _process_assets_for_zoom(
        self, 
        albedo_path: str, 
        normal_path: str,
        victim_id: int, 
        attacker_id: int,
        zoom_region: Optional[Tuple[int, int, int, int]] = None
    ) -> AssetBundle:
        """
        Process assets optimized for zoom - crop regions early to speed up processing.
        
        Args:
            albedo_path: Path to original albedo texture
            normal_path: Path to original normal map
            victim_id: User ID for legitimate binding
            attacker_id: User ID for attack scenario
            zoom_region: Optional zoom region to optimize for
            
        Returns:
            AssetBundle containing cropped processed assets
        """
        # Validate input files exist
        if not os.path.exists(albedo_path):
            raise FileNotFoundError(f"Albedo texture not found: {albedo_path}")
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal map not found: {normal_path}")
        
        try:
            # Load original assets for reference
            print("Loading original textures...")
            original_albedo = self._load_texture_as_array(albedo_path)
            original_normal = self._load_texture_as_array(normal_path)
            
            # Determine zoom region if not provided
            if zoom_region is None:
                # Use a default center region for initial processing
                h, w = original_albedo.shape[:2]
                crop_size = min(h, w) // 4  # Use 1/4 of image for faster processing
                x = w // 2 - crop_size // 2
                y = h // 2 - crop_size // 2
                zoom_region = (x, y, crop_size, crop_size)
                print(f"Auto-selected zoom region: {zoom_region}")
            
            # Extract crop regions from originals for faster processing
            x, y, crop_w, crop_h = zoom_region
            
            # Expand crop region slightly for context (but keep it manageable)
            expand = min(50, min(crop_w, crop_h) // 4)  # Add small border
            x_exp = max(0, x - expand)
            y_exp = max(0, y - expand)
            w_exp = min(original_albedo.shape[1] - x_exp, crop_w + 2 * expand)
            h_exp = min(original_albedo.shape[0] - y_exp, crop_h + 2 * expand)
            
            # Crop original textures to expanded region
            crop_albedo = original_albedo[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
            crop_normal = original_normal[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
            
            print(f"Processing cropped region: {crop_albedo.shape} (from {original_albedo.shape})")
            
            # Save cropped textures temporarily
            temp_dir = Path("temp_zoom_crop")
            temp_dir.mkdir(exist_ok=True)
            
            crop_albedo_path = temp_dir / "crop_albedo.png"
            crop_normal_path = temp_dir / "crop_normal.png"
            
            self._save_array_as_image(crop_albedo, crop_albedo_path)
            self._save_array_as_image(crop_normal, crop_normal_path)
            
            # Process victim scenario (legitimate binding) on cropped region
            print(f"Processing legitimate binding scenario (victim_id: {victim_id}) on cropped region...")
            victim_result = self.asset_binder.bind_textures(
                albedo_path=str(crop_albedo_path),
                normal_path=str(crop_normal_path),
                user_id=victim_id,
                poison_strength=0.3,
                output_prefix="victim_crop"
            )
            
            # Load victim bound assets
            victim_albedo = self._load_texture_as_array(victim_result['output_paths']['albedo'])
            victim_normal = self._load_texture_as_array(victim_result['output_paths']['normal'])
            
            # Process attacker scenario (attack binding) on cropped region
            print(f"Processing attack binding scenario (attacker_id: {attacker_id}) on cropped region...")
            attacker_result = self.asset_binder.bind_textures(
                albedo_path=str(crop_albedo_path),
                normal_path=str(crop_normal_path),
                user_id=attacker_id,
                poison_strength=0.3,
                output_prefix="attacker_crop"
            )
            
            # Load attacker bound assets
            attacker_albedo = self._load_texture_as_array(attacker_result['output_paths']['albedo'])
            attacker_normal = self._load_texture_as_array(attacker_result['output_paths']['normal'])
            
            # Cleanup temporary crop files
            self._cleanup_temp_files(temp_dir)
            
            # Create and return asset bundle with cropped assets
            return AssetBundle(
                original_albedo=crop_albedo,
                original_normal=crop_normal,
                victim_albedo=victim_albedo,
                victim_normal=victim_normal,
                attacker_albedo=attacker_albedo,
                attacker_normal=attacker_normal
            )
            
        except Exception as e:
            # Cleanup on error
            if 'temp_dir' in locals():
                self._cleanup_temp_files(temp_dir)
            raise ValueError(f"Optimized asset processing failed: {str(e)}")
    
    def _render_scenarios_cropped(self, assets: AssetBundle) -> RenderResults:
        """
        Generate renders for cropped asset scenarios (faster than full resolution).
        
        Args:
            assets: AssetBundle containing cropped processed textures
            
        Returns:
            RenderResults containing rendered images
        """
        try:
            # Create temporary files for rendering
            temp_dir = Path("temp_render_crop")
            temp_dir.mkdir(exist_ok=True)
            
            # Save cropped assets for rendering
            original_albedo_path = temp_dir / "original_albedo_crop.png"
            original_normal_path = temp_dir / "original_normal_crop.png"
            self._save_array_as_image(assets.original_albedo, original_albedo_path)
            self._save_array_as_image(assets.original_normal, original_normal_path)
            
            victim_albedo_path = temp_dir / "victim_albedo_crop.png"
            victim_normal_path = temp_dir / "victim_normal_crop.png"
            self._save_array_as_image(assets.victim_albedo, victim_albedo_path)
            self._save_array_as_image(assets.victim_normal, victim_normal_path)
            
            attacker_normal_path = temp_dir / "attacker_normal_crop.png"
            self._save_array_as_image(assets.attacker_normal, attacker_normal_path)
            
            print("Rendering cropped scenarios...")
            
            # Render scenarios on cropped regions (much faster)
            original_render = self.render_simulator.render(
                str(original_albedo_path),
                str(original_normal_path),
                light_dir=[0, 0, 1]
            )
            
            legitimate_render = self.render_simulator.render(
                str(victim_albedo_path),
                str(victim_normal_path),
                light_dir=[0, 0, 1]
            )
            
            attack_render = self.render_simulator.render(
                str(victim_albedo_path),
                str(attacker_normal_path),
                light_dir=[0, 0, 1]
            )
            
            # Cleanup temporary files
            self._cleanup_temp_files(temp_dir)
            
            return RenderResults(
                original_render=original_render,
                legitimate_render=legitimate_render,
                attack_render=attack_render
            )
            
        except Exception as e:
            if 'temp_dir' in locals():
                self._cleanup_temp_files(temp_dir)
            raise ValueError(f"Cropped rendering failed: {str(e)}")
    
    def _calculate_metrics_fast(self, renders: RenderResults) -> QualityMetrics:
        """
        Calculate quality metrics quickly on cropped renders.
        
        Args:
            renders: RenderResults containing cropped rendered images
            
        Returns:
            QualityMetrics with calculated values
        """
        try:
            temp_dir = Path("temp_metrics_crop")
            temp_dir.mkdir(exist_ok=True)
            
            original_ref_path = temp_dir / "original_reference_crop.png"
            self._save_array_as_image(renders.original_render, original_ref_path)
            
            print("Calculating quality metrics on cropped regions...")
            
            # Calculate metrics (faster on smaller images)
            legitimate_psnr, legitimate_ssim = self.render_simulator.evaluate(
                str(original_ref_path),
                renders.legitimate_render
            )
            
            attack_psnr, attack_ssim = self.render_simulator.evaluate(
                str(original_ref_path),
                renders.attack_render
            )
            
            # Calculate deltas
            psnr_delta = legitimate_psnr - attack_psnr
            ssim_delta = legitimate_ssim - attack_ssim
            
            # Cleanup
            self._cleanup_temp_files(temp_dir)
            
            print(f"Quality metrics (cropped): Legitimate PSNR={legitimate_psnr:.2f}dB, Attack PSNR={attack_psnr:.2f}dB")
            
            return QualityMetrics(
                legitimate_psnr=legitimate_psnr,
                legitimate_ssim=legitimate_ssim,
                attack_psnr=attack_psnr,
                attack_ssim=attack_ssim,
                psnr_delta=psnr_delta,
                ssim_delta=ssim_delta
            )
            
        except Exception as e:
            if 'temp_dir' in locals():
                self._cleanup_temp_files(temp_dir)
            raise ValueError(f"Fast metrics calculation failed: {str(e)}")
    
    def _compose_zoomed_chart_optimized(
        self, 
        fig: plt.Figure, 
        axes: np.ndarray, 
        assets: AssetBundle, 
        renders: RenderResults, 
        metrics: QualityMetrics,
        zoom_factor: float
    ) -> None:
        """
        Compose chart with pre-cropped assets (no additional zoom extraction needed).
        
        Args:
            fig: Matplotlib figure
            axes: 2x3 array of subplot axes
            assets: AssetBundle with cropped processed textures
            renders: RenderResults with cropped rendered images
            metrics: QualityMetrics with calculated values
            zoom_factor: Magnification factor for display
        """
        # Since assets are already cropped, we can display them directly
        
        # Top row - Original assets and render
        axes[0,0].imshow(assets.original_albedo)
        axes[0,0].set_title(f'Original Albedo ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        normal_display = self._prepare_normal_for_display(assets.original_normal)
        axes[0,1].imshow(normal_display)
        axes[0,1].set_title(f'Original Normal Map ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        axes[0,2].imshow(renders.original_render)
        axes[0,2].set_title(f'Original Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Bottom row - Processed renders
        axes[1,0].imshow(renders.legitimate_render)
        axes[1,0].set_title(f'Legitimate Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        self._add_metrics_overlay(
            axes[1,0], 
            f"PSNR: {metrics.legitimate_psnr:.1f}dB\nSSIM: {metrics.legitimate_ssim:.3f}",
            color='white'
        )
        
        axes[1,1].imshow(renders.attack_render)
        axes[1,1].set_title(f'Attack Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        self._add_metrics_overlay(
            axes[1,1], 
            f"PSNR: {metrics.attack_psnr:.1f}dB\nSSIM: {metrics.attack_ssim:.3f}",
            color='white'
        )
        
        # Generate and display difference map
        difference_map = self._generate_difference_map(
            renders.original_render, 
            renders.attack_render
        )
        
        axes[1,2].imshow(difference_map)
        axes[1,2].set_title(f'Difference Map ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        self._add_metrics_overlay(
            axes[1,2], 
            f"Δ PSNR: {metrics.psnr_delta:.1f}dB\nΔ SSIM: {metrics.ssim_delta:.3f}\nCropped Region",
            color='cyan'
        )
        
        # Remove axis ticks and labels for all panels
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    def _create_zoomed_chart_layout(self, zoom_factor: float) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create 2x3 matplotlib figure and axes grid for zoomed visualization.
        
        Args:
            zoom_factor: Magnification factor for the zoom
            
        Returns:
            Tuple of (figure, axes_array) for chart composition
        """
        # Create figure with specified size and DPI
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Create 2x3 subplot grid
        axes = fig.subplots(2, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
        
        # Set main title with zoom indication
        fig.suptitle(
            f'Perceptual Interdependence: Noise Pattern Analysis ({zoom_factor:.0f}x Zoom)',
            fontsize=16,
            fontweight='bold',
            y=0.95
        )
        
        # Configure individual panel titles
        panel_titles = [
            f'Original Albedo ({zoom_factor:.0f}x)',
            f'Original Normal Map ({zoom_factor:.0f}x)', 
            f'Original Render ({zoom_factor:.0f}x)',
            f'Legitimate Render ({zoom_factor:.0f}x)',
            f'Attack Render ({zoom_factor:.0f}x)',
            f'Difference Map ({zoom_factor:.0f}x)'
        ]
        
        # Apply titles to each subplot
        for i, (ax, title) in enumerate(zip(axes.flat, panel_titles)):
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        return fig, axes
    
    def _select_optimal_zoom_region(
        self, 
        assets: AssetBundle, 
        renders: RenderResults
    ) -> Tuple[int, int, int, int]:
        """
        Automatically select an optimal region for zooming based on difference analysis.
        
        Args:
            assets: AssetBundle with all processed textures
            renders: RenderResults with rendered images
            
        Returns:
            Tuple of (x, y, width, height) for zoom region
        """
        # Calculate difference map to find interesting regions
        diff_map = self._calculate_difference_map(
            renders.original_render, 
            renders.attack_render, 
            method='l2'
        )
        
        # Find regions with significant differences
        threshold = 0.05  # Lower threshold for finding subtle differences
        significant_regions = diff_map > threshold
        
        if not np.any(significant_regions):
            # If no significant differences, use center region
            h, w = diff_map.shape
            zoom_size = min(h, w) // 10  # 10% of image size
            x = w // 2 - zoom_size // 2
            y = h // 2 - zoom_size // 2
            return (x, y, zoom_size, zoom_size)
        
        # Find the region with highest concentration of differences
        from scipy import ndimage
        try:
            # Use morphological operations to find connected regions
            labeled_regions, num_regions = ndimage.label(significant_regions)
            
            if num_regions > 0:
                # Find the largest connected region
                region_sizes = [(labeled_regions == i).sum() for i in range(1, num_regions + 1)]
                largest_region_idx = np.argmax(region_sizes) + 1
                largest_region = labeled_regions == largest_region_idx
                
                # Get bounding box of largest region
                coords = np.where(largest_region)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Expand region slightly and ensure minimum size
                h, w = diff_map.shape
                min_size = min(h, w) // 15  # Minimum zoom region size
                
                region_h = max(y_max - y_min, min_size)
                region_w = max(x_max - x_min, min_size)
                
                # Center the region and ensure it fits in image
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2
                
                y = max(0, min(center_y - region_h // 2, h - region_h))
                x = max(0, min(center_x - region_w // 2, w - region_w))
                
                return (x, y, region_w, region_h)
        except ImportError:
            pass
        
        # Fallback: use center region
        h, w = diff_map.shape
        zoom_size = min(h, w) // 10
        x = w // 2 - zoom_size // 2
        y = h // 2 - zoom_size // 2
        return (x, y, zoom_size, zoom_size)
    
    def _extract_zoom_region(
        self, 
        image: np.ndarray, 
        zoom_region: Tuple[int, int, int, int],
        zoom_factor: float
    ) -> np.ndarray:
        """
        Extract and interpolate a zoom region from an image.
        
        Args:
            image: Source image array
            zoom_region: (x, y, width, height) region to extract
            zoom_factor: Magnification factor
            
        Returns:
            Zoomed image region
        """
        x, y, w, h = zoom_region
        
        # Extract the region
        if len(image.shape) == 3:
            region = image[y:y+h, x:x+w, :]
        else:
            region = image[y:y+h, x:x+w]
        
        # Calculate target size
        target_h = int(h * zoom_factor)
        target_w = int(w * zoom_factor)
        
        # Use PIL for high-quality interpolation
        if len(region.shape) == 3:
            # RGB image
            region_uint8 = (np.clip(region, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(region_uint8, mode='RGB')
        else:
            # Grayscale image
            region_uint8 = (np.clip(region, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(region_uint8, mode='L')
        
        # Resize with high-quality interpolation (Lanczos)
        zoomed_pil = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        zoomed_array = np.array(zoomed_pil, dtype=np.float32) / 255.0
        
        return zoomed_array
    
    def _compose_zoomed_chart(
        self, 
        fig: plt.Figure, 
        axes: np.ndarray, 
        assets: AssetBundle, 
        renders: RenderResults, 
        metrics: QualityMetrics,
        zoom_region: Tuple[int, int, int, int],
        zoom_factor: float
    ) -> None:
        """
        Populate chart panels with zoomed images, metrics, and difference maps.
        
        Args:
            fig: Matplotlib figure
            axes: 2x3 array of subplot axes
            assets: AssetBundle with all processed textures
            renders: RenderResults with rendered images
            metrics: QualityMetrics with calculated values
            zoom_region: (x, y, width, height) region to zoom into
            zoom_factor: Magnification factor
        """
        # Panel layout: 
        # [0,0] Original Albedo    [0,1] Original Normal    [0,2] Original Render
        # [1,0] Legitimate Render  [1,1] Attack Render      [1,2] Difference Map
        
        # Extract zoomed regions for all images
        zoomed_original_albedo = self._extract_zoom_region(assets.original_albedo, zoom_region, zoom_factor)
        zoomed_original_normal = self._extract_zoom_region(assets.original_normal, zoom_region, zoom_factor)
        zoomed_original_render = self._extract_zoom_region(renders.original_render, zoom_region, zoom_factor)
        zoomed_legitimate_render = self._extract_zoom_region(renders.legitimate_render, zoom_region, zoom_factor)
        zoomed_attack_render = self._extract_zoom_region(renders.attack_render, zoom_region, zoom_factor)
        
        # Generate zoomed difference map
        zoomed_difference_map = self._generate_difference_map(
            zoomed_original_render, 
            zoomed_attack_render
        )
        
        # Top row - Original assets and render (zoomed)
        axes[0,0].imshow(zoomed_original_albedo)
        axes[0,0].set_title(f'Original Albedo ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Convert normal map for display (from [-1,1] to [0,1] if needed)
        normal_display = self._prepare_normal_for_display(zoomed_original_normal)
        axes[0,1].imshow(normal_display)
        axes[0,1].set_title(f'Original Normal Map ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        axes[0,2].imshow(zoomed_original_render)
        axes[0,2].set_title(f'Original Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Bottom row - Legitimate render with metrics (zoomed)
        axes[1,0].imshow(zoomed_legitimate_render)
        axes[1,0].set_title(f'Legitimate Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Add metrics overlay to legitimate render
        self._add_metrics_overlay(
            axes[1,0], 
            f"PSNR: {metrics.legitimate_psnr:.1f}dB\nSSIM: {metrics.legitimate_ssim:.3f}",
            color='white'
        )
        
        # Attack render with metrics (zoomed)
        axes[1,1].imshow(zoomed_attack_render)
        axes[1,1].set_title(f'Attack Render ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Add metrics overlay to attack render
        self._add_metrics_overlay(
            axes[1,1], 
            f"PSNR: {metrics.attack_psnr:.1f}dB\nSSIM: {metrics.attack_ssim:.3f}",
            color='white'
        )
        
        # Display the zoomed RGB difference map
        axes[1,2].imshow(zoomed_difference_map)
        axes[1,2].set_title(f'Difference Map ({zoom_factor:.0f}x)', fontsize=12, fontweight='bold')
        
        # Add delta metrics and zoom info overlay to difference map
        zoom_info = f"Region: {zoom_region[0]},{zoom_region[1]} ({zoom_region[2]}×{zoom_region[3]})"
        self._add_metrics_overlay(
            axes[1,2], 
            f"Δ PSNR: {metrics.psnr_delta:.1f}dB\nΔ SSIM: {metrics.ssim_delta:.3f}\n{zoom_info}",
            color='cyan'
        )
        
        # Remove axis ticks and labels for all panels
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    def _cleanup_asset_binder_temp(self) -> None:
        """Clean up temporary AssetBinder files."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
        except Exception:
            # Ignore cleanup errors
            pass