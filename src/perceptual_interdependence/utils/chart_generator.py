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
                poison_strength=0.2,
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
                poison_strength=0.2,
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
        
        # Generate and display difference map
        difference_map = self._generate_difference_map(
            renders.original_render, 
            renders.attack_render
        )
        axes[1,2].imshow(difference_map, cmap='hot')
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
    
    def _generate_difference_map(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray
    ) -> np.ndarray:
        """
        Generate difference map between two images.
        
        Args:
            reference: Reference image
            comparison: Image to compare against reference
            
        Returns:
            Difference map with enhanced visualization
        """
        # Calculate absolute difference
        diff = np.abs(reference - comparison)
        
        # Convert to grayscale for difference visualization
        if len(diff.shape) == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        
        # Enhance differences for visibility
        diff_enhanced = np.power(diff_gray, 0.5)  # Gamma correction to enhance small differences
        
        # Normalize to [0,1] range
        if diff_enhanced.max() > 0:
            diff_enhanced = diff_enhanced / diff_enhanced.max()
        
        return diff_enhanced
    
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
    
    def _cleanup_asset_binder_temp(self) -> None:
        """Clean up temporary AssetBinder files."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
        except Exception:
            # Ignore cleanup errors
            pass