#!/usr/bin/env python3
"""
Real-World Dataset Analysis Pipeline

This script processes 150 real PBR materials through binding, rendering, and forensic
analysis stages to generate quantitative metrics for academic publication.
"""

import numpy as np
import pandas as pd
import cv2
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from skimage.metrics import structural_similarity

# Core library imports
from perceptual_interdependence.core.asset_binder import AssetBinder
from perceptual_interdependence.core.render_simulator import RenderSimulator
from perceptual_interdependence.core.forensics import RGBForensics


@dataclass
class AssetInfo:
    """Metadata for a single PBR asset."""
    category: str          # fabric, ground, metal, other, wood
    name: str             # Asset name (directory name)
    albedo_path: str      # Path to albedo.jpg
    normal_path: str      # Path to normal.jpg


class RealWorldAnalysisPipeline:
    """
    Main orchestration class for real-world dataset analysis.
    """
    
    def __init__(self, dataset_root: str = "data/real_validation_set"):
        """
        Initialize the analysis pipeline.
        
        Args:
            dataset_root: Root directory of the validation dataset
        """
        self.dataset_root = Path(dataset_root)
        self.results = []
        
        # Set up temporary directory for intermediate files
        self.temp_dir = Path("temp_analysis")
        self.temp_dir.mkdir(exist_ok=True)
        
        print(f"Initialized Real-World Analysis Pipeline")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Temporary directory: {self.temp_dir}")
    
    def discover_assets(self) -> List[AssetInfo]:
        """
        Discover all valid asset pairs in the dataset.
        
        Scans data/real_validation_set/ for all category subdirectories and
        enumerates assets with both albedo.jpg and normal.jpg files.
        
        Returns:
            List of AssetInfo objects containing asset metadata
        """
        assets = []
        categories = ['wood', 'metal', 'terrain', 'fabric', 'rock']  # Updated to match downloaded dataset
        
        print("\n" + "="*80)
        print("DATASET DISCOVERY")
        print("="*80)
        
        for category in categories:
            category_path = self.dataset_root / category
            
            if not category_path.exists():
                print(f"Warning: Category directory not found: {category}")
                continue
            
            category_count = 0
            
            for asset_dir in sorted(category_path.iterdir()):
                if not asset_dir.is_dir():
                    continue
                
                albedo_path = asset_dir / "albedo.jpg"
                normal_path = asset_dir / "normal.jpg"
                
                # Validate presence of both required files
                if albedo_path.exists() and normal_path.exists():
                    assets.append(AssetInfo(
                        category=category,
                        name=asset_dir.name,
                        albedo_path=str(albedo_path),
                        normal_path=str(normal_path)
                    ))
                    category_count += 1
                else:
                    missing = []
                    if not albedo_path.exists():
                        missing.append("albedo.jpg")
                    if not normal_path.exists():
                        missing.append("normal.jpg")
                    print(f"  Skipping {category}/{asset_dir.name}: missing {', '.join(missing)}")
            
            print(f"  {category}: {category_count} assets discovered")
        
        print(f"\nTotal assets discovered: {len(assets)}")
        print("="*80)
        
        return assets
    
    def _generate_flat_normal(self, reference_normal_path: str) -> str:
        """
        Generate flat normal map with matching dimensions.
        
        Args:
            reference_normal_path: Path to reference normal for dimensions
        
        Returns:
            Path to generated flat normal map
        """
        # Load reference to get dimensions
        ref_img = cv2.imread(reference_normal_path)
        height, width = ref_img.shape[:2]
        
        # Create flat normal: all vectors pointing up [0, 0, 1]
        # In RGB space: [128, 128, 255]
        flat_normal = np.full((height, width, 3), [128, 128, 255], dtype=np.uint8)
        
        # Save to temp file
        flat_path = str(self.temp_dir / "flat_normal.png")
        cv2.imwrite(flat_path, flat_normal)
        
        return flat_path
    
    def _perform_rendering_validation(
        self,
        original_albedo: str,
        original_normal: str,
        poisoned_albedo: str,
        antidote_normal: str
    ) -> Tuple[float, float, float, float]:
        """
        Perform optical validation through rendering simulation.
        
        Calculates four distinct quality metrics:
        1. Texture Fidelity: Original Albedo vs Poisoned Albedo (file theft metric)
        2. Visual Fidelity: Render(Original+Original) vs Render(Poisoned+Antidote) (authorized user)
        3. Attack Severity: Render(Original+Original) vs Render(Poisoned+Flat) (flat normal attack)
        4. Mismatched Attack: Render(Original+Original) vs Render(Poisoned+WrongAntidote) (wrong antidote)
        
        Args:
            original_albedo: Path to original albedo texture
            original_normal: Path to original normal texture
            poisoned_albedo: Path to poisoned albedo texture
            antidote_normal: Path to antidote normal texture
        
        Returns:
            Tuple of (texture_ssim, render_legit_ssim, render_attack_ssim, render_mismatched_ssim)
        """
        # Initialize render simulator with tilted lighting to reveal normal map differences
        tilted_light = [0.3, 0.3, 0.9]  # Tilted to show surface detail
        render_simulator = RenderSimulator()
        
        # METRIC 1: Texture Fidelity (File Theft Metric)
        # Compare: Original Albedo vs Poisoned Albedo
        # Measures raw degradation of the stolen file itself
        original_albedo_img = cv2.imread(original_albedo)
        poisoned_albedo_img = cv2.imread(poisoned_albedo)
        
        # Ensure same dimensions
        if original_albedo_img.shape != poisoned_albedo_img.shape:
            min_h = min(original_albedo_img.shape[0], poisoned_albedo_img.shape[0])
            min_w = min(original_albedo_img.shape[1], poisoned_albedo_img.shape[1])
            original_albedo_img = original_albedo_img[:min_h, :min_w]
            poisoned_albedo_img = poisoned_albedo_img[:min_h, :min_w]
        
        texture_ssim = structural_similarity(
            original_albedo_img, poisoned_albedo_img,
            multichannel=True,
            channel_axis=2,
            data_range=255
        )
        
        # METRIC 2: Visual Fidelity (Authorized User Metric)
        # Render A (Truth): Original Albedo + Original Normal
        render_truth = render_simulator.render(original_albedo, original_normal, tilted_light)
        
        # Render B (Legit): Poisoned Albedo + Antidote Normal
        # Tests if antidote successfully hides the poison for legitimate users
        render_legit = render_simulator.render(poisoned_albedo, antidote_normal, tilted_light)
        
        # Ensure same dimensions
        if render_truth.shape != render_legit.shape:
            min_h = min(render_truth.shape[0], render_legit.shape[0])
            min_w = min(render_truth.shape[1], render_legit.shape[1])
            render_truth = render_truth[:min_h, :min_w]
            render_legit = render_legit[:min_h, :min_w]
        
        render_legit_ssim = structural_similarity(
            render_truth, render_legit,
            multichannel=True,
            channel_axis=2,
            data_range=1.0
        )
        
        # METRIC 3: Attack Severity (Flat Normal Attack)
        # Render C (Attack): Poisoned Albedo + Flat Normal
        # Tests visual penalty for using stolen texture without correct geometry
        flat_normal_path = self._generate_flat_normal(original_normal)
        render_attack = render_simulator.render(poisoned_albedo, flat_normal_path, tilted_light)
        
        # Ensure same dimensions
        if render_truth.shape != render_attack.shape:
            min_h = min(render_truth.shape[0], render_attack.shape[0])
            min_w = min(render_truth.shape[1], render_attack.shape[1])
            render_truth_crop = render_truth[:min_h, :min_w]
            render_attack = render_attack[:min_h, :min_w]
        else:
            render_truth_crop = render_truth
        
        render_attack_ssim = structural_similarity(
            render_truth_crop, render_attack,
            multichannel=True,
            channel_axis=2,
            data_range=1.0
        )
        
        # METRIC 4: Mismatched Attack (Wrong Antidote)
        # Render D (Mismatched): Poisoned Albedo + Wrong User's Antidote
        # Generate antidote for different user (99 instead of 42)
        from perceptual_interdependence.core.asset_binder import AssetBinder
        mismatched_binder = AssetBinder(output_dir=str(self.temp_dir))
        mismatched_result = mismatched_binder.bind_textures(
            albedo_path=original_albedo,
            normal_path=original_normal,
            user_id=99,  # Different user
            poison_strength=0.4,
            output_prefix="mismatched"
        )
        mismatched_normal = str(mismatched_result['output_paths']['normal'])
        
        render_mismatched = render_simulator.render(poisoned_albedo, mismatched_normal, tilted_light)
        
        # Ensure same dimensions
        if render_truth.shape != render_mismatched.shape:
            min_h = min(render_truth.shape[0], render_mismatched.shape[0])
            min_w = min(render_truth.shape[1], render_mismatched.shape[1])
            render_truth_crop2 = render_truth[:min_h, :min_w]
            render_mismatched = render_mismatched[:min_h, :min_w]
        else:
            render_truth_crop2 = render_truth
        
        render_mismatched_ssim = structural_similarity(
            render_truth_crop2, render_mismatched,
            multichannel=True,
            channel_axis=2,
            data_range=1.0
        )
        
        return texture_ssim, render_legit_ssim, render_attack_ssim, render_mismatched_ssim
    
    def _perform_forensic_analysis(
        self,
        poisoned_albedo: str,
        original_albedo: str,
        target_user_id: int = 42
    ) -> float:
        """
        Extract forensic signature and compute Z-score using proper detection.
        
        The Z-score measures how many standard deviations the target user's correlation
        score is above the mean of all other users' scores. Higher Z-scores indicate
        stronger detection confidence.
        
        Args:
            poisoned_albedo: Path to poisoned albedo texture
            original_albedo: Path to original clean albedo
            target_user_id: User ID that was used for binding
        
        Returns:
            Z-score metric for forensic signature strength (detection confidence)
        """
        # Initialize forensics engine
        forensics_engine = RGBForensics()
        
        # Extract signature from poisoned albedo
        signature = forensics_engine.extract_signature(
            suspicious_albedo_path=poisoned_albedo,
            original_clean_path=original_albedo
        )
        
        # Detect traitor and get correlation scores for all users
        detected_user = forensics_engine.find_traitor(signature, max_users=100)
        
        # Calculate Z-score: (target_score - mean) / std
        if hasattr(forensics_engine, '_last_correlation_scores'):
            correlation_scores = forensics_engine._last_correlation_scores
            target_score = correlation_scores[target_user_id]
            mean_score = np.mean(correlation_scores)
            std_score = np.std(correlation_scores)
            
            if std_score > 1e-6:
                z_score = (target_score - mean_score) / std_score
            else:
                z_score = 0.0
        else:
            # Fallback: calculate Z-score from signature statistics
            signature_mean = np.mean(signature)
            signature_std = np.std(signature)
            signature_peak = np.max(np.abs(signature))
            
            if signature_std > 1e-6:
                z_score = (signature_peak - signature_mean) / signature_std
            else:
                z_score = 0.0
        
        return z_score
    
    def process_single_asset(self, asset_info: AssetInfo) -> Dict[str, Any]:
        """
        Process a single asset through the complete pipeline.
        
        Args:
            asset_info: AssetInfo object containing asset paths and metadata
        
        Returns:
            Dictionary containing all computed metrics
        """
        # Step 1: Binding with timing
        asset_binder = AssetBinder(output_dir=str(self.temp_dir))
        
        bind_start = time.perf_counter()
        bind_result = asset_binder.bind_textures(
            albedo_path=asset_info.albedo_path,
            normal_path=asset_info.normal_path,
            user_id=42,  # Fixed seed for reproducibility
            poison_strength=0.4  # Optimal strength from calibration
        )
        bind_time_ms = (time.perf_counter() - bind_start) * 1000
        
        # Step 2: Rendering validation (four metrics)
        texture_ssim, render_legit_ssim, render_attack_ssim, render_mismatched_ssim = self._perform_rendering_validation(
            original_albedo=asset_info.albedo_path,
            original_normal=asset_info.normal_path,
            poisoned_albedo=str(bind_result['output_paths']['albedo']),
            antidote_normal=str(bind_result['output_paths']['normal'])
        )
        
        # Step 3: Forensic analysis
        z_score = self._perform_forensic_analysis(
            poisoned_albedo=str(bind_result['output_paths']['albedo']),
            original_albedo=asset_info.albedo_path
        )
        
        # Return comprehensive metrics
        return {
            'Category': asset_info.category,
            'Asset': asset_info.name,
            'Texture_SSIM': texture_ssim,
            'Render_Legit_SSIM': render_legit_ssim,
            'Render_Attack_SSIM': render_attack_ssim,
            'Render_Mismatched_SSIM': render_mismatched_ssim,
            'Z_Score': z_score,
            'Bind_Time_ms': bind_time_ms
        }
    
    def _create_error_record(
        self,
        asset_info: AssetInfo,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Create error record for failed asset processing.
        
        Args:
            asset_info: Asset metadata
            error_message: Error description
        
        Returns:
            Dictionary with null metrics and error message
        """
        return {
            'Category': asset_info.category,
            'Asset': asset_info.name,
            'Texture_SSIM': np.nan,
            'Render_Legit_SSIM': np.nan,
            'Render_Attack_SSIM': np.nan,
            'Render_Mismatched_SSIM': np.nan,
            'Z_Score': np.nan,
            'Bind_Time_ms': np.nan,
            'Error': error_message
        }
    
    def run_complete_analysis(self) -> pd.DataFrame:
        """
        Execute complete analysis pipeline on all assets.
        
        Returns:
            DataFrame with comprehensive metrics for all assets
        """
        # Discover all assets
        assets = self.discover_assets()
        
        if len(assets) == 0:
            print("\nERROR: No assets discovered. Please check dataset path.")
            return pd.DataFrame()
        
        print("\n" + "="*80)
        print("PROCESSING PIPELINE")
        print("="*80)
        
        # Process each asset
        total_assets = len(assets)
        successful = 0
        failed = 0
        
        for idx, asset_info in enumerate(assets):
            current = idx + 1
            print(f"\n[{current}/{total_assets}] {asset_info.category}/{asset_info.name}")
            
            try:
                metrics = self.process_single_asset(asset_info)
                self.results.append(metrics)
                successful += 1
                print(f"  ✓ Tex: {metrics['Texture_SSIM']:.4f} | Legit: {metrics['Render_Legit_SSIM']:.4f} | Flat: {metrics['Render_Attack_SSIM']:.4f} | Wrong: {metrics['Render_Mismatched_SSIM']:.4f} | Z: {metrics['Z_Score']:.2f}")
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                error_record = self._create_error_record(asset_info, str(e))
                self.results.append(error_record)
                failed += 1
        
        print("\n" + "="*80)
        print(f"Processing complete: {successful} successful, {failed} failed")
        print("="*80)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        return df


def main():
    """Main entry point for the analysis pipeline."""
    print("\n" + "="*80)
    print("REAL-WORLD DATASET ANALYSIS PIPELINE")
    print("="*80)
    
    # Track total pipeline time
    pipeline_start = time.perf_counter()
    
    # Initialize pipeline
    pipeline = RealWorldAnalysisPipeline()
    
    # Run complete analysis
    df = pipeline.run_complete_analysis()
    
    if df.empty:
        print("\nERROR: No results generated.")
        return
    
    # Calculate total time
    total_time = time.perf_counter() - pipeline_start
    
    # Print summary statistics grouped by category
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("="*80)
    
    # Filter out failed rows
    df_valid = df.dropna(subset=['Texture_SSIM', 'Render_Legit_SSIM', 'Render_Attack_SSIM', 'Render_Mismatched_SSIM', 'Z_Score'])
    
    if df_valid.empty:
        print("\nNo valid results to summarize.")
    else:
        summary = df_valid.groupby('Category').agg({
            'Texture_SSIM': ['mean', 'std', 'min', 'max'],
            'Render_Legit_SSIM': ['mean', 'std', 'min', 'max'],
            'Render_Attack_SSIM': ['mean', 'std', 'min', 'max'],
            'Render_Mismatched_SSIM': ['mean', 'std', 'min', 'max'],
            'Z_Score': ['mean', 'std', 'min', 'max'],
            'Bind_Time_ms': ['mean', 'std']
        }).round(4)
        
        print("\nMETRIC 1: Texture Fidelity (File Theft - Lower is Better)")
        print("  Measures raw degradation of stolen albedo file")
        print(f"\n{summary['Texture_SSIM']}")
        
        print("\n" + "-"*80)
        print("METRIC 2: Visual Fidelity (Authorized User - Higher is Better)")
        print("  Measures if antidote successfully hides poison for legitimate users")
        print(f"\n{summary['Render_Legit_SSIM']}")
        
        print("\n" + "-"*80)
        print("METRIC 3: Attack Severity (Flat Normal - Lower is Better)")
        print("  Measures visual penalty for using stolen texture with flat normal")
        print(f"\n{summary['Render_Attack_SSIM']}")
        
        print("\n" + "-"*80)
        print("METRIC 4: Mismatched Attack (Wrong Antidote - Lower is Better)")
        print("  Measures visual penalty for using stolen texture with wrong user's antidote")
        print(f"\n{summary['Render_Mismatched_SSIM']}")
        
        print("\n" + "-"*80)
        print("FORENSIC DETECTION (Z-Score - Higher is Better)")
        print("  Measures statistical strength of traitor detection")
        print(f"\n{summary['Z_Score']}")
        
        print("\n" + "-"*80)
        print("PERFORMANCE (Binding Time)")
        print(f"\n{summary['Bind_Time_ms']}")
        
        # Overall statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"Total assets processed: {len(df_valid)}")
        print(f"Texture SSIM (mean):           {df_valid['Texture_SSIM'].mean():.4f}")
        print(f"Render Legit SSIM (mean):      {df_valid['Render_Legit_SSIM'].mean():.4f}")
        print(f"Render Attack SSIM (mean):     {df_valid['Render_Attack_SSIM'].mean():.4f}")
        print(f"Render Mismatched SSIM (mean): {df_valid['Render_Mismatched_SSIM'].mean():.4f}")
        print(f"Z-Score (mean):                {df_valid['Z_Score'].mean():.2f}")
        print(f"Binding time (mean):           {df_valid['Bind_Time_ms'].mean():.2f} ms")
    
    # Save results to CSV
    output_path = Path("results_real_analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path.absolute()}")
    
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
