"""
Research Orchestrator for RGB Forensics Experimental Pipeline

This module implements the ResearchOrchestrator class that coordinates complete
experimental pipelines for validating the RGB forensics traitor tracing system.
The orchestrator manages asset generation, simulation scenarios, forensic analysis,
and comprehensive reporting for academic research and system validation.
"""

import numpy as np
from PIL import Image
from skimage import data, filters
from typing import Tuple, Dict, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.asset_binder_complex import AssetBinderComplex
from core.render_simulator import RenderSimulator
from core.rgb_forensics import RGBForensics
from utils.result_chart_generator import ResultChartGenerator


class ResearchOrchestrator:
    """
    Research orchestrator for comprehensive experimental pipeline validation.
    
    This class coordinates the complete experimental workflow including:
    - Dummy asset generation using skimage data
    - Asset binding for victim and attacker users
    - Simulation of legitimate and attack scenarios
    - Forensic analysis and traitor detection
    - Quality assessment and academic reporting
    """
    
    # Experimental constants
    VICTIM_USER_ID = 42      # Default victim user ID for experiments
    ATTACKER_USER_ID = 99    # Default attacker user ID for experiments
    NORMAL_FLAT_BLUE = [128, 128, 255]  # Flat blue normal map values
    
    def __init__(self):
        """
        Initialize the ResearchOrchestrator.
        
        Sets up the orchestrator without parameters, ready for experimental
        pipeline execution through method calls.
        """
        # Initialize component systems
        self.asset_binder = AssetBinderComplex()
        self.render_simulator = RenderSimulator()
        self.rgb_forensics = RGBForensics()
        self.chart_generator = ResultChartGenerator()
        
        # Storage for experimental assets and results
        self.original_albedo = None
        self.original_normal = None
        self.texture_name = None  # Track which texture is being used
        self.victim_assets = {}
        self.attacker_assets = {}
        self.experimental_results = {}
    
    def setup_experiment(self, texture_prefix: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up experimental environment with dynamic texture selection.
        
        Automatically detects available textures in assets/raw/ directory.
        Prioritizes velvet textures if available, falls back to brick textures.
        
        Args:
            texture_prefix (str, optional): Specific texture prefix to use (e.g., 'velour_velvet', 'church_bricks_03')
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Original albedo and normal map arrays
            
        Raises:
            RuntimeError: If asset loading fails
        """
        try:
            # Auto-detect available textures
            raw_dir = 'assets/raw'
            available_files = os.listdir(raw_dir)
            
            # Find texture pairs (diffuse + normal)
            texture_pairs = {}
            for file in available_files:
                if file.endswith(('.jpg', '.png')):
                    # Extract base name and type
                    if '_diff_' in file:
                        base = file.split('_diff_')[0]
                        texture_pairs[base] = texture_pairs.get(base, {})
                        texture_pairs[base]['albedo'] = file
                    elif '_nor_' in file:
                        base = file.split('_nor_')[0]
                        texture_pairs[base] = texture_pairs.get(base, {})
                        texture_pairs[base]['normal'] = file
            
            # Select texture pair
            selected_base = None
            if texture_prefix:
                if texture_prefix in texture_pairs:
                    selected_base = texture_prefix
                else:
                    print(f"Warning: Requested texture '{texture_prefix}' not found")
            
            if not selected_base:
                # Prioritize velvet, then brick
                if 'velour_velvet' in texture_pairs:
                    selected_base = 'velour_velvet'
                elif 'church_bricks_03' in texture_pairs:
                    selected_base = 'church_bricks_03'
                else:
                    selected_base = list(texture_pairs.keys())[0] if texture_pairs else None
            
            if not selected_base or 'albedo' not in texture_pairs[selected_base] or 'normal' not in texture_pairs[selected_base]:
                raise RuntimeError("No complete texture pairs found in assets/raw/")
            
            # Build paths
            albedo_path = os.path.join(raw_dir, texture_pairs[selected_base]['albedo'])
            normal_path = os.path.join(raw_dir, texture_pairs[selected_base]['normal'])
            
            print(f"Loading {selected_base} textures:")
            print(f"  Albedo: {albedo_path}")
            print(f"  Normal: {normal_path}")
            
            # Load albedo texture
            albedo_image = Image.open(albedo_path).convert('RGB')
            albedo_array = np.array(albedo_image, dtype=np.float32)
            
            # Convert to float32 and normalize to [0.0, 1.0] range
            self.original_albedo = albedo_array / 255.0
            
            # Load normal map texture
            normal_image = Image.open(normal_path).convert('RGB')
            normal_array = np.array(normal_image, dtype=np.float32)
            
            # Convert normal map from [0, 255] to [0.0, 1.0] range (keep as texture format)
            self.original_normal = normal_array / 255.0
            
            # Save original assets for reference
            self._save_original_assets()
            
            # Save texture name for reporting
            self.texture_name = selected_base
            
            print(f"{selected_base.replace('_', ' ').title()} texture loading complete:")
            print(f"  Original albedo shape: {self.original_albedo.shape}")
            print(f"  Original normal shape: {self.original_normal.shape}")
            print(f"  Assets saved as 'original_albedo.png' and 'original_normal.png'")
            
            return self.original_albedo, self.original_normal
            
        except Exception as e:
            raise RuntimeError(f"Failed to load textures: {str(e)}")
    
    def _save_original_assets(self) -> None:
        """
        Save original assets to disk for reference and processing.
        
        Raises:
            IOError: If file saving fails
        """
        try:
            # Convert albedo to uint8 for saving
            albedo_uint8 = (self.original_albedo * 255.0).astype(np.uint8)
            albedo_image = Image.fromarray(albedo_uint8, mode='RGB')
            albedo_image.save('original_albedo.png', format='PNG')
            
            # Convert normal to uint8 for saving
            normal_uint8 = (self.original_normal * 255.0).astype(np.uint8)
            normal_image = Image.fromarray(normal_uint8, mode='RGB')
            normal_image.save('original_normal.png', format='PNG')
            
        except Exception as e:
            raise IOError(f"Failed to save original assets: {str(e)}")
    
    def generate_test_assets(self, victim_id: int = None, attacker_id: int = None) -> None:
        """
        Generate bound assets for victim and attacker users using AssetBinderComplex.
        
        Coordinates asset generation with AssetBinderComplex for both victim and
        attacker users, creating the necessary test assets for experimental scenarios.
        
        Args:
            victim_id (int, optional): Victim user ID. Defaults to VICTIM_USER_ID (42).
            attacker_id (int, optional): Attacker user ID. Defaults to ATTACKER_USER_ID (99).
            
        Raises:
            RuntimeError: If asset generation fails
            ValueError: If experiment setup has not been completed
        """
        # Use default IDs if not provided
        if victim_id is None:
            victim_id = self.VICTIM_USER_ID
        if attacker_id is None:
            attacker_id = self.ATTACKER_USER_ID
        
        # Validate experiment setup
        if self.original_albedo is None or self.original_normal is None:
            raise ValueError("Experiment setup not completed. Call setup_experiment() first.")
        
        try:
            # Generate bound assets for victim user (ID 42)
            print(f"Generating assets for victim user ID: {victim_id}")
            self.asset_binder.bind_textures(
                clean_albedo_path='original_albedo.png',
                original_normal_path='original_normal.png',
                user_seed=victim_id,
                poison_strength=0.2
            )
            
            # Store victim asset paths
            self.victim_assets = {
                'albedo_path': f'bound_albedo_{victim_id}.png',
                'normal_path': f'bound_normal_{victim_id}.png',
                'user_id': victim_id
            }
            
            # Generate bound assets for attacker user (ID 99)
            print(f"Generating assets for attacker user ID: {attacker_id}")
            self.asset_binder.bind_textures(
                clean_albedo_path='original_albedo.png',
                original_normal_path='original_normal.png',
                user_seed=attacker_id,
                poison_strength=0.2
            )
            
            # Store attacker asset paths
            self.attacker_assets = {
                'albedo_path': f'bound_albedo_{attacker_id}.png',
                'normal_path': f'bound_normal_{attacker_id}.png',
                'user_id': attacker_id
            }
            
            print(f"Asset generation complete:")
            print(f"  Victim assets: {self.victim_assets['albedo_path']}, {self.victim_assets['normal_path']}")
            print(f"  Attacker assets: {self.attacker_assets['albedo_path']}, {self.attacker_assets['normal_path']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate test assets: {str(e)}")
    
    def run_simulation_scenarios(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run legitimate and attack simulation scenarios using RenderSimulator.
        
        Implements legitimate scenario (render victim user's complete asset pair)
        and attack scenario (render victim's albedo with attacker's normal map)
        as specified in requirements 4.5, 5.1, 5.2, 5.3, 5.4.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Legitimate and attack rendered images
            
        Raises:
            RuntimeError: If simulation fails
            ValueError: If assets have not been generated
        """
        # Validate assets are available
        if not self.victim_assets or not self.attacker_assets:
            raise ValueError("Test assets not generated. Call generate_test_assets() first.")
        
        try:
            # Implement legitimate scenario: render victim user's complete asset pair (ID 42)
            print("Running legitimate scenario: victim albedo + victim normal")
            legitimate_rendered = self.render_simulator.render(
                albedo_path=self.victim_assets['albedo_path'],
                normal_path=self.victim_assets['normal_path'],
                light_dir=[0, 0, 1]  # Default lighting direction
            )
            
            # Implement attack scenario: render victim's albedo with attacker's normal map (ID 99)
            print("Running attack scenario: victim albedo + attacker normal")
            attack_rendered = self.render_simulator.render(
                albedo_path=self.victim_assets['albedo_path'],
                normal_path=self.attacker_assets['normal_path'],
                light_dir=[0, 0, 1]  # Same lighting for fair comparison
            )
            
            # Save rendered outputs as fig_legit.png and fig_attack.png
            self._save_rendered_scenarios(legitimate_rendered, attack_rendered)
            
            # Calculate PSNR and SSIM quality metrics for both scenarios
            quality_metrics = self.calculate_quality_metrics(legitimate_rendered, attack_rendered)
            
            # Store metrics for later use
            self.experimental_results['quality_metrics'] = quality_metrics
            
            print("Simulation scenarios complete:")
            print(f"  Legitimate rendering saved as 'fig_legit.png'")
            print(f"  Attack rendering saved as 'fig_attack.png'")
            print(f"  Quality metrics calculated:")
            print(f"    Legitimate - PSNR: {quality_metrics['legitimate_psnr']:.2f} dB, SSIM: {quality_metrics['legitimate_ssim']:.4f}")
            print(f"    Attack - PSNR: {quality_metrics['attack_psnr']:.2f} dB, SSIM: {quality_metrics['attack_ssim']:.4f}")
            
            # Generate comprehensive result chart
            try:
                chart_path = self.chart_generator.create_comprehensive_chart(
                    original_albedo_path='original_albedo.png',
                    original_normal_path='original_normal.png', 
                    legitimate_render_path='fig_legit.png',
                    attack_render_path='fig_attack.png',
                    texture_name=self.texture_name or 'unknown',
                    quality_metrics=quality_metrics
                )
                print(f"  Comprehensive result chart saved as '{chart_path}'")
            except Exception as e:
                print(f"  Warning: Failed to generate result chart: {str(e)}")
            
            return legitimate_rendered, attack_rendered
            
        except Exception as e:
            raise RuntimeError(f"Failed to run simulation scenarios: {str(e)}")
    
    def _save_rendered_scenarios(self, legitimate: np.ndarray, attack: np.ndarray) -> None:
        """
        Save rendered scenario outputs as PNG files.
        
        Args:
            legitimate (np.ndarray): Legitimate scenario rendered image
            attack (np.ndarray): Attack scenario rendered image
            
        Raises:
            IOError: If file saving fails
        """
        try:
            # Convert legitimate rendering to uint8 and save as fig_legit.png
            legitimate_uint8 = (legitimate * 255.0).astype(np.uint8)
            legitimate_image = Image.fromarray(legitimate_uint8, mode='RGB')
            legitimate_image.save('fig_legit.png', format='PNG')
            
            # Convert attack rendering to uint8 and save as fig_attack.png
            attack_uint8 = (attack * 255.0).astype(np.uint8)
            attack_image = Image.fromarray(attack_uint8, mode='RGB')
            attack_image.save('fig_attack.png', format='PNG')
            
        except Exception as e:
            raise IOError(f"Failed to save rendered scenarios: {str(e)}")
    
    def perform_forensic_analysis(self) -> int:
        """
        Perform forensic analysis using RGBForensics class on victim's bound albedo.
        
        Passes victim user's bound albedo to forensic analysis system and validates
        that detected user ID matches expected victim ID (42) as specified in
        requirements 5.1, 5.2. Includes comprehensive detection accuracy validation
        and detailed error reporting.
        
        Returns:
            int: Detected user ID from forensic analysis
            
        Raises:
            RuntimeError: If forensic analysis fails
            ValueError: If victim assets are not available
            AssertionError: If detection accuracy validation fails
        """
        # Validate victim assets are available
        if not self.victim_assets:
            raise ValueError("Victim assets not available. Call generate_test_assets() first.")
        
        # Initialize forensic analysis results tracking
        forensic_results = {
            'signature_extracted': False,
            'correlation_completed': False,
            'visualization_generated': False,
            'detection_accurate': False,
            'error_details': None
        }
        
        try:
            # Extract signature from victim user's bound albedo
            print("Performing forensic analysis on victim's bound albedo")
            print(f"  Suspicious albedo: {self.victim_assets['albedo_path']}")
            print(f"  Original clean: original_albedo.png")
            
            signature = self.rgb_forensics.extract_signature(
                suspicious_albedo_path=self.victim_assets['albedo_path'],
                original_clean_path='original_albedo.png'
            )
            forensic_results['signature_extracted'] = True
            print(f"  Signature extracted: shape {signature.shape}, mean {np.mean(signature):.6f}, std {np.std(signature):.6f}")
            
            # Find traitor by correlating with known user patterns
            print("Correlating signature with known user patterns...")
            detected_user_id = self.rgb_forensics.find_traitor(
                signature=signature,
                max_users=100  # Test up to 100 users
            )
            forensic_results['correlation_completed'] = True
            
            # Generate forensic visualization
            print("Generating forensic visualization...")
            self.rgb_forensics.visualize_results()
            forensic_results['visualization_generated'] = True
            
            # Comprehensive detection accuracy validation
            expected_victim_id = self.victim_assets['user_id']
            detection_accurate = detected_user_id == expected_victim_id
            forensic_results['detection_accurate'] = detection_accurate
            
            # Detailed accuracy reporting
            print(f"\nForensic Analysis Results:")
            print(f"{'='*50}")
            print(f"  Expected victim ID: {expected_victim_id}")
            print(f"  Detected user ID: {detected_user_id}")
            print(f"  Detection accuracy: {'✓ PASS' if detection_accurate else '✗ FAIL'}")
            
            # Get correlation scores for detailed analysis
            if hasattr(self.rgb_forensics, '_last_correlation_scores'):
                scores = self.rgb_forensics._last_correlation_scores
                victim_score = scores[expected_victim_id] if expected_victim_id < len(scores) else None
                detected_score = scores[detected_user_id] if detected_user_id < len(scores) else 0.0
                
                if victim_score is not None:
                    print(f"  Victim correlation score: {victim_score:.6f}")
                    print(f"  Detected correlation score: {detected_score:.6f}")
                    print(f"  Score difference: {detected_score - victim_score:.6f}")
                else:
                    print(f"  Victim ID {expected_victim_id} not tested (max_users={len(scores)})")
                    print(f"  Detected correlation score: {detected_score:.6f}")
                
                # Additional accuracy metrics
                top_5_users = np.argsort(scores)[-5:][::-1]  # Top 5 highest scoring users
                if victim_score is not None:
                    victim_rank = np.where(np.argsort(scores)[::-1] == expected_victim_id)[0]
                    victim_rank = victim_rank[0] + 1 if len(victim_rank) > 0 else len(scores)
                    print(f"  Victim rank in results: {victim_rank}")
                else:
                    print(f"  Victim rank in results: Not tested")
                
                print(f"  Top 5 user IDs: {top_5_users.tolist()}")
                print(f"  Top 5 scores: {[f'{scores[uid]:.6f}' for uid in top_5_users]}")
            
            # Store results for experimental pipeline integration
            self.experimental_results['forensic_analysis'] = {
                'expected_victim_id': expected_victim_id,
                'detected_user_id': detected_user_id,
                'detection_accurate': detection_accurate,
                'signature_shape': signature.shape,
                'signature_stats': {
                    'mean': float(np.mean(signature)),
                    'std': float(np.std(signature)),
                    'min': float(np.min(signature)),
                    'max': float(np.max(signature))
                },
                'processing_status': forensic_results
            }
            
            # Assert detection accuracy as specified in requirements
            # Only assert if victim ID was actually tested
            max_users_tested = len(scores) if hasattr(self.rgb_forensics, '_last_correlation_scores') else 100
            if expected_victim_id >= max_users_tested:
                print(f"\n{'!'*50}")
                print(f"WARNING: VICTIM ID NOT TESTED")
                print(f"{'!'*50}")
                print(f"Victim ID {expected_victim_id} exceeds max_users limit ({max_users_tested})")
                print(f"Cannot validate detection accuracy - victim not in test range")
                
                # Store warning in results but don't fail
                self.experimental_results['forensic_analysis'] = {
                    'expected_victim_id': expected_victim_id,
                    'detected_user_id': detected_user_id,
                    'detection_accurate': None,  # Cannot determine
                    'victim_not_tested': True,
                    'max_users_tested': max_users_tested,
                    'signature_shape': signature.shape,
                    'signature_stats': {
                        'mean': float(np.mean(signature)),
                        'std': float(np.std(signature)),
                        'min': float(np.min(signature)),
                        'max': float(np.max(signature))
                    },
                    'processing_status': forensic_results
                }
                
                print(f"{'='*50}")
                print("⚠ Forensic analysis completed with warning")
                print("⚠ Detection accuracy cannot be validated")
                
                return detected_user_id
            
            if not detection_accurate:
                error_msg = (f"Forensic detection accuracy validation failed: "
                           f"expected victim ID {expected_victim_id}, "
                           f"but detected user ID {detected_user_id}")
                forensic_results['error_details'] = error_msg
                
                # Provide detailed error reporting
                print(f"\n{'!'*50}")
                print(f"DETECTION ACCURACY VALIDATION FAILED")
                print(f"{'!'*50}")
                print(f"Error: {error_msg}")
                
                if hasattr(self.rgb_forensics, '_last_correlation_scores'):
                    scores = self.rgb_forensics._last_correlation_scores
                    print(f"Diagnostic Information:")
                    print(f"  Total users tested: {len(scores)}")
                    print(f"  Highest score: {np.max(scores):.6f} (User {np.argmax(scores)})")
                    if expected_victim_id < len(scores):
                        print(f"  Victim score: {scores[expected_victim_id]:.6f} (User {expected_victim_id})")
                        print(f"  Score ratio: {scores[expected_victim_id] / np.max(scores):.4f}")
                    else:
                        print(f"  Victim ID {expected_victim_id} not tested (exceeds max_users={len(scores)})")
                
                raise AssertionError(error_msg)
            
            print(f"{'='*50}")
            print("✓ Forensic analysis validation: PASSED")
            print("✓ Detection accuracy verification: SUCCESSFUL")
            
            return detected_user_id
            
        except Exception as e:
            # Enhanced error reporting with forensic analysis context
            forensic_results['error_details'] = str(e)
            
            print(f"\n{'!'*50}")
            print(f"FORENSIC ANALYSIS ERROR")
            print(f"{'!'*50}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Processing status: {forensic_results}")
            
            # Store error information for experimental pipeline
            self.experimental_results['forensic_analysis'] = {
                'error_occurred': True,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'processing_status': forensic_results
            }
            
            if isinstance(e, AssertionError):
                raise  # Re-raise assertion errors for detection accuracy failures
            else:
                raise RuntimeError(f"Failed to perform forensic analysis: {str(e)}") 
    
    def calculate_quality_metrics(self, legitimate: np.ndarray, attack: np.ndarray) -> Dict[str, float]:
        """
        Calculate PSNR and SSIM quality metrics for both scenarios.
        
        Compares both legitimate and attack scenarios against the original albedo
        to assess quality degradation as specified in requirements 6.1, 6.2.
        
        Args:
            legitimate (np.ndarray): Legitimate scenario rendered image
            attack (np.ndarray): Attack scenario rendered image
            
        Returns:
            Dict[str, float]: Quality metrics for both scenarios
            
        Raises:
            RuntimeError: If quality calculation fails
            ValueError: If original assets are not available
        """
        # Validate original assets are available for comparison
        if self.original_albedo is None:
            raise ValueError("Original assets not available. Call setup_experiment() first.")
        
        try:
            # Calculate quality metrics for legitimate scenario against original albedo
            legitimate_psnr, legitimate_ssim = self.render_simulator.evaluate(
                clean_ref_path='original_albedo.png',
                rendered_img=legitimate
            )
            
            # Calculate quality metrics for attack scenario against original albedo
            attack_psnr, attack_ssim = self.render_simulator.evaluate(
                clean_ref_path='original_albedo.png',
                rendered_img=attack
            )
            
            # Store and return comprehensive metrics
            metrics = {
                'legitimate_psnr': float(legitimate_psnr),
                'legitimate_ssim': float(legitimate_ssim),
                'attack_psnr': float(attack_psnr),
                'attack_ssim': float(attack_ssim)
            }
            
            print("Quality metrics calculated:")
            print(f"  Legitimate - PSNR: {legitimate_psnr:.2f} dB, SSIM: {legitimate_ssim:.4f}")
            print(f"  Attack - PSNR: {attack_psnr:.2f} dB, SSIM: {attack_ssim:.4f}")
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate quality metrics: {str(e)}")
    
    def generate_report(self, metrics: Dict[str, float] = None) -> None:
        """
        Generate comprehensive experimental report with LaTeX-compatible formatting.
        
        Implements academic publication formatting with PSNR values in decibels
        and SSIM values for structural similarity assessment. Creates LaTeX-compatible
        table with proper formatting and includes experimental metadata and summary
        statistics as specified in requirements 6.3, 6.4, 6.5.
        
        Args:
            metrics (Dict[str, float], optional): Quality metrics dictionary.
                                                If None, uses stored experimental results.
            
        Raises:
            ValueError: If metrics dictionary is incomplete or unavailable
            RuntimeError: If report generation fails
        """
        # Use stored metrics if not provided
        if metrics is None:
            if 'quality_metrics' in self.experimental_results:
                metrics = self.experimental_results['quality_metrics']
            else:
                raise ValueError("No quality metrics available. Run simulation scenarios first.")
        
        # Validate metrics dictionary contains required keys
        required_keys = ['legitimate_psnr', 'legitimate_ssim', 'attack_psnr', 'attack_ssim']
        for key in required_keys:
            if key not in metrics:
                raise ValueError(f"Missing required metric: {key}")
        
        try:
            # Calculate and format PSNR values in decibels for quality assessment
            legitimate_psnr_db = float(metrics['legitimate_psnr'])
            attack_psnr_db = float(metrics['attack_psnr'])
            
            # Calculate and format SSIM values for structural similarity assessment
            legitimate_ssim = float(metrics['legitimate_ssim'])
            attack_ssim = float(metrics['attack_ssim'])
            
            # Generate comprehensive report header
            print("\n" + "="*70)
            print("RGB FORENSICS TRAITOR TRACING - EXPERIMENTAL REPORT")
            print("="*70)
            
            # Add formatted output printing with proper table structure
            print("\nQUALITY ASSESSMENT RESULTS")
            print("-" * 50)
            
            # Create LaTeX-compatible table with columns for Scenario, PSNR (dB), and SSIM
            print("┌─────────────┬───────────┬──────────┐")
            print("│ Scenario    │ PSNR (dB) │ SSIM     │")
            print("├─────────────┼───────────┼──────────┤")
            print(f"│ Legitimate  │ {legitimate_psnr_db:8.2f}  │ {legitimate_ssim:8.4f} │")
            print(f"│ Attack      │ {attack_psnr_db:8.2f}  │ {attack_ssim:8.4f} │")
            print("└─────────────┴───────────┴──────────┘")
            
            # Calculate quality degradation metrics
            psnr_delta = legitimate_psnr_db - attack_psnr_db
            ssim_delta = legitimate_ssim - attack_ssim
            psnr_degradation_percent = (psnr_delta / legitimate_psnr_db) * 100 if legitimate_psnr_db != 0 else 0
            ssim_degradation_percent = (ssim_delta / legitimate_ssim) * 100 if legitimate_ssim != 0 else 0
            
            print(f"\nQUALITY DEGRADATION ANALYSIS")
            print("-" * 30)
            print(f"  PSNR Delta: {psnr_delta:+.2f} dB ({psnr_degradation_percent:+.1f}%)")
            print(f"  SSIM Delta: {ssim_delta:+.4f} ({ssim_degradation_percent:+.1f}%)")
            print(f"  Quality Impact: {'SIGNIFICANT' if psnr_delta > 1.0 or ssim_delta > 0.01 else 'MINIMAL'}")
            
            # Include experimental metadata and summary statistics
            print(f"\nEXPERIMENTAL METADATA")
            print("-" * 25)
            
            # Get forensic analysis results if available
            forensic_results = self.experimental_results.get('forensic_analysis', {})
            detection_status = "PASSED" if forensic_results.get('detection_accurate', False) else "PENDING"
            detected_user = forensic_results.get('detected_user_id', 'N/A')
            expected_user = forensic_results.get('expected_victim_id', 'N/A')
            
            print(f"  Victim User ID: {self.victim_assets.get('user_id', expected_user)}")
            print(f"  Attacker User ID: {self.attacker_assets.get('user_id', 'N/A')}")
            print(f"  Detected User ID: {detected_user}")
            print(f"  Detection Accuracy: {detection_status}")
            print(f"  Original Asset Source: Real {self.texture_name.replace('_', ' ')} textures ({self.texture_name})")
            print(f"  Normal Map Source: Real {self.texture_name.replace('_', ' ')} normal map ({self.texture_name}_nor_gl)")
            print(f"  Asset Binding Strength: 0.2")
            print(f"  Rendering Light Direction: [0, 0, 1]")
            print(f"  Maximum Users Tested: 100")
            
            # Add signature analysis statistics if available
            if 'signature_stats' in forensic_results:
                sig_stats = forensic_results['signature_stats']
                print(f"\nSIGNATURE ANALYSIS STATISTICS")
                print("-" * 35)
                print(f"  Signature Shape: {forensic_results.get('signature_shape', 'N/A')}")
                print(f"  Mean Value: {sig_stats.get('mean', 0):.6f}")
                print(f"  Standard Deviation: {sig_stats.get('std', 0):.6f}")
                print(f"  Value Range: [{sig_stats.get('min', 0):.6f}, {sig_stats.get('max', 0):.6f}]")
            
            # Summary statistics and system status
            print(f"\nSYSTEM PERFORMANCE SUMMARY")
            print("-" * 30)
            print(f"  Forensic Detection: {detection_status}")
            print(f"  Quality Preservation: {'GOOD' if psnr_delta < 2.0 and ssim_delta < 0.05 else 'DEGRADED'}")
            print(f"  Traitor Identification: {'SUCCESSFUL' if detection_status == 'PASSED' else 'FAILED'}")
            print(f"  System Status: OPERATIONAL")
            print(f"  Pipeline Completion: {'FULL' if len(self.experimental_results) > 0 else 'PARTIAL'}")
            
            print("="*70)
            
            # Save comprehensive LaTeX table for academic publication
            self._save_latex_table(metrics, forensic_results)
            
            # Save formatted text report
            self._save_text_report(metrics, forensic_results)
            
            print(f"\nReport files generated:")
            print(f"  - forensics_results_table.tex (LaTeX table)")
            print(f"  - experimental_report.txt (Full text report)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate comprehensive report: {str(e)}")
    
    def _save_latex_table(self, metrics: Dict[str, float], forensic_results: Dict = None) -> None:
        """
        Save LaTeX-formatted table for academic publication with comprehensive formatting.
        
        Creates LaTeX-compatible table with proper academic formatting including
        scenario names, PSNR values in decibels, and SSIM values as specified
        in requirements 6.3, 6.4, 6.5.
        
        Args:
            metrics (Dict[str, float]): Quality metrics dictionary
            forensic_results (Dict, optional): Forensic analysis results for metadata
        """
        try:
            # Calculate quality degradation metrics for table caption
            psnr_delta = metrics['legitimate_psnr'] - metrics['attack_psnr']
            ssim_delta = metrics['legitimate_ssim'] - metrics['attack_ssim']
            
            # Get detection information if available
            detection_info = ""
            if forensic_results:
                detected_id = forensic_results.get('detected_user_id', 'N/A')
                expected_id = forensic_results.get('expected_victim_id', 'N/A')
                accuracy = forensic_results.get('detection_accurate', False)
                detection_info = f"Forensic detection: User {detected_id} (Expected: {expected_id}, Accuracy: {'Pass' if accuracy else 'Fail'})"
            
            # Create comprehensive LaTeX table with academic formatting
            latex_content = f"""% RGB Forensics Traitor Tracing - Quality Assessment Results
% Generated by ResearchOrchestrator.generate_report()

\\begin{{table}}[htbp]
\\centering
\\caption{{Quality Assessment Results for RGB Forensics Traitor Tracing System}}
\\label{{tab:rgb_forensics_quality_results}}
\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\textbf{{Scenario}} & \\textbf{{PSNR (dB)}} & \\textbf{{SSIM}} \\\\
\\hline
Legitimate & {metrics['legitimate_psnr']:.2f} & {metrics['legitimate_ssim']:.4f} \\\\
Attack & {metrics['attack_psnr']:.2f} & {metrics['attack_ssim']:.4f} \\\\
\\hline
\\textbf{{Degradation}} & \\textbf{{{psnr_delta:+.2f}}} & \\textbf{{{ssim_delta:+.4f}}} \\\\
\\hline
\\end{{tabular}}
\\\\[0.5em]
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
\\textbf{{Experimental Parameters:}} \\\\
Victim User ID: {self.victim_assets.get('user_id', 'N/A')}, 
Attacker User ID: {self.attacker_assets.get('user_id', 'N/A')} \\\\
Asset Source: Real {self.texture_name.replace('_', ' ')} textures ({self.texture_name}), 
Binding Strength: 0.2, 
Light Direction: [0,0,1] \\\\
{detection_info}
\\end{{minipage}}
\\end{{table}}

% Additional LaTeX table for detailed analysis
\\begin{{table}}[htbp]
\\centering
\\caption{{Detailed Quality Metrics and System Performance}}
\\label{{tab:rgb_forensics_detailed_metrics}}
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Legitimate}} & \\textbf{{Attack}} & \\textbf{{Delta}} \\\\
\\hline
PSNR (dB) & {metrics['legitimate_psnr']:.2f} & {metrics['attack_psnr']:.2f} & {psnr_delta:+.2f} \\\\
SSIM & {metrics['legitimate_ssim']:.4f} & {metrics['attack_ssim']:.4f} & {ssim_delta:+.4f} \\\\
Quality Impact & \\multicolumn{{2}}{{c|}}{{{'Significant' if psnr_delta > 1.0 or ssim_delta > 0.01 else 'Minimal'}}} & - \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            
            # Save LaTeX content to file
            with open('forensics_results_table.tex', 'w') as f:
                f.write(latex_content)
            
            print("LaTeX table saved as 'forensics_results_table.tex'")
            
        except Exception as e:
            print(f"Warning: Failed to save LaTeX table: {e}")
    
    def _save_text_report(self, metrics: Dict[str, float], forensic_results: Dict = None) -> None:
        """
        Save comprehensive text report for documentation and analysis.
        
        Args:
            metrics (Dict[str, float]): Quality metrics dictionary
            forensic_results (Dict, optional): Forensic analysis results
        """
        try:
            # Calculate derived metrics
            psnr_delta = metrics['legitimate_psnr'] - metrics['attack_psnr']
            ssim_delta = metrics['legitimate_ssim'] - metrics['attack_ssim']
            
            # Create comprehensive text report
            report_content = f"""RGB FORENSICS TRAITOR TRACING - EXPERIMENTAL REPORT
{'='*70}

QUALITY ASSESSMENT RESULTS
{'-'*50}
Scenario      | PSNR (dB) | SSIM
Legitimate    | {metrics['legitimate_psnr']:8.2f}  | {metrics['legitimate_ssim']:8.4f}
Attack        | {metrics['attack_psnr']:8.2f}  | {metrics['attack_ssim']:8.4f}
Degradation   | {psnr_delta:+8.2f}  | {ssim_delta:+8.4f}

QUALITY DEGRADATION ANALYSIS
{'-'*30}
PSNR Delta: {psnr_delta:+.2f} dB ({(psnr_delta/metrics['legitimate_psnr']*100):+.1f}%)
SSIM Delta: {ssim_delta:+.4f} ({(ssim_delta/metrics['legitimate_ssim']*100):+.1f}%)
Quality Impact: {'SIGNIFICANT' if psnr_delta > 1.0 or ssim_delta > 0.01 else 'MINIMAL'}

EXPERIMENTAL METADATA
{'-'*25}
Victim User ID: {self.victim_assets.get('user_id', 'N/A')}
Attacker User ID: {self.attacker_assets.get('user_id', 'N/A')}
Original Asset Source: Real {self.texture_name.replace('_', ' ')} textures ({self.texture_name})
Normal Map Source: Real {self.texture_name.replace('_', ' ')} normal map ({self.texture_name}_nor_gl)
Asset Binding Strength: 0.2
Rendering Light Direction: [0, 0, 1]
Maximum Users Tested: 100
"""
            
            # Add forensic analysis results if available
            if forensic_results:
                detection_status = "PASSED" if forensic_results.get('detection_accurate', False) else "FAILED"
                detected_user = forensic_results.get('detected_user_id', 'N/A')
                expected_user = forensic_results.get('expected_victim_id', 'N/A')
                
                report_content += f"""
FORENSIC ANALYSIS RESULTS
{'-'*30}
Expected User ID: {expected_user}
Detected User ID: {detected_user}
Detection Accuracy: {detection_status}
"""
                
                # Add signature statistics if available
                if 'signature_stats' in forensic_results:
                    sig_stats = forensic_results['signature_stats']
                    report_content += f"""
SIGNATURE ANALYSIS STATISTICS
{'-'*35}
Signature Shape: {forensic_results.get('signature_shape', 'N/A')}
Mean Value: {sig_stats.get('mean', 0):.6f}
Standard Deviation: {sig_stats.get('std', 0):.6f}
Value Range: [{sig_stats.get('min', 0):.6f}, {sig_stats.get('max', 0):.6f}]
"""
            
            # Add system performance summary
            report_content += f"""
SYSTEM PERFORMANCE SUMMARY
{'-'*30}
Forensic Detection: {'PASSED' if forensic_results and forensic_results.get('detection_accurate') else 'PENDING'}
Quality Preservation: {'GOOD' if psnr_delta < 2.0 and ssim_delta < 0.05 else 'DEGRADED'}
Traitor Identification: {'SUCCESSFUL' if forensic_results and forensic_results.get('detection_accurate') else 'FAILED'}
System Status: OPERATIONAL
Pipeline Completion: {'FULL' if len(self.experimental_results) > 0 else 'PARTIAL'}

{'='*70}
Report generated by RGB Forensics Research Orchestrator
"""
            
            # Save text report to file
            with open('experimental_report.txt', 'w') as f:
                f.write(report_content)
            
            print("Text report saved as 'experimental_report.txt'")
            
        except Exception as e:
            print(f"Warning: Failed to save text report: {e}")
    
    def run_complete_experiment(self) -> Dict[str, any]:
        """
        Run the complete experimental pipeline from setup through reporting.
        
        Orchestrates the entire experimental workflow including setup, asset generation,
        simulation, forensics, and reporting as specified in the requirements.
        
        Returns:
            Dict[str, any]: Complete experimental results
            
        Raises:
            RuntimeError: If any stage of the experiment fails
        """
        try:
            print("Starting complete RGB forensics experimental pipeline...")
            
            # Stage 1: Setup experiment with velvet textures
            print("\n[1/6] Setting up experiment...")
            self.setup_experiment('velour_velvet')
            
            # Stage 2: Generate test assets for victim and attacker
            print("\n[2/6] Generating test assets...")
            self.generate_test_assets()
            
            # Stage 3: Run simulation scenarios
            print("\n[3/6] Running simulation scenarios...")
            legitimate_rendered, attack_rendered = self.run_simulation_scenarios()
            
            # Stage 4: Calculate quality metrics
            print("\n[4/6] Calculating quality metrics...")
            metrics = self.calculate_quality_metrics(legitimate_rendered, attack_rendered)
            
            # Stage 5: Perform forensic analysis
            print("\n[5/6] Performing forensic analysis...")
            detected_user_id = self.perform_forensic_analysis()
            self._detection_passed = True  # Mark detection as passed for reporting
            
            # Stage 6: Generate comprehensive report
            print("\n[6/6] Generating experimental report...")
            self.generate_report()  # Use stored metrics from experimental_results
            
            # Compile complete results
            complete_results = {
                'setup_successful': True,
                'assets_generated': True,
                'simulation_completed': True,
                'quality_metrics': metrics,
                'forensic_detection': detected_user_id,
                'detection_accuracy': detected_user_id == self.VICTIM_USER_ID,
                'victim_user_id': self.VICTIM_USER_ID,
                'attacker_user_id': self.ATTACKER_USER_ID,
                'experiment_status': 'COMPLETED'
            }
            
            print(f"\nExperimental pipeline completed successfully!")
            print(f"All results saved to current directory.")
            
            return complete_results
            
        except Exception as e:
            print(f"\nExperimental pipeline failed: {str(e)}")
            raise RuntimeError(f"Complete experiment failed: {str(e)}")