#!/usr/bin/env python3
"""
RGB Forensics Traitor Tracing - Main Experimental Script

This script implements the complete experimental pipeline for the RGB forensics
traitor tracing system. It coordinates all modules together to run comprehensive
experiments from setup through reporting, including asset generation, rendering
simulation, forensic analysis, and academic reporting.

Usage:
    python run_paper_experiment.py [options]

Example:
    python run_paper_experiment.py --victim-id 42 --attacker-id 99 --poison-strength 0.2
"""

import argparse
import sys
import os
import traceback
from typing import Dict, Any, Optional
import time

# Import all required modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.research_orchestrator import ResearchOrchestrator
from core.rgb_forensics import RGBForensics
from core.asset_binder_complex import AssetBinderComplex
from core.render_simulator import RenderSimulator
from core.render_simulator import RenderSimulator


class ExperimentalPipeline:
    """
    Main experimental pipeline coordinator for RGB forensics traitor tracing.
    
    This class orchestrates the complete experimental workflow including:
    - Command-line interface for experimental parameters
    - Error handling and progress reporting throughout pipeline
    - Coordination of all modules (ResearchOrchestrator, RGBForensics, etc.)
    - Comprehensive logging and status reporting
    """
    
    def __init__(self):
        """Initialize the experimental pipeline coordinator."""
        self.orchestrator = ResearchOrchestrator()
        self.start_time = None
        self.experiment_results = {}
        
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments for experimental parameters.
        
        Returns:
            argparse.Namespace: Parsed command-line arguments
        """
        parser = argparse.ArgumentParser(
            description='RGB Forensics Traitor Tracing - Main Experimental Script',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python run_paper_experiment.py
  python run_paper_experiment.py --victim-id 42 --attacker-id 99
  python run_paper_experiment.py --poison-strength 0.3 --max-users 50
  python run_paper_experiment.py --light-dir 0.5 0.5 1.0 --verbose
            """
        )
        
        # Experimental parameters
        parser.add_argument(
            '--victim-id', 
            type=int, 
            default=42,
            help='Victim user ID for asset generation (default: 42)'
        )
        
        parser.add_argument(
            '--attacker-id', 
            type=int, 
            default=99,
            help='Attacker user ID for asset generation (default: 99)'
        )
        
        parser.add_argument(
            '--poison-strength', 
            type=float, 
            default=0.2,
            help='Poison strength for asset binding [0.0, 1.0] (default: 0.2)'
        )
        
        parser.add_argument(
            '--max-users', 
            type=int, 
            default=100,
            help='Maximum number of users to test in forensics (default: 100)'
        )
        
        parser.add_argument(
            '--light-dir', 
            nargs=3, 
            type=float, 
            default=[0.0, 0.0, 1.0],
            help='Light direction vector [x y z] (default: 0 0 1)'
        )
        
        # Output and control options
        parser.add_argument(
            '--output-dir', 
            type=str, 
            default='.',
            help='Output directory for results (default: current directory)'
        )
        
        parser.add_argument(
            '--verbose', 
            action='store_true',
            help='Enable verbose progress reporting'
        )
        
        parser.add_argument(
            '--skip-forensics', 
            action='store_true',
            help='Skip forensic analysis (for testing rendering only)'
        )
        
        parser.add_argument(
            '--skip-visualization', 
            action='store_true',
            help='Skip forensic visualization generation'
        )
        
        return parser.parse_args()
    
    def validate_arguments(self, args: argparse.Namespace) -> None:
        """
        Validate command-line arguments and experimental parameters.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
            
        Raises:
            ValueError: If arguments are invalid
        """
        # Validate user IDs
        if args.victim_id < 0:
            raise ValueError(f"Victim ID must be non-negative, got {args.victim_id}")
        if args.attacker_id < 0:
            raise ValueError(f"Attacker ID must be non-negative, got {args.attacker_id}")
        if args.victim_id == args.attacker_id:
            raise ValueError(f"Victim and attacker IDs must be different: {args.victim_id}")
        
        # Validate poison strength
        if not (0.0 <= args.poison_strength <= 1.0):
            raise ValueError(f"Poison strength must be in range [0.0, 1.0], got {args.poison_strength}")
        
        # Validate max users against victim/attacker IDs
        if args.max_users <= max(args.victim_id, args.attacker_id):
            print(f"Warning: max_users ({args.max_users}) is less than or equal to victim ID ({args.victim_id}) or attacker ID ({args.attacker_id})")
            print(f"This means the victim/attacker may not be tested in forensic analysis")
            print(f"Consider increasing --max-users to at least {max(args.victim_id, args.attacker_id) + 1}")
            
            # Ask user if they want to continue
            import sys
            if not sys.stdin.isatty():  # Non-interactive mode
                print(f"Continuing in non-interactive mode...")
            else:
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    raise ValueError("Experiment cancelled by user")
        
        # Validate light direction
        if len(args.light_dir) != 3:
            raise ValueError(f"Light direction must have 3 components, got {len(args.light_dir)}")
        
        # Check if light direction is zero vector
        import numpy as np
        light_magnitude = np.linalg.norm(args.light_dir)
        if light_magnitude < 1e-6:
            raise ValueError(f"Light direction cannot be zero vector: {args.light_dir}")
        
        # Validate output directory
        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create output directory '{args.output_dir}': {e}")
        
        if not os.access(args.output_dir, os.W_OK):
            raise ValueError(f"Output directory '{args.output_dir}' is not writable")
    
    def setup_experiment_environment(self, args: argparse.Namespace) -> None:
        """
        Set up experimental environment and change to output directory.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
        """
        # Change to output directory
        if args.output_dir != '.':
            original_dir = os.getcwd()
            os.chdir(args.output_dir)
            if args.verbose:
                print(f"Changed to output directory: {args.output_dir}")
                print(f"Original directory: {original_dir}")
        
        # Set up experiment parameters in orchestrator
        self.orchestrator.VICTIM_USER_ID = args.victim_id
        self.orchestrator.ATTACKER_USER_ID = args.attacker_id
        
        if args.verbose:
            print(f"Experimental parameters configured:")
            print(f"  Victim User ID: {args.victim_id}")
            print(f"  Attacker User ID: {args.attacker_id}")
            print(f"  Poison Strength: {args.poison_strength}")
            print(f"  Max Users: {args.max_users}")
            print(f"  Light Direction: {args.light_dir}")
    
    def run_complete_experimental_flow(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Execute the complete experimental flow from setup through reporting.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments
            
        Returns:
            Dict[str, Any]: Complete experimental results
            
        Raises:
            RuntimeError: If any stage of the experiment fails
        """
        try:
            self.start_time = time.time()
            
            print("="*70)
            print("RGB FORENSICS TRAITOR TRACING - EXPERIMENTAL PIPELINE")
            print("="*70)
            
            # Stage 1: Setup experiment with real velvet textures
            print("\n[STAGE 1/6] Setting up experiment with real velvet textures...")
            if args.verbose:
                print("  - Loading real velvet albedo: velour_velvet_diff_2k.jpg")
                print("  - Loading real velvet normal map: velour_velvet_nor_gl_2k.jpg")
            
            original_albedo, original_normal = self.orchestrator.setup_experiment('velour_velvet')
            
            print(f"✓ Experiment setup completed")
            if args.verbose:
                print(f"  - Original albedo shape: {original_albedo.shape}")
                print(f"  - Original normal shape: {original_normal.shape}")
            
            # Stage 2: Generate bound assets for victim and attacker users
            print(f"\n[STAGE 2/6] Generating bound assets...")
            if args.verbose:
                print(f"  - Generating assets for User {args.victim_id} (Victim)")
                print(f"  - Generating assets for User {args.attacker_id} (Attacker)")
                print(f"  - Using poison strength: {args.poison_strength}")
            
            # Override poison strength in asset binder
            original_bind_method = self.orchestrator.asset_binder.bind_textures
            def bind_with_custom_strength(*bind_args, **bind_kwargs):
                bind_kwargs['poison_strength'] = args.poison_strength
                return original_bind_method(*bind_args, **bind_kwargs)
            self.orchestrator.asset_binder.bind_textures = bind_with_custom_strength
            
            self.orchestrator.generate_test_assets(
                victim_id=args.victim_id,
                attacker_id=args.attacker_id
            )
            
            print(f"✓ Asset generation completed")
            if args.verbose:
                print(f"  - Victim assets: bound_albedo_{args.victim_id}.png, bound_normal_{args.victim_id}.png")
                print(f"  - Attacker assets: bound_albedo_{args.attacker_id}.png, bound_normal_{args.attacker_id}.png")
            
            # Stage 3: Run simulation scenarios (legitimate and attack)
            print(f"\n[STAGE 3/6] Running simulation scenarios...")
            if args.verbose:
                print(f"  - Legitimate scenario: victim albedo + victim normal")
                print(f"  - Attack scenario: victim albedo + attacker normal")
                print(f"  - Light direction: {args.light_dir}")
            
            # Override light direction in render simulator
            original_render_method = self.orchestrator.render_simulator.render
            def render_with_custom_light(*render_args, **render_kwargs):
                render_kwargs['light_dir'] = args.light_dir
                return original_render_method(*render_args, **render_kwargs)
            self.orchestrator.render_simulator.render = render_with_custom_light
            
            legitimate_rendered, attack_rendered = self.orchestrator.run_simulation_scenarios()
            
            print(f"✓ Simulation scenarios completed")
            if args.verbose:
                print(f"  - Legitimate rendering saved as 'fig_legit.png'")
                print(f"  - Attack rendering saved as 'fig_attack.png'")
            
            # Stage 4: Calculate quality metrics (PSNR and SSIM)
            print(f"\n[STAGE 4/6] Calculating quality metrics...")
            if args.verbose:
                print(f"  - Computing PSNR and SSIM for both scenarios")
                print(f"  - Comparing against original albedo reference")
            
            quality_metrics = self.orchestrator.calculate_quality_metrics(
                legitimate_rendered, attack_rendered
            )
            
            print(f"✓ Quality metrics calculated")
            if args.verbose:
                print(f"  - Legitimate PSNR: {quality_metrics['legitimate_psnr']:.2f} dB")
                print(f"  - Attack PSNR: {quality_metrics['attack_psnr']:.2f} dB")
                print(f"  - Legitimate SSIM: {quality_metrics['legitimate_ssim']:.4f}")
                print(f"  - Attack SSIM: {quality_metrics['attack_ssim']:.4f}")
            
            # Stage 5: Perform forensic analysis and traitor detection
            detected_user_id = None
            if not args.skip_forensics:
                print(f"\n[STAGE 5/6] Performing forensic analysis...")
                if args.verbose:
                    print(f"  - Extracting signature from victim's bound albedo")
                    print(f"  - Testing correlation against {args.max_users} users")
                    print(f"  - Expected detection: User {args.victim_id}")
                
                # Override max_users in forensics
                original_find_method = self.orchestrator.rgb_forensics.find_traitor
                def find_with_custom_max(*find_args, **find_kwargs):
                    find_kwargs['max_users'] = args.max_users
                    return original_find_method(*find_args, **find_kwargs)
                self.orchestrator.rgb_forensics.find_traitor = find_with_custom_max
                
                # Skip visualization if requested
                if args.skip_visualization:
                    original_visualize_method = self.orchestrator.rgb_forensics.visualize_results
                    def skip_visualization(*vis_args, **vis_kwargs):
                        print("  - Skipping forensic visualization as requested")
                        return None
                    self.orchestrator.rgb_forensics.visualize_results = skip_visualization
                
                detected_user_id = self.orchestrator.perform_forensic_analysis()
                
                print(f"✓ Forensic analysis completed")
                if args.verbose:
                    detection_status = "PASSED" if detected_user_id == args.victim_id else "FAILED"
                    print(f"  - Detected User ID: {detected_user_id}")
                    print(f"  - Expected User ID: {args.victim_id}")
                    print(f"  - Detection Status: {detection_status}")
                    if not args.skip_visualization:
                        print(f"  - Forensic visualization saved as 'forensic_report.png'")
            else:
                print(f"\n[STAGE 5/6] Skipping forensic analysis as requested...")
                detected_user_id = None
            
            # Stage 6: Generate comprehensive experimental report
            print(f"\n[STAGE 6/6] Generating experimental report...")
            if args.verbose:
                print(f"  - Creating LaTeX-compatible tables")
                print(f"  - Generating comprehensive text report")
                print(f"  - Including experimental metadata and statistics")
            
            self.orchestrator.generate_report(quality_metrics)
            
            print(f"✓ Experimental report generated")
            if args.verbose:
                print(f"  - LaTeX table saved as 'forensics_results_table.tex'")
                print(f"  - Text report saved as 'experimental_report.txt'")
            
            # Compile comprehensive results
            execution_time = time.time() - self.start_time
            
            experiment_results = {
                'execution_successful': True,
                'execution_time_seconds': execution_time,
                'victim_user_id': args.victim_id,
                'attacker_user_id': args.attacker_id,
                'poison_strength': args.poison_strength,
                'max_users_tested': args.max_users,
                'light_direction': args.light_dir,
                'quality_metrics': quality_metrics,
                'forensic_detection': {
                    'detected_user_id': detected_user_id,
                    'expected_user_id': args.victim_id,
                    'detection_accurate': detected_user_id == args.victim_id if detected_user_id is not None else None,
                    'analysis_skipped': args.skip_forensics
                },
                'output_files': {
                    'original_assets': ['original_albedo.png', 'original_normal.png'],
                    'bound_assets': [
                        f'bound_albedo_{args.victim_id}.png',
                        f'bound_normal_{args.victim_id}.png',
                        f'bound_albedo_{args.attacker_id}.png',
                        f'bound_normal_{args.attacker_id}.png'
                    ],
                    'rendered_scenarios': ['fig_legit.png', 'fig_attack.png'],
                    'result_charts': [f'figures/{self.orchestrator.texture_name or "unknown"}_comprehensive_results.png'],
                    'forensic_visualization': [] if args.skip_visualization or args.skip_forensics else ['forensic_report.png'],
                    'reports': ['forensics_results_table.tex', 'experimental_report.txt']
                },
                'experiment_status': 'COMPLETED'
            }
            
            return experiment_results
            
        except Exception as e:
            execution_time = time.time() - self.start_time if self.start_time else 0
            print(f"\n✗ Experimental pipeline failed after {execution_time:.1f} seconds")
            print(f"Error: {str(e)}")
            
            if args.verbose:
                print(f"\nDetailed error traceback:")
                traceback.print_exc()
            
            # Return partial results for debugging
            return {
                'execution_successful': False,
                'execution_time_seconds': execution_time,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'experiment_status': 'FAILED'
            }
    
    def print_final_summary(self, results: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Print final experimental summary and results.
        
        Args:
            results (Dict[str, Any]): Complete experimental results
            args (argparse.Namespace): Command-line arguments
        """
        print("\n" + "="*70)
        print("EXPERIMENTAL PIPELINE SUMMARY")
        print("="*70)
        
        if results['execution_successful']:
            print(f"✓ Status: COMPLETED SUCCESSFULLY")
            print(f"✓ Execution Time: {results['execution_time_seconds']:.1f} seconds")
            
            # Quality metrics summary
            if 'quality_metrics' in results:
                metrics = results['quality_metrics']
                psnr_delta = metrics['legitimate_psnr'] - metrics['attack_psnr']
                ssim_delta = metrics['legitimate_ssim'] - metrics['attack_ssim']
                
                print(f"\nQuality Assessment:")
                print(f"  Legitimate - PSNR: {metrics['legitimate_psnr']:.2f} dB, SSIM: {metrics['legitimate_ssim']:.4f}")
                print(f"  Attack     - PSNR: {metrics['attack_psnr']:.2f} dB, SSIM: {metrics['attack_ssim']:.4f}")
                print(f"  Degradation - PSNR: {psnr_delta:+.2f} dB, SSIM: {ssim_delta:+.4f}")
            
            # Forensic analysis summary
            if 'forensic_detection' in results and not results['forensic_detection']['analysis_skipped']:
                forensic = results['forensic_detection']
                if forensic['detection_accurate'] is None:
                    detection_status = "⚠ WARNING (Victim not tested)"
                elif forensic['detection_accurate']:
                    detection_status = "✓ PASSED"
                else:
                    detection_status = "✗ FAILED"
                print(f"\nForensic Analysis:")
                print(f"  Expected User ID: {forensic['expected_user_id']}")
                print(f"  Detected User ID: {forensic['detected_user_id']}")
                print(f"  Detection Status: {detection_status}")
                if forensic['detection_accurate'] is None:
                    print(f"  Note: Victim ID exceeds max_users limit")
            
            # Output files summary
            if 'output_files' in results:
                files = results['output_files']
                total_files = sum(len(file_list) for file_list in files.values())
                print(f"\nGenerated Files ({total_files} total):")
                for category, file_list in files.items():
                    if file_list:
                        print(f"  {category.replace('_', ' ').title()}: {', '.join(file_list)}")
            
            print(f"\n✓ All results saved to: {os.getcwd()}")
            
        else:
            print(f"✗ Status: FAILED")
            print(f"✗ Execution Time: {results['execution_time_seconds']:.1f} seconds")
            print(f"✗ Error: {results.get('error_message', 'Unknown error')}")
            print(f"✗ Error Type: {results.get('error_type', 'Unknown')}")
            
            print(f"\nTroubleshooting:")
            print(f"  - Check that all required modules are available")
            print(f"  - Verify input parameters are valid")
            print(f"  - Ensure sufficient disk space and write permissions")
            print(f"  - Run with --verbose for detailed error information")
        
        print("="*70)


def main():
    """
    Main entry point for the experimental script.
    
    Coordinates the complete experimental pipeline including argument parsing,
    validation, execution, and reporting.
    """
    pipeline = ExperimentalPipeline()
    
    try:
        # Parse and validate command-line arguments
        args = pipeline.parse_arguments()
        pipeline.validate_arguments(args)
        
        # Set up experimental environment
        pipeline.setup_experiment_environment(args)
        
        # Run complete experimental flow
        results = pipeline.run_complete_experimental_flow(args)
        
        # Print final summary
        pipeline.print_final_summary(results, args)
        
        # Exit with appropriate code
        exit_code = 0 if results['execution_successful'] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n\nExperiment interrupted by user (Ctrl+C)")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"\nFatal error in experimental pipeline: {str(e)}")
        if '--verbose' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()