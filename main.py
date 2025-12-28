#!/usr/bin/env python3
"""
Main entry point for Perceptual Interdependence research project.

This script provides a unified interface to all project functionality.
"""

import sys
import os
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description='Perceptual Interdependence: Geometry-Aware Asset Binding Protocol',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run complete experimental pipeline')
    exp_parser.add_argument('--victim-id', type=int, default=42, help='Victim user ID')
    exp_parser.add_argument('--attacker-id', type=int, default=99, help='Attacker user ID')
    exp_parser.add_argument('--max-users', type=int, default=100, help='Maximum users for forensics')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration with real textures')
    demo_parser.add_argument('--texture-dir', default='assets/raw', help='Directory containing raw textures')
    
    # Bind command
    bind_parser = subparsers.add_parser('bind', help='Bind specific texture assets')
    bind_parser.add_argument('--albedo', required=True, help='Path to albedo texture')
    bind_parser.add_argument('--normal', required=True, help='Path to normal map')
    bind_parser.add_argument('--user-id', type=int, required=True, help='User ID for binding')
    bind_parser.add_argument('--strength', type=float, default=0.2, help='Poison strength')
    
    args = parser.parse_args()
    
    if args.command == 'experiment':
        from experiments.run_paper_experiment import main as run_experiment
        sys.argv = ['run_paper_experiment.py', 
                   f'--victim-id={args.victim_id}',
                   f'--attacker-id={args.attacker_id}',
                   f'--max-users={args.max_users}']
        run_experiment()
        
    elif args.command == 'demo':
        from utils.demo_pipeline import create_demonstration_images
        create_demonstration_images()
        
    elif args.command == 'bind':
        from core.asset_binder_complex import AssetBinderComplex
        binder = AssetBinderComplex()
        binder.bind_textures(args.albedo, args.normal, args.user_id, args.strength)
        print(f"Assets bound for user {args.user_id}")
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()