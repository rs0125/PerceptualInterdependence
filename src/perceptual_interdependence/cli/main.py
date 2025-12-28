#!/usr/bin/env python3
"""
Command-line interface for Perceptual Interdependence research project.

This module provides a unified interface to all project functionality including
binding operations, experiments, demonstrations, and GUI launch.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from ..core.asset_binder import AssetBinder


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Perceptual Interdependence: Geometry-Aware Asset Binding Protocol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bind --albedo texture.png --normal normal.png --user-id 42
  %(prog)s experiment --victim-id 42 --attacker-id 99
  %(prog)s gui --port 8501
  %(prog)s demo --output-dir ./demo_results
  %(prog)s chart --albedo texture.png --normal normal.png --victim-id 42 --attacker-id 99
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Bind command
    bind_parser = subparsers.add_parser(
        'bind', 
        help='Bind specific texture assets',
        description='Bind albedo and normal textures for a specific user'
    )
    bind_parser.add_argument(
        '--albedo', 
        type=Path, 
        required=True, 
        help='Path to albedo texture file'
    )
    bind_parser.add_argument(
        '--normal', 
        type=Path, 
        required=True, 
        help='Path to normal map file'
    )
    bind_parser.add_argument(
        '--user-id', 
        type=int, 
        required=True, 
        help='User ID for binding (used as random seed)'
    )
    bind_parser.add_argument(
        '--strength', 
        type=float, 
        default=0.2, 
        help='Poison strength [0.0-1.0] (default: 0.2)'
    )
    bind_parser.add_argument(
        '--output-dir', 
        type=Path, 
        help='Output directory (default: current directory)'
    )
    bind_parser.add_argument(
        '--prefix', 
        default='bound', 
        help='Output filename prefix (default: bound)'
    )
    
    # Experiment command
    exp_parser = subparsers.add_parser(
        'experiment', 
        help='Run complete experimental pipeline',
        description='Run comprehensive experiments with victim/attacker scenarios'
    )
    exp_parser.add_argument(
        '--victim-id', 
        type=int, 
        default=42, 
        help='Victim user ID (default: 42)'
    )
    exp_parser.add_argument(
        '--attacker-id', 
        type=int, 
        default=99, 
        help='Attacker user ID (default: 99)'
    )
    exp_parser.add_argument(
        '--max-users', 
        type=int, 
        default=100, 
        help='Maximum users for forensics (default: 100)'
    )
    exp_parser.add_argument(
        '--output-dir', 
        type=Path, 
        default='results', 
        help='Results output directory (default: results)'
    )
    
    # Demo command
    demo_parser = subparsers.add_parser(
        'demo', 
        help='Run demonstration with sample textures',
        description='Generate demonstration results with sample textures'
    )
    demo_parser.add_argument(
        '--texture-dir', 
        type=Path, 
        default='data/samples', 
        help='Directory containing sample textures (default: data/samples)'
    )
    demo_parser.add_argument(
        '--output-dir', 
        type=Path, 
        default='demo_results', 
        help='Demo output directory (default: demo_results)'
    )
    
    # GUI command
    gui_parser = subparsers.add_parser(
        'gui', 
        help='Launch interactive Streamlit GUI',
        description='Start the web-based graphical user interface'
    )
    gui_parser.add_argument(
        '--port', 
        type=int, 
        default=8501, 
        help='Port for Streamlit server (default: 8501)'
    )
    gui_parser.add_argument(
        '--host', 
        default='localhost', 
        help='Host for Streamlit server (default: localhost)'
    )
    
    # Benchmark command
    bench_parser = subparsers.add_parser(
        'benchmark', 
        help='Run performance benchmarks',
        description='Benchmark CPU math operations performance'
    )
    bench_parser.add_argument(
        '--size', 
        type=int, 
        nargs=2, 
        default=[2048, 2048], 
        metavar=('WIDTH', 'HEIGHT'),
        help='Image size for benchmarking (default: 2048 2048)'
    )
    bench_parser.add_argument(
        '--iterations', 
        type=int, 
        default=5, 
        help='Number of benchmark iterations (default: 5)'
    )
    
    # Chart command
    chart_parser = subparsers.add_parser(
        'chart',
        help='Generate demonstration charts',
        description='Create comprehensive visualization charts showing the effects of binding operations'
    )
    chart_parser.add_argument(
        '--albedo',
        type=Path,
        required=True,
        help='Path to albedo texture file'
    )
    chart_parser.add_argument(
        '--normal',
        type=Path,
        required=True,
        help='Path to normal map file'
    )
    chart_parser.add_argument(
        '--victim-id',
        type=int,
        default=42,
        help='User ID for legitimate binding scenario (default: 42)'
    )
    chart_parser.add_argument(
        '--attacker-id',
        type=int,
        default=99,
        help='User ID for attack scenario (default: 99)'
    )
    chart_parser.add_argument(
        '--output-dir',
        type=Path,
        default='.',
        help='Output directory for chart (default: current directory)'
    )
    chart_parser.add_argument(
        '--output-name',
        default='demonstration_chart.png',
        help='Chart filename (default: demonstration_chart.png)'
    )
    
    return parser


def cmd_bind(args) -> None:
    """Execute bind command."""
    print(f"🔗 Binding textures for user {args.user_id}")
    
    # Validate inputs
    if not args.albedo.exists():
        raise FileNotFoundError(f"Albedo texture not found: {args.albedo}")
    if not args.normal.exists():
        raise FileNotFoundError(f"Normal map not found: {args.normal}")
    
    # Create binder and bind textures
    binder = AssetBinder(output_dir=args.output_dir)
    results = binder.bind_textures(
        albedo_path=args.albedo,
        normal_path=args.normal,
        user_id=args.user_id,
        poison_strength=args.strength,
        output_prefix=args.prefix
    )
    
    print(f"✅ Assets bound successfully!")
    print(f"   Albedo: {results['output_paths']['albedo']}")
    print(f"   Normal: {results['output_paths']['normal']}")
    print(f"   Saturation ratio: {results['statistics']['saturation_ratio']:.1%}")


def cmd_experiment(args) -> None:
    """Execute experiment command."""
    print(f"🧪 Running experimental pipeline")
    print(f"   Victim ID: {args.victim_id}")
    print(f"   Attacker ID: {args.attacker_id}")
    print(f"   Max users: {args.max_users}")
    
    try:
        # Import here to avoid circular dependencies
        from ..experiments.research_pipeline import run_full_experiment
        
        results = run_full_experiment(
            victim_id=args.victim_id,
            attacker_id=args.attacker_id,
            max_users=args.max_users,
            output_dir=args.output_dir
        )
        
        print(f"✅ Experiment completed successfully!")
        print(f"   Results saved to: {args.output_dir}")
        
    except ImportError:
        print("❌ Experiment module not available")
        print("   Please ensure all experiment dependencies are installed")


def cmd_demo(args) -> None:
    """Execute demo command."""
    print(f"🎨 Running demonstration")
    print(f"   Texture directory: {args.texture_dir}")
    print(f"   Output directory: {args.output_dir}")
    
    try:
        from ..utils.demo_pipeline import create_demonstration_images
        
        create_demonstration_images(
            texture_dir=args.texture_dir,
            output_dir=args.output_dir
        )
        
        print(f"✅ Demo completed successfully!")
        
    except ImportError:
        print("❌ Demo module not available")


def cmd_gui(args) -> None:
    """Execute GUI command."""
    print(f"🚀 Launching Streamlit GUI...")
    print(f"🔗 Opening http://{args.host}:{args.port}")
    
    try:
        import subprocess
        from ..gui.streamlit_app import get_app_path
        
        app_path = get_app_path()
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--browser.gatherUsageStats", "false"
        ])
        
    except ImportError:
        print("❌ GUI dependencies not available")
        print("   Please install: pip install streamlit")
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")


def cmd_benchmark(args) -> None:
    """Execute benchmark command."""
    print(f"⚡ Running performance benchmarks")
    print(f"   Image size: {args.size[0]}x{args.size[1]}")
    print(f"   Iterations: {args.iterations}")
    
    from ..algorithms.cpu_math import get_cpu_math
    
    cpu_math = get_cpu_math()
    
    # Run multiple iterations for more accurate results
    total_times = []
    for i in range(args.iterations):
        print(f"\n📊 Iteration {i+1}/{args.iterations}")
        results = cpu_math.benchmark_performance(tuple(args.size))
        total_times.append(results['total'])
    
    # Calculate statistics
    import numpy as np
    mean_time = np.mean(total_times)
    std_time = np.std(total_times)
    
    print(f"\n✅ Benchmark Results:")
    print(f"   Mean time: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"   Min time: {min(total_times):.3f}s")
    print(f"   Max time: {max(total_times):.3f}s")
    
    # Calculate throughput
    pixels_per_sec = (args.size[0] * args.size[1]) / mean_time
    print(f"   Throughput: {pixels_per_sec/1e6:.1f} Mpixels/sec")


def cmd_chart(args) -> None:
    """Execute chart generation command."""
    print(f"📊 Generating demonstration chart")
    print(f"   Albedo: {args.albedo}")
    print(f"   Normal: {args.normal}")
    print(f"   Victim ID: {args.victim_id}")
    print(f"   Attacker ID: {args.attacker_id}")
    
    # Validate input files
    if not args.albedo.exists():
        raise FileNotFoundError(f"Albedo texture not found: {args.albedo}")
    if not args.normal.exists():
        raise FileNotFoundError(f"Normal map not found: {args.normal}")
    
    # Validate user IDs
    if args.victim_id <= 0:
        raise ValueError(f"Victim ID must be positive integer, got: {args.victim_id}")
    if args.attacker_id <= 0:
        raise ValueError(f"Attacker ID must be positive integer, got: {args.attacker_id}")
    
    # Prepare output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    
    try:
        # Import ChartGenerator
        from ..utils.chart_generator import ChartGenerator
        
        # Create chart generator and generate chart
        print("🎨 Initializing chart generator...")
        chart_generator = ChartGenerator()
        
        print("🔄 Processing assets and generating chart...")
        saved_path = chart_generator.generate_demonstration_chart(
            albedo_path=str(args.albedo),
            normal_path=str(args.normal),
            victim_id=args.victim_id,
            attacker_id=args.attacker_id,
            output_path=str(output_path)
        )
        
        print(f"✅ Chart generated successfully!")
        print(f"   Output: {saved_path}")
        print(f"   User IDs: Victim={args.victim_id}, Attacker={args.attacker_id}")
        
    except ImportError as e:
        print(f"❌ Chart generation dependencies not available: {e}")
        print("   Please ensure matplotlib and PIL are installed")
    except Exception as e:
        print(f"❌ Chart generation failed: {e}")
        raise


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Route to appropriate command handler
        if args.command == 'bind':
            cmd_bind(args)
        elif args.command == 'experiment':
            cmd_experiment(args)
        elif args.command == 'demo':
            cmd_demo(args)
        elif args.command == 'gui':
            cmd_gui(args)
        elif args.command == 'benchmark':
            cmd_benchmark(args)
        elif args.command == 'chart':
            cmd_chart(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()