# Project Structure

This document describes the organization of the Perceptual Interdependence research project.

## Directory Structure

```
perceptual-interdependence/
├── README.md                    # Main project documentation
├── main.py                      # Unified entry point
├── setup.py                     # Package configuration
├── requirements.txt             # Python dependencies
├── 
├── core/                        # Core algorithm implementations
│   ├── __init__.py
│   ├── asset_binder_complex.py  # Geometry-aware binding algorithm
│   ├── render_simulator.py      # PBR-lite rendering engine
│   └── rgb_forensics.py         # Forensic analysis system
├── 
├── experiments/                 # Research experimental framework
│   ├── __init__.py
│   ├── research_orchestrator.py # Experimental coordination
│   └── run_paper_experiment.py  # Main experimental pipeline
├── 
├── utils/                       # Utility modules and tools
│   ├── __init__.py
│   ├── texture_generator.py     # Texture generation utilities
│   ├── demo_pipeline.py         # Complete demonstration workflow
│   └── example_usage.py         # Usage examples
├── 
├── assets/                      # Texture assets and data
│   ├── raw/                     # Raw input textures (user-provided)
│   ├── processed/               # Processed/generated textures
│   └── bound/                   # User-bound texture outputs
├── 
├── results/                     # Experimental results and reports
│   ├── experimental_report.txt  # Comprehensive text report
│   └── forensics_results_table.tex # LaTeX results table
├── 
├── figures/                     # Generated visualization outputs
│   ├── fig_legit.png           # Legitimate rendering scenario
│   ├── fig_attack.png          # Attack rendering scenario
│   └── forensic_report.png     # Forensic analysis visualization
├── 
└── docs/                        # Additional documentation
    └── PROJECT_STRUCTURE.md    # This file
```

## Usage Patterns

### Quick Start
```bash
# Run complete demonstration
python main.py demo

# Run experimental pipeline
python main.py experiment --victim-id 42 --max-users 100

# Bind specific assets
python main.py bind --albedo assets/raw/wood.png --normal assets/raw/wood_normal.png --user-id 42
```

### Direct Module Usage
```bash
# Run experiments directly
python experiments/run_paper_experiment.py --victim-id 50 --max-users 100

# Run demonstration pipeline
python utils/demo_pipeline.py
```

## Asset Management

- **Raw Assets**: Place your original textures in `assets/raw/`
- **Processed Assets**: Generated/downloaded textures stored in `assets/processed/`
- **Bound Assets**: User-specific bound textures output to `assets/bound/`

## Output Organization

- **Results**: Quantitative experimental results in `results/`
- **Figures**: Visual outputs and demonstrations in `figures/`
- **Reports**: Comprehensive analysis reports for research publication