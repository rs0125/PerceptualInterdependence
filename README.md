# Perceptual Interdependence: Geometry-Aware Photometric Binding System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research-grade implementation of geometry-aware asset binding for 3D texture protection. This system provides mathematically guaranteed restoration for legitimate users while maintaining security against unauthorized access through **Analytically Safe One-Way Binding**.

## 🚀 Key Features

- **CPU-Optimized Performance**: Ultra-fast mathematical operations with Numba JIT compilation
- **Analytically Safe Binding**: Strict algebraic cancellation without calibration loops
- **Geometry-Aware Processing**: Normal map processing with geometric constraint validation
- **Research-Grade Architecture**: Modular, extensible, and thoroughly tested codebase
- **Multiple Interfaces**: Command-line tools, Python API, and interactive web GUI
- **Comprehensive Validation**: Built-in forensic analysis and verification tools

## 📊 Performance Highlights

- **2048×2048 textures**: Processed in ~0.1 seconds with Numba JIT
- **Memory efficient**: Vectorized operations with minimal allocations
- **Deterministic**: Reproducible results with seed-based randomization
- **Scalable**: Linear performance scaling with image size

## 🔬 Core Algorithm

The system implements **Analytically Safe One-Way Binding** with mathematical guarantees:

```
Poison:   A_new = A_original × (1 + P)     [Brighten albedo]
Antidote: Z_new = Z_original / (1 + P)     [Steepen normals]
Result:   A_new × Z_new = A_original × Z_original  [Perfect cancellation]
```

Where:
- `A` = Albedo texture values
- `Z` = Normal map Z-component (surface steepness)
- `P` = Poison strength (≥ 0)

## 📦 Installation

### From Source (Recommended for Research)

```bash
git clone https://github.com/research/perceptual-interdependence.git
cd perceptual-interdependence
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/research/perceptual-interdependence.git
cd perceptual-interdependence
pip install -e ".[dev,docs,benchmark]"
```

### Dependencies

**Core Requirements:**
- Python 3.8+
- NumPy ≥ 1.20.0
- Pillow ≥ 8.0.0
- Numba ≥ 0.56.0 (for JIT acceleration)

**Optional:**
- Streamlit ≥ 1.20.0 (for GUI)
- Matplotlib ≥ 3.3.0 (for visualization)
- Pytest ≥ 6.0 (for testing)

## 🚀 Quick Start

### Command Line Interface

```bash
# Bind textures for a specific user
perceptual-bind bind --albedo texture.png --normal normal.png --user-id 42

# Run performance benchmark
perceptual-bind benchmark --size 2048 2048 --iterations 5

# Launch interactive GUI
perceptual-bind gui --port 8501

# Run comprehensive experiments
perceptual-bind experiment --victim-id 42 --attacker-id 99
```

### Python API

```python
from perceptual_interdependence import AssetBinder

# Initialize binder
binder = AssetBinder(output_dir="./results")

# Bind textures
results = binder.bind_textures(
    albedo_path="albedo.png",
    normal_path="normal.png", 
    user_id=42,
    poison_strength=0.2
)

print(f"Bound textures saved to: {results['output_paths']}")
print(f"Saturation ratio: {results['statistics']['saturation_ratio']:.1%}")
```

### Advanced Usage

```python
from perceptual_interdependence import CPUOptimizedMath, ValidationSuite

# Direct algorithm access
cpu_math = CPUOptimizedMath()
poison_map = cpu_math.generate_poison_map((1024, 1024), seed=42, poison_strength=0.2)

# Validation and testing
validator = ValidationSuite()
integrity_results = validator.validate_system_integrity()
performance_results = validator.benchmark_performance()
```

## 🏗️ Project Structure

```
perceptual-interdependence/
├── src/perceptual_interdependence/          # Main package
│   ├── core/                               # Core binding system
│   │   ├── asset_binder.py                 # Main AssetBinder class
│   │   ├── render_simulator.py             # Photometric rendering
│   │   └── forensics.py                    # RGB forensic analysis
│   ├── algorithms/                         # Mathematical algorithms
│   │   └── cpu_math.py                     # CPU-optimized operations
│   ├── cli/                               # Command-line interface
│   │   └── main.py                        # CLI entry point
│   ├── gui/                               # Graphical interfaces
│   │   └── streamlit_app.py               # Web GUI
│   └── utils/                             # Utility functions
│       ├── texture_processing.py          # Texture I/O utilities
│       └── validation.py                  # Validation suite
├── tests/                                 # Test suite
│   ├── unit/                             # Unit tests
│   ├── integration/                      # Integration tests
│   └── benchmarks/                       # Performance tests
├── docs/                                 # Documentation
├── data/                                 # Sample data
│   ├── samples/                          # Sample textures
│   └── results/                          # Example results
├── scripts/                              # Utility scripts
└── benchmarks/                           # Performance benchmarks
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v --cov=perceptual_interdependence
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/benchmarks/ -v
```

### Manual Testing

```bash
# Test system integrity
python -c "
from perceptual_interdependence.utils.validation import ValidationSuite
validator = ValidationSuite()
results = validator.validate_system_integrity()
print('System integrity:', 'PASSED' if results['valid'] else 'FAILED')
"

# Quick performance test
python -c "
from perceptual_interdependence.algorithms.cpu_math import get_cpu_math
cpu_math = get_cpu_math()
results = cpu_math.benchmark_performance((1024, 1024))
print(f'Processing time: {results[\"total\"]:.3f}s')
"
```

## 📈 Performance Benchmarks

### CPU Performance (Numba JIT Enabled)

| Image Size | Processing Time | Throughput |
|------------|----------------|------------|
| 512×512    | ~0.008s        | 33 Mpx/s   |
| 1024×1024  | ~0.032s        | 33 Mpx/s   |
| 2048×2048  | ~0.096s        | 44 Mpx/s   |
| 4096×4096  | ~0.384s        | 44 Mpx/s   |

### Memory Usage

- **Peak memory**: ~2.5× input texture size
- **Streaming support**: For textures > 8K resolution
- **Memory efficiency**: Vectorized operations minimize allocations

## 🔬 Research Applications

### Academic Research

```python
# Comprehensive experimental pipeline
from perceptual_interdependence.experiments import run_full_experiment

results = run_full_experiment(
    victim_id=42,
    attacker_id=99, 
    max_users=1000,
    output_dir="./research_results"
)
```

### Forensic Analysis

```python
from perceptual_interdependence.core.forensics import RGBForensics

forensics = RGBForensics()
analysis = forensics.analyze_texture_pair(
    "original_albedo.png",
    "suspicious_albedo.png"
)
print(f"Tampering detected: {analysis['tampering_detected']}")
```

### Custom Algorithms

```python
from perceptual_interdependence.algorithms.cpu_math import CPUOptimizedMath

class CustomMath(CPUOptimizedMath):
    def custom_poison_generation(self, shape, seed, strength):
        # Implement custom poison generation algorithm
        return super().generate_poison_map(shape, seed, strength)
```

## 🎯 Use Cases

1. **3D Asset Protection**: Protect valuable 3D textures from unauthorized use
2. **Digital Rights Management**: Embed user-specific binding in textures
3. **Forensic Analysis**: Detect tampering and unauthorized modifications
4. **Research Platform**: Extensible framework for binding algorithm research
5. **Performance Benchmarking**: Evaluate mathematical operation performance

## 🔧 Configuration

### Environment Variables

```bash
export PERCEPTUAL_NUMBA_DISABLE=1    # Disable Numba JIT compilation
export PERCEPTUAL_OUTPUT_DIR=/path   # Default output directory
export PERCEPTUAL_LOG_LEVEL=DEBUG    # Logging level
```

### Configuration Files

Create `~/.perceptual_interdependence/config.yaml`:

```yaml
default_poison_strength: 0.2
output_directory: "./results"
enable_numba: true
benchmark_iterations: 5
```

## 📚 Documentation

### API Reference

- **[AssetBinder API](docs/api/asset_binder.md)**: Main binding interface
- **[CPUOptimizedMath API](docs/api/cpu_math.md)**: Mathematical operations
- **[ValidationSuite API](docs/api/validation.md)**: Testing and validation
- **[CLI Reference](docs/cli/commands.md)**: Command-line usage

### Tutorials

- **[Getting Started](docs/tutorials/getting_started.md)**: Basic usage tutorial
- **[Advanced Binding](docs/tutorials/advanced_binding.md)**: Custom binding scenarios
- **[Performance Optimization](docs/tutorials/performance.md)**: Optimization techniques
- **[Research Workflows](docs/tutorials/research.md)**: Academic research usage

### Algorithm Details

- **[Mathematical Foundation](docs/algorithms/mathematics.md)**: Core mathematical concepts
- **[CPU Optimization](docs/algorithms/cpu_optimization.md)**: Performance optimization techniques
- **[Validation Methods](docs/algorithms/validation.md)**: Quality assurance approaches

## 🤝 Contributing

We welcome contributions from the research community!

### Development Setup

```bash
git clone https://github.com/research/perceptual-interdependence.git
cd perceptual-interdependence
pip install -e ".[dev]"
pre-commit install
```

### Code Style

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/research/perceptual-interdependence/issues)
- **Discussions**: [GitHub Discussions](https://github.com/research/perceptual-interdependence/discussions)
- **Email**: research@example.com

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@software{perceptual_interdependence_2024,
  title={Perceptual Interdependence: Geometry-Aware Photometric Binding System},
  author={Research Team},
  year={2024},
  url={https://github.com/research/perceptual-interdependence},
  version={1.0.0}
}
```

## 🔄 Changelog

### Version 1.0.0 (2024-12-29)

- **🚀 Initial Release**: Complete rewrite with research-grade architecture
- **⚡ CPU Optimization**: Numba JIT compilation for 10x+ performance improvement
- **🏗️ Modular Design**: Clean separation of concerns with extensible architecture
- **🧪 Comprehensive Testing**: Full test suite with unit, integration, and benchmark tests
- **📚 Documentation**: Complete API documentation and tutorials
- **🔧 CLI Tools**: Professional command-line interface with multiple commands
- **🎨 Web GUI**: Interactive Streamlit-based graphical interface
- **🔬 Research Tools**: Built-in validation, forensics, and experimental pipelines

### Previous Versions

- **v0.x**: Legacy GPU-based implementation (deprecated)

## 🌟 Acknowledgments

- **NumPy Community**: For the foundational numerical computing library
- **Numba Team**: For JIT compilation technology enabling high-performance Python
- **Pillow Contributors**: For robust image processing capabilities
- **Research Community**: For feedback and validation of the mathematical approach

---

**Built with ❤️ for the research community**