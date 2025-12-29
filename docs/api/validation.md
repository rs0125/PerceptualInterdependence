# ValidationSuite API Reference

The `ValidationSuite` class provides comprehensive testing and validation capabilities for the perceptual interdependence system, ensuring mathematical correctness and performance validation.

## Class: ValidationSuite

### Constructor

```python
ValidationSuite()
```

Initialize the validation suite with default test configurations.

**Example:**
```python
from perceptual_interdependence.utils.validation import ValidationSuite

validator = ValidationSuite()
```

### System Validation Methods

#### validate_system_integrity()

```python
validate_system_integrity() -> Dict[str, Any]
```

Perform comprehensive system integrity validation.

**Returns:**
- `Dict[str, Any]`: Validation results containing:
  - `valid` (bool): Overall system validity
  - `mathematical_correctness` (Dict): Mathematical validation results
  - `performance_metrics` (Dict): Performance test results
  - `component_status` (Dict): Individual component validation
  - `error_details` (List): Any validation errors found

**Tests Performed:**
- Mathematical correctness of binding operations
- Poison/antidote cancellation verification
- Geometric constraint validation
- Component integration testing
- Performance regression detection

**Example:**
```python
results = validator.validate_system_integrity()

if results['valid']:
    print("System validation PASSED")
    print(f"Mathematical correctness: {results['mathematical_correctness']['score']:.1%}")
else:
    print("System validation FAILED")
    for error in results['error_details']:
        print(f"Error: {error}")
```

#### validate_mathematical_correctness()

```python
validate_mathematical_correctness(
    test_cases: int = 100,
    tolerance: float = 1e-6
) -> Dict[str, Any]
```

Validate mathematical correctness of binding operations.

**Parameters:**
- `test_cases` (int): Number of random test cases to run (default: 100)
- `tolerance` (float): Numerical tolerance for equality checks (default: 1e-6)

**Returns:**
- `Dict[str, Any]`: Mathematical validation results
  - `passed` (bool): Whether all tests passed
  - `success_rate` (float): Fraction of successful test cases
  - `max_error` (float): Maximum observed error
  - `mean_error` (float): Average error across test cases
  - `failed_cases` (List): Details of any failed test cases

**Mathematical Properties Tested:**
- Perfect cancellation: A_new × Z_new = A_original × Z_original
- Poison map generation consistency
- Geometric constraint preservation
- Numerical stability across different input ranges

**Example:**
```python
math_results = validator.validate_mathematical_correctness(
    test_cases=1000,
    tolerance=1e-8
)

print(f"Success rate: {math_results['success_rate']:.1%}")
print(f"Max error: {math_results['max_error']:.2e}")
```

#### validate_geometric_constraints()

```python
validate_geometric_constraints(
    normal_maps: List[np.ndarray]
) -> Dict[str, Any]
```

Validate geometric constraints of normal maps.

**Parameters:**
- `normal_maps` (List[np.ndarray]): List of normal maps to validate

**Returns:**
- `Dict[str, Any]`: Geometric validation results
  - `valid` (bool): Whether all constraints are satisfied
  - `unit_vector_compliance` (float): Percentage of valid unit vectors
  - `z_component_range` (Tuple): Min/max Z-component values
  - `geometric_errors` (int): Number of invalid normal vectors

**Constraints Validated:**
- Normal vectors are unit length (|n| = 1)
- Z-component is positive (surface facing outward)
- No degenerate vectors (zero length)
- Reasonable Z-component range for surface normals

**Example:**
```python
normal_maps = [load_normal_map(path) for path in normal_paths]
geo_results = validator.validate_geometric_constraints(normal_maps)

if geo_results['valid']:
    print(f"Geometric validation passed: {geo_results['unit_vector_compliance']:.1%} compliance")
else:
    print(f"Geometric errors found: {geo_results['geometric_errors']} invalid vectors")
```

### Performance Validation Methods

#### benchmark_performance()

```python
benchmark_performance(
    image_sizes: List[Tuple[int, int]] = None,
    iterations: int = 5
) -> Dict[str, Any]
```

Benchmark system performance across different image sizes.

**Parameters:**
- `image_sizes` (List[Tuple[int, int]]): Image sizes to test (default: standard sizes)
- `iterations` (int): Number of iterations per size (default: 5)

**Returns:**
- `Dict[str, Any]`: Performance benchmark results
  - `results_by_size` (Dict): Performance data for each image size
  - `throughput_analysis` (Dict): Throughput metrics
  - `scaling_analysis` (Dict): Performance scaling characteristics
  - `regression_detected` (bool): Whether performance regression was detected

**Default Test Sizes:**
- 512×512, 1024×1024, 2048×2048, 4096×4096

**Example:**
```python
perf_results = validator.benchmark_performance(
    image_sizes=[(1024, 1024), (2048, 2048)],
    iterations=10
)

for size, metrics in perf_results['results_by_size'].items():
    print(f"{size}: {metrics['mean_time']:.3f}s ± {metrics['std_time']:.3f}s")
```

#### validate_performance_regression()

```python
validate_performance_regression(
    baseline_results: Dict[str, float],
    tolerance: float = 0.1
) -> Dict[str, Any]
```

Detect performance regression against baseline measurements.

**Parameters:**
- `baseline_results` (Dict[str, float]): Baseline performance measurements
- `tolerance` (float): Acceptable performance degradation (default: 10%)

**Returns:**
- `Dict[str, Any]`: Regression analysis results
  - `regression_detected` (bool): Whether regression was found
  - `performance_changes` (Dict): Per-operation performance changes
  - `worst_regression` (float): Largest performance degradation

### Component Validation Methods

#### validate_asset_binder()

```python
validate_asset_binder() -> Dict[str, Any]
```

Validate AssetBinder component functionality.

**Returns:**
- `Dict[str, Any]`: AssetBinder validation results

**Tests:**
- Texture loading and processing
- Poison generation consistency
- Output file creation
- Error handling robustness

#### validate_cpu_math()

```python
validate_cpu_math() -> Dict[str, Any]
```

Validate CPUOptimizedMath component functionality.

**Returns:**
- `Dict[str, Any]`: CPU math validation results

**Tests:**
- Numba JIT compilation status
- Mathematical operation accuracy
- Performance optimization effectiveness
- Memory usage validation

#### validate_forensics()

```python
validate_forensics() -> Dict[str, Any]
```

Validate RGBForensics component functionality.

**Returns:**
- `Dict[str, Any]`: Forensics validation results

**Tests:**
- Signature extraction accuracy
- Traitor detection reliability
- Correlation analysis correctness
- Visualization generation

### Integration Testing Methods

#### test_end_to_end_workflow()

```python
test_end_to_end_workflow(
    test_textures: List[Tuple[str, str]]
) -> Dict[str, Any]
```

Test complete end-to-end workflow with real textures.

**Parameters:**
- `test_textures` (List[Tuple[str, str]]): List of (albedo_path, normal_path) pairs

**Returns:**
- `Dict[str, Any]`: End-to-end test results

**Workflow Tested:**
1. Texture loading and validation
2. Binding operation execution
3. Quality metrics calculation
4. Chart generation
5. Forensic analysis
6. Output file verification

#### test_cross_platform_compatibility()

```python
test_cross_platform_compatibility() -> Dict[str, Any]
```

Test system compatibility across different platforms.

**Returns:**
- `Dict[str, Any]`: Cross-platform compatibility results

**Tests:**
- File path handling
- Numerical precision consistency
- Library dependency availability
- Output format compatibility

### Quality Assurance Methods

#### validate_output_quality()

```python
validate_output_quality(
    output_files: List[str],
    quality_thresholds: Dict[str, float] = None
) -> Dict[str, Any]
```

Validate quality of generated output files.

**Parameters:**
- `output_files` (List[str]): List of output file paths to validate
- `quality_thresholds` (Dict[str, float]): Quality thresholds for validation

**Returns:**
- `Dict[str, Any]`: Output quality validation results

**Quality Metrics:**
- Image resolution and format compliance
- Color space accuracy
- File size reasonableness
- Metadata completeness

#### validate_numerical_stability()

```python
validate_numerical_stability(
    stress_test_iterations: int = 1000
) -> Dict[str, Any]
```

Test numerical stability under stress conditions.

**Parameters:**
- `stress_test_iterations` (int): Number of stress test iterations

**Returns:**
- `Dict[str, Any]`: Numerical stability results

**Stress Tests:**
- Extreme poison strength values
- Edge case texture content (all black, all white, high contrast)
- Large image sizes
- Repeated operations on same data

## Usage Patterns

### Basic System Validation

```python
from perceptual_interdependence.utils.validation import ValidationSuite

validator = ValidationSuite()

# Quick system check
results = validator.validate_system_integrity()
if results['valid']:
    print("System is ready for use")
else:
    print("System validation failed - check configuration")
```

### Comprehensive Testing Suite

```python
# Run full validation suite
validator = ValidationSuite()

# Mathematical correctness
math_results = validator.validate_mathematical_correctness(test_cases=1000)

# Performance benchmarking
perf_results = validator.benchmark_performance(iterations=10)

# Component validation
asset_binder_results = validator.validate_asset_binder()
cpu_math_results = validator.validate_cpu_math()
forensics_results = validator.validate_forensics()

# Generate validation report
validation_report = {
    'mathematical_correctness': math_results,
    'performance': perf_results,
    'components': {
        'asset_binder': asset_binder_results,
        'cpu_math': cpu_math_results,
        'forensics': forensics_results
    }
}
```

### Continuous Integration Testing

```python
def ci_validation_pipeline():
    """Validation pipeline for continuous integration."""
    validator = ValidationSuite()
    
    # Quick validation for CI
    results = validator.validate_system_integrity()
    
    if not results['valid']:
        raise RuntimeError(f"System validation failed: {results['error_details']}")
    
    # Performance regression check
    baseline = load_baseline_performance()
    perf_results = validator.benchmark_performance(iterations=3)
    regression_results = validator.validate_performance_regression(baseline)
    
    if regression_results['regression_detected']:
        print(f"Warning: Performance regression detected: {regression_results['worst_regression']:.1%}")
    
    return results

# Use in CI pipeline
if __name__ == "__main__":
    ci_validation_pipeline()
```

### Research Validation

```python
# Validate research results
def validate_research_experiment(texture_dataset):
    validator = ValidationSuite()
    
    # Test with research dataset
    end_to_end_results = validator.test_end_to_end_workflow(texture_dataset)
    
    # Validate mathematical properties
    math_results = validator.validate_mathematical_correctness(
        test_cases=10000,  # High precision for research
        tolerance=1e-10
    )
    
    # Quality validation
    output_files = [f"result_{i}.png" for i in range(len(texture_dataset))]
    quality_results = validator.validate_output_quality(output_files)
    
    return {
        'end_to_end': end_to_end_results,
        'mathematical': math_results,
        'quality': quality_results
    }
```

## Error Handling

```python
try:
    results = validator.validate_system_integrity()
except ImportError as e:
    print(f"Missing dependencies: {e}")
except MemoryError as e:
    print(f"Insufficient memory for validation: {e}")
except Exception as e:
    print(f"Validation error: {e}")
```

## Performance Considerations

- **Validation Time**: Full system validation takes 30-60 seconds
- **Memory Usage**: Peak memory ~4GB for comprehensive testing
- **Parallel Testing**: Some tests can be parallelized for faster execution
- **CI Integration**: Quick validation mode available for continuous integration