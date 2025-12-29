# CPUOptimizedMath API Reference

The `CPUOptimizedMath` class provides high-performance mathematical operations optimized for CPU execution with Numba JIT compilation.

## Class: CPUOptimizedMath

### Constructor

```python
CPUOptimizedMath()
```

Initializes the CPU-optimized mathematical operations system with automatic Numba JIT compilation detection.

**Example:**
```python
from perceptual_interdependence.algorithms.cpu_math import CPUOptimizedMath

cpu_math = CPUOptimizedMath()
```

### Factory Function

#### get_cpu_math()

```python
get_cpu_math() -> CPUOptimizedMath
```

Factory function that returns a CPUOptimizedMath instance with automatic acceleration detection.

**Returns:**
- `CPUOptimizedMath`: Configured instance with optimal acceleration

**Example:**
```python
from perceptual_interdependence.algorithms.cpu_math import get_cpu_math

cpu_math = get_cpu_math()
```

### Core Methods

#### generate_poison_map()

```python
generate_poison_map(
    shape: Tuple[int, int],
    seed: int,
    poison_strength: float
) -> np.ndarray
```

Generate pseudo-random poison map using optimized algorithms.

**Parameters:**
- `shape` (Tuple[int, int]): Output shape (height, width)
- `seed` (int): Random seed for reproducible generation
- `poison_strength` (float): Maximum poison value [0.0, 1.0]

**Returns:**
- `np.ndarray`: Poison map with values in [0, poison_strength] range

**Performance:**
- **2048×2048**: ~0.015s with Numba JIT
- **Memory**: ~16MB for float32 output

**Example:**
```python
poison_map = cpu_math.generate_poison_map(
    shape=(2048, 2048),
    seed=42,
    poison_strength=0.2
)
```

#### apply_poison_to_albedo()

```python
apply_poison_to_albedo(
    albedo: np.ndarray,
    poison_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]
```

Apply poison to albedo texture with saturation handling.

**Parameters:**
- `albedo` (np.ndarray): Input albedo texture [0.0, 1.0]
- `poison_map` (np.ndarray): Poison map to apply

**Returns:**
- `Tuple[np.ndarray, Dict[str, float]]`: Poisoned albedo and statistics
  - `poisoned_albedo`: Modified albedo texture
  - `statistics`: Dictionary with saturation metrics

**Example:**
```python
poisoned_albedo, stats = cpu_math.apply_poison_to_albedo(albedo, poison_map)
print(f"Saturation ratio: {stats['saturation_ratio']:.1%}")
```

#### calculate_antidote_normal()

```python
calculate_antidote_normal(
    normal_map: np.ndarray,
    poison_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]
```

Calculate antidote normal map with geometric constraint validation.

**Parameters:**
- `normal_map` (np.ndarray): Input normal map [0.0, 1.0] or [-1.0, 1.0]
- `poison_map` (np.ndarray): Poison map used for albedo

**Returns:**
- `Tuple[np.ndarray, Dict[str, float]]`: Antidote normal and statistics
  - `antidote_normal`: Modified normal map
  - `statistics`: Dictionary with geometric validation metrics

**Example:**
```python
antidote_normal, stats = cpu_math.calculate_antidote_normal(normal_map, poison_map)
print(f"Z-range: [{stats['z_min']:.3f}, {stats['z_max']:.3f}]")
```

### Performance Methods

#### benchmark_performance()

```python
benchmark_performance(
    shape: Tuple[int, int] = (2048, 2048)
) -> Dict[str, float]
```

Benchmark mathematical operations performance.

**Parameters:**
- `shape` (Tuple[int, int]): Test image size (default: 2048×2048)

**Returns:**
- `Dict[str, float]`: Performance metrics
  - `poison_generation`: Time for poison map generation
  - `albedo_processing`: Time for albedo poisoning
  - `normal_processing`: Time for normal antidote calculation
  - `total`: Total processing time

**Example:**
```python
results = cpu_math.benchmark_performance((1024, 1024))
print(f"Total time: {results['total']:.3f}s")
print(f"Throughput: {(1024*1024)/results['total']/1e6:.1f} Mpx/s")
```

### Utility Methods

#### validate_geometric_constraints()

```python
validate_geometric_constraints(
    normal_map: np.ndarray
) -> Dict[str, Any]
```

Validate geometric constraints of normal map.

**Parameters:**
- `normal_map` (np.ndarray): Normal map to validate

**Returns:**
- `Dict[str, Any]`: Validation results
  - `valid`: Boolean indicating if constraints are satisfied
  - `z_range`: Tuple of (min, max) Z-component values
  - `flat_pixels`: Number of flat surface pixels
  - `geometric_errors`: Number of invalid normal vectors

#### normalize_normal_map()

```python
normalize_normal_map(
    normal_map: np.ndarray
) -> np.ndarray
```

Normalize normal map vectors to unit length.

**Parameters:**
- `normal_map` (np.ndarray): Input normal map

**Returns:**
- `np.ndarray`: Normalized normal map

### Advanced Usage

#### Custom Poison Generation

```python
class CustomMath(CPUOptimizedMath):
    def generate_custom_poison(self, shape, seed, strength, pattern="perlin"):
        if pattern == "perlin":
            # Custom Perlin noise implementation
            return self._generate_perlin_noise(shape, seed, strength)
        else:
            return super().generate_poison_map(shape, seed, strength)
```

#### Batch Processing Optimization

```python
# Efficient batch processing
cpu_math = get_cpu_math()

# Pre-generate poison maps for multiple users
poison_maps = {}
for user_id in range(100):
    poison_maps[user_id] = cpu_math.generate_poison_map(
        shape=(2048, 2048),
        seed=user_id,
        poison_strength=0.2
    )

# Apply to textures
for user_id, poison_map in poison_maps.items():
    poisoned_albedo, _ = cpu_math.apply_poison_to_albedo(albedo, poison_map)
    antidote_normal, _ = cpu_math.calculate_antidote_normal(normal, poison_map)
```

## Performance Optimization

### Numba JIT Compilation

The system automatically detects and uses Numba JIT compilation:

```python
# Check if Numba is available
if cpu_math.numba_available:
    print("Numba JIT acceleration enabled")
else:
    print("Using NumPy fallback implementation")
```

### Memory Management

```python
# For large textures, consider memory-efficient processing
def process_large_texture(albedo_path, chunk_size=1024):
    # Load texture in chunks
    for chunk in load_texture_chunks(albedo_path, chunk_size):
        processed_chunk = cpu_math.apply_poison_to_albedo(chunk, poison_chunk)
        yield processed_chunk
```

### Threading Considerations

```python
# CPU math operations are thread-safe for read-only operations
import concurrent.futures

def process_user(user_id):
    poison_map = cpu_math.generate_poison_map((2048, 2048), user_id, 0.2)
    return poison_map

# Parallel poison generation
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_user, uid) for uid in range(100)]
    poison_maps = [f.result() for f in futures]
```

## Error Handling

```python
try:
    poison_map = cpu_math.generate_poison_map((2048, 2048), 42, 0.2)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except MemoryError as e:
    print(f"Insufficient memory: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Benchmarks

| Operation | 1024×1024 | 2048×2048 | 4096×4096 |
|-----------|-----------|-----------|-----------|
| Poison Generation | 0.008s | 0.015s | 0.060s |
| Albedo Processing | 0.012s | 0.045s | 0.180s |
| Normal Processing | 0.015s | 0.055s | 0.220s |
| **Total** | **0.035s** | **0.115s** | **0.460s** |

*Benchmarks on Intel i7-10700K with Numba JIT enabled*