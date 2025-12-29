# AssetBinder API Reference

The `AssetBinder` class is the main interface for binding texture assets using the Analytically Safe One-Way Binding algorithm.

## Class: AssetBinder

### Constructor

```python
AssetBinder(output_dir: Optional[Path] = None)
```

**Parameters:**
- `output_dir` (Path, optional): Directory for output files. Defaults to current directory.

**Example:**
```python
from perceptual_interdependence import AssetBinder

# Use current directory
binder = AssetBinder()

# Specify output directory
binder = AssetBinder(output_dir="./results")
```

### Methods

#### bind_textures()

```python
bind_textures(
    albedo_path: str,
    normal_path: str,
    user_id: int,
    poison_strength: float = 0.2,
    output_prefix: str = "bound"
) -> Dict[str, Any]
```

Bind albedo and normal textures for a specific user using Analytically Safe One-Way Binding.

**Parameters:**
- `albedo_path` (str): Path to input albedo texture file
- `normal_path` (str): Path to input normal map file
- `user_id` (int): User ID for binding (used as random seed)
- `poison_strength` (float): Poison strength [0.0-1.0] (default: 0.2)
- `output_prefix` (str): Output filename prefix (default: "bound")

**Returns:**
- `Dict[str, Any]`: Results dictionary containing:
  - `output_paths`: Dictionary with 'albedo' and 'normal' output file paths
  - `statistics`: Dictionary with binding statistics
  - `user_id`: The user ID used for binding
  - `poison_strength`: The poison strength applied

**Raises:**
- `FileNotFoundError`: If input files don't exist
- `ValueError`: If parameters are invalid or binding fails

**Example:**
```python
results = binder.bind_textures(
    albedo_path="texture.png",
    normal_path="normal.png",
    user_id=42,
    poison_strength=0.25,
    output_prefix="user42"
)

print(f"Bound albedo: {results['output_paths']['albedo']}")
print(f"Saturation ratio: {results['statistics']['saturation_ratio']:.1%}")
```

#### _generate_poison_map()

```python
_generate_poison_map(
    shape: Tuple[int, int],
    seed: int,
    poison_strength: float
) -> np.ndarray
```

Generate poison map for texture modification (internal method).

**Parameters:**
- `shape` (Tuple[int, int]): Shape of the poison map (height, width)
- `seed` (int): Random seed for reproducible generation
- `poison_strength` (float): Maximum poison strength [0.0-1.0]

**Returns:**
- `np.ndarray`: Poison map array with values in [0, poison_strength] range

#### _apply_poison_to_albedo()

```python
_apply_poison_to_albedo(
    albedo: np.ndarray,
    poison_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Apply poison to albedo texture with saturation handling (internal method).

**Parameters:**
- `albedo` (np.ndarray): Input albedo texture array
- `poison_map` (np.ndarray): Poison map to apply

**Returns:**
- `Tuple[np.ndarray, Dict[str, Any]]`: Poisoned albedo and statistics

#### _calculate_antidote_normal()

```python
_calculate_antidote_normal(
    original_normal: np.ndarray,
    poison_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Calculate antidote normal map with geometric validation (internal method).

**Parameters:**
- `original_normal` (np.ndarray): Input normal map array
- `poison_map` (np.ndarray): Poison map used for albedo

**Returns:**
- `Tuple[np.ndarray, Dict[str, Any]]`: Antidote normal map and statistics

## Usage Patterns

### Basic Binding

```python
from perceptual_interdependence import AssetBinder

binder = AssetBinder(output_dir="./bound_assets")

# Bind textures for user 42
results = binder.bind_textures(
    albedo_path="materials/brick_albedo.png",
    normal_path="materials/brick_normal.png",
    user_id=42
)

print(f"Binding complete: {results['output_paths']}")
```

### Batch Processing

```python
import os
from pathlib import Path

binder = AssetBinder(output_dir="./batch_results")

# Process multiple users
for user_id in range(1, 101):
    results = binder.bind_textures(
        albedo_path="base_texture.png",
        normal_path="base_normal.png",
        user_id=user_id,
        output_prefix=f"user_{user_id:03d}"
    )
    print(f"User {user_id}: {results['statistics']['saturation_ratio']:.1%} saturation")
```

### Custom Poison Strength

```python
# High security binding with stronger poison
results = binder.bind_textures(
    albedo_path="sensitive_texture.png",
    normal_path="sensitive_normal.png",
    user_id=123,
    poison_strength=0.3,  # Stronger poison for higher security
    output_prefix="high_security"
)
```

## Error Handling

```python
try:
    results = binder.bind_textures(
        albedo_path="texture.png",
        normal_path="normal.png",
        user_id=42
    )
except FileNotFoundError as e:
    print(f"Input file not found: {e}")
except ValueError as e:
    print(f"Invalid parameters or binding failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- **Memory Usage**: Peak memory ~2.5× input texture size
- **Processing Time**: ~0.1s for 2048×2048 textures with Numba JIT
- **Batch Processing**: Reuse AssetBinder instance for multiple bindings
- **Large Textures**: Consider processing in chunks for >8K textures

## Mathematical Foundation

The AssetBinder implements the core mathematical relationship:

```
Poison:   A_new = A_original × (1 + P)
Antidote: Z_new = Z_original / (1 + P)
Result:   A_new × Z_new = A_original × Z_original
```

Where perfect cancellation is guaranteed through analytical computation without iterative calibration.