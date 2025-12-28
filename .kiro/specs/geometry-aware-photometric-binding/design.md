# Design Document

## Overview

The AssetBinderComplex class implements a sophisticated geometry-aware photometric binding algorithm that applies controlled noise to albedo textures while maintaining geometric consistency through corresponding normal map adjustments. The system uses block-based noise generation optimized for BC7 compression survival and employs vector mathematics to preserve surface normal characteristics.

## Architecture

The system follows a modular design with clear separation between binding and validation components:

```
Geometry-Aware Photometric Binding System
├── AssetBinderComplex (Core Binding)
│   ├── Input Processing (load & validate textures)
│   ├── Noise Generation (block-based poison generation)
│   ├── Binding Algorithm (albedo poisoning + normal antidote)
│   └── Output Processing (save bound texture pairs)
└── RenderSimulator (Validation & Quality Assessment)
    ├── PBR-Lite Renderer (simplified lighting simulation)
    ├── Quality Metrics (PSNR & SSIM calculation)
    └── Experiment Logic (legitimate vs attack testing)
```

### Core Dependencies
- **NumPy**: Mathematical operations, array processing, and random number generation
- **PIL (Pillow)**: Image loading, processing, and saving operations
- **scikit-image**: SSIM metric calculation for quality assessment

## Components and Interfaces

### AssetBinderComplex Class

```python
class AssetBinderComplex:
    def __init__(self):
        # Initialize class without parameters
        
    def bind_textures(self, clean_albedo_path: str, original_normal_path: str, 
                     user_seed: int, poison_strength: float = 0.2) -> None:
        # Main binding method orchestrating the entire process
        
    def _load_and_validate_inputs(self, albedo_path: str, normal_path: str) -> tuple:
        # Load and validate input textures
        
    def _generate_block_noise(self, shape: tuple, seed: int, poison_strength: float) -> np.ndarray:
        # Generate 4x4 block-based noise for BC7 compression survival
        
    def _apply_poison_to_albedo(self, albedo: np.ndarray, noise: np.ndarray) -> tuple:
        # Apply brightening poison and calculate effective noise
        
    def _generate_antidote_normal(self, normal: np.ndarray, s_effective: np.ndarray) -> np.ndarray:
        # Generate compensating normal map maintaining geometric consistency
        
    def _save_outputs(self, bound_albedo: np.ndarray, bound_normal: np.ndarray, seed: int) -> None:
        # Save processed textures with appropriate naming convention
```

### RenderSimulator Class

```python
class RenderSimulator:
    def __init__(self):
        # Initialize validation system
        
    def render(self, albedo_path: str, normal_path: str, light_dir: list = [0, 0, 1]) -> np.ndarray:
        # PBR-lite rendering with simplified lighting model
        
    def evaluate(self, clean_ref_path: str, rendered_img: np.ndarray) -> tuple:
        # Calculate PSNR and SSIM quality metrics
        
    def _load_textures_for_rendering(self, albedo_path: str, normal_path: str) -> tuple:
        # Load and prepare textures for rendering
        
    def _calculate_psnr(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        # Compute Peak Signal-to-Noise Ratio
        
    def _calculate_ssim(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        # Compute Structural Similarity Index using skimage
```

### Input Processing Module

**Purpose**: Handle texture loading and validation
- Load albedo textures and convert to float [0.0, 1.0]
- Load normal maps and unpack from [0, 255] to [-1.0, 1.0]
- Normalize normal vectors to unit length
- Validate file existence and format compatibility

### Noise Generation Module

**Purpose**: Create BC7-optimized block noise
- Generate deterministic noise using seeded RandomState
- Create 4x4 pixel blocks for compression survival
- Produce scalar values in range [1.0, 1.0 + poison_strength]
- Ensure noise pattern maintains spatial coherence

### Binding Algorithm Module

**Purpose**: Core mathematical operations for texture binding

#### Albedo Poisoning Process:
1. **Poison Application**: `Albedo_New = Albedo_Old * Noise_S`
2. **Range Clipping**: Clip results to [0.0, 1.0]
3. **Effective Noise Calculation**: `S_effective = Albedo_New / (Albedo_Old + 1e-6)`

#### Normal Antidote Process:
1. **Z-Component Adjustment**: `Z_new = Z_old / S_effective`
2. **Lateral Magnitude Preservation**: 
   - `Lat_old = sqrt(x_old² + y_old²)`
   - `Lat_new = sqrt(1 - Z_new²)`
3. **Vector Scaling**: `k = Lat_new / (Lat_old + 1e-6)`
4. **Component Recalculation**: 
   - `X_new = X_old * k`
   - `Y_new = Y_old * k`

### Output Processing Module

**Purpose**: Handle texture saving and format conversion
- Pack float albedo values back to [0, 255] range
- Pack normal vectors from [-1.0, 1.0] to RGB [0, 255]
- Generate appropriate filenames with seed suffix
- Save in PNG format maintaining quality

### Validation Processing Module

**Purpose**: Render simulation and quality assessment for bound assets

#### PBR-Lite Rendering Process:
1. **Texture Loading**: Load albedo and normal textures for rendering
2. **Normal Unpacking**: Convert normal maps from [0, 255] to [-1, 1] vectors
3. **Lighting Calculation**: `Shading = dot(Normal, LightDir)`, clamped to [0.0, 1.0]
4. **Pixel Composition**: `Final_Pixel = Albedo * Shading` (broadcast across RGB)
5. **Output Generation**: Return float array [0.0, 1.0]

#### Quality Metrics Calculation:
1. **PSNR Computation**: Calculate Peak Signal-to-Noise Ratio between images
2. **SSIM Computation**: Use skimage.metrics for Structural Similarity calculation
3. **Luminance Processing**: Compute SSIM for luminance or average across RGB channels
4. **Metric Reporting**: Return both PSNR and SSIM as numerical values

## Data Models

### Texture Data Structure
```python
# Albedo texture: numpy array shape (height, width, 3)
# Values: float32 [0.0, 1.0]
albedo_texture: np.ndarray

# Normal map: numpy array shape (height, width, 3)  
# Values: float32 [-1.0, 1.0] representing normalized vectors
normal_texture: np.ndarray

# Block noise: numpy array shape (height, width)
# Values: float32 [1.0, 1.0 + poison_strength]
block_noise: np.ndarray
```

### Mathematical Constants
```python
EPSILON = 1e-6  # Division by zero protection
BLOCK_SIZE = 4  # 4x4 pixel blocks for BC7 optimization
RGB_MAX = 255.0  # Maximum RGB value for conversion
```

## Error Handling

### Input Validation Errors
- **FileNotFoundError**: Handle missing texture files gracefully
- **PIL.UnidentifiedImageError**: Validate image format compatibility
- **ValueError**: Check parameter ranges (poison_strength [0.0, 1.0])

### Mathematical Edge Cases
- **Division by Zero**: Use epsilon values (1e-6) for safe division
- **Vector Normalization**: Handle zero-length vectors in normal maps
- **Range Clipping**: Ensure all output values remain within valid ranges

### Memory Management
- **Large Texture Handling**: Process textures in chunks if memory constraints exist
- **Temporary Array Cleanup**: Explicit cleanup of intermediate arrays for large textures

## Testing Strategy

### Unit Testing Focus Areas

1. **Input Processing Tests**
   - Texture loading with various formats (PNG, JPEG, TIFF)
   - Normal vector unpacking and normalization accuracy
   - Parameter validation (seed, poison_strength ranges)

2. **Noise Generation Tests**
   - Deterministic output with same seed values
   - Block structure validation (4x4 patterns)
   - Value range verification [1.0, 1.0 + poison_strength]

3. **Mathematical Algorithm Tests**
   - Albedo poisoning accuracy with known inputs
   - Normal vector preservation (unit length maintenance)
   - Azimuth preservation in vector bending operations

4. **Output Processing Tests**
   - Correct packing of float values to RGB
   - File naming convention adherence
   - Output image quality and format validation

### Integration Testing

1. **End-to-End Workflow**
   - Complete binding process with sample textures
   - Verify geometric consistency between albedo and normal outputs
   - Compression survival testing with BC7 codec

2. **Edge Case Scenarios**
   - Extreme poison_strength values (near 0.0 and 1.0)
   - High-contrast input textures
   - Various texture resolutions and aspect ratios

### Performance Considerations

- **Memory Efficiency**: Process large textures without excessive memory usage
- **Computational Optimization**: Vectorized operations using NumPy
- **I/O Performance**: Efficient image loading and saving operations