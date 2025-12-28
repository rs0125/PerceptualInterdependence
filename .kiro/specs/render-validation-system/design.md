# Design Document

## Overview

The RenderSimulator class implements a PBR-lite rendering system and quality assessment framework for validating bound texture assets. The system provides simplified physically-based rendering to simulate how textures appear under lighting conditions, combined with quantitative quality metrics (PSNR and SSIM) to measure binding effectiveness and detect potential attack scenarios.

## Architecture

The system follows a modular design with clear separation between rendering and quality assessment:

```
Render Validation System
├── RenderSimulator (Main Class)
│   ├── Texture Loading (albedo & normal map processing)
│   ├── PBR-Lite Renderer (lighting calculations & pixel composition)
│   ├── Quality Metrics (PSNR & SSIM calculation)
│   └── Experiment Logic (legitimate vs attack testing)
```

### Core Dependencies
- **NumPy**: Mathematical operations, array processing, and vectorized calculations
- **PIL (Pillow)**: Image loading and processing operations
- **scikit-image**: SSIM metric calculation using skimage.metrics.structural_similarity

## Components and Interfaces

### RenderSimulator Class

```python
class RenderSimulator:
    def __init__(self):
        # Initialize validation system without parameters
        
    def render(self, albedo_path: str, normal_path: str, light_dir: list = [0, 0, 1]) -> np.ndarray:
        # Main rendering method implementing PBR-lite pipeline
        
    def evaluate(self, clean_ref_path: str, rendered_img: np.ndarray) -> tuple:
        # Calculate PSNR and SSIM quality metrics
        
    def run_binding_experiment(self, clean_ref_path: str, bound_albedo_a: str, 
                              bound_normal_a: str, bound_normal_b: str, 
                              light_dir: list = [0, 0, 1]) -> dict:
        # Run legitimate vs attack validation experiment
        
    def _load_textures_for_rendering(self, albedo_path: str, normal_path: str) -> tuple:
        # Load and prepare textures for rendering
        
    def _normalize_light_direction(self, light_dir: list) -> np.ndarray:
        # Normalize light direction vector
        
    def _calculate_psnr(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        # Compute Peak Signal-to-Noise Ratio
        
    def _calculate_ssim(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        # Compute Structural Similarity Index using skimage
```

### Texture Loading Module

**Purpose**: Handle texture loading and format conversion for rendering
- Load albedo textures and convert from [0, 255] to float [0.0, 1.0]
- Load normal maps and unpack from [0, 255] to [-1.0, 1.0] vectors
- Validate image dimensions match between albedo and normal textures
- Handle various image formats and provide error handling

### PBR-Lite Rendering Module

**Purpose**: Implement simplified physically-based rendering pipeline

#### Rendering Pipeline:
1. **Texture Loading**: Load and prepare albedo and normal textures
2. **Light Direction Normalization**: Ensure light vector is unit length
3. **Lighting Calculation**: `Shading = dot(Normal, LightDir)`, clamped to [0.0, 1.0]
4. **Pixel Composition**: `Final_Pixel = Albedo * Shading` (broadcast across RGB)
5. **Output Generation**: Return float array [0.0, 1.0]

#### Mathematical Operations:
- **Dot Product Calculation**: Vectorized dot product across all pixels
- **Clamping**: Ensure shading values stay within [0.0, 1.0] range
- **Broadcasting**: Apply scalar shading to RGB channels
- **Normalization**: Handle light direction vector normalization

### Quality Metrics Module

**Purpose**: Calculate quantitative quality metrics for validation

#### PSNR Calculation:
1. **MSE Computation**: Calculate Mean Squared Error between images
2. **Peak Value**: Use maximum possible pixel value (1.0 for float images)
3. **PSNR Formula**: `PSNR = 20 * log10(MAX_VAL / sqrt(MSE))`
4. **Edge Case Handling**: Handle identical images (infinite PSNR)

#### SSIM Calculation:
1. **Luminance Processing**: Convert RGB to luminance or average channels
2. **SSIM Computation**: Use skimage.metrics.structural_similarity
3. **Parameter Configuration**: Set appropriate window size and other parameters
4. **Single Score Output**: Return single SSIM value for comparison

### Experiment Logic Module

**Purpose**: Orchestrate binding validation experiments

#### Experiment Workflow:
1. **Legitimate Test**: Render bound_albedo_A + bound_normal_A vs clean reference
2. **Attack Test**: Render bound_albedo_A + bound_normal_B vs clean reference
3. **Metric Calculation**: Compute PSNR and SSIM for both tests
4. **Delta Analysis**: Calculate differences between legitimate and attack scores
5. **Result Reporting**: Return comprehensive experiment results

## Data Models

### Image Data Structure
```python
# Albedo texture: numpy array shape (height, width, 3)
# Values: float32 [0.0, 1.0]
albedo_texture: np.ndarray

# Normal map: numpy array shape (height, width, 3)
# Values: float32 [-1.0, 1.0] representing normalized vectors
normal_texture: np.ndarray

# Rendered output: numpy array shape (height, width, 3)
# Values: float32 [0.0, 1.0]
rendered_image: np.ndarray
```

### Light Direction Vector
```python
# Light direction: numpy array shape (3,)
# Values: float32 normalized vector
light_direction: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
```

### Quality Metrics Structure
```python
# Quality metrics tuple
quality_metrics: tuple = (psnr_value: float, ssim_value: float)

# Experiment results dictionary
experiment_results: dict = {
    'legitimate_psnr': float,
    'legitimate_ssim': float,
    'attack_psnr': float,
    'attack_ssim': float,
    'psnr_delta': float,
    'ssim_delta': float
}
```

### Mathematical Constants
```python
EPSILON = 1e-6      # Division by zero protection
RGB_MAX = 255.0     # Maximum RGB value for conversion
FLOAT_MAX = 1.0     # Maximum float value for PSNR calculation
```

## Error Handling

### Input Validation Errors
- **FileNotFoundError**: Handle missing texture or reference files
- **PIL.UnidentifiedImageError**: Validate image format compatibility
- **ValueError**: Check image dimension mismatches and invalid parameters
- **TypeError**: Validate parameter types (paths as strings, arrays as numpy arrays)

### Mathematical Edge Cases
- **Division by Zero**: Handle identical images in PSNR calculation
- **Invalid Vectors**: Handle zero-length normal vectors or light directions
- **Range Validation**: Ensure all pixel values remain within valid ranges
- **Memory Management**: Handle large image processing efficiently

### Rendering Edge Cases
- **Negative Dot Products**: Clamp negative lighting values to zero
- **Vector Normalization**: Handle zero-length light direction vectors
- **Dimension Mismatches**: Ensure albedo and normal textures have matching dimensions

## Testing Strategy

### Unit Testing Focus Areas

1. **Texture Loading Tests**
   - Image loading with various formats (PNG, JPEG, TIFF)
   - Normal vector unpacking accuracy and range validation
   - Dimension matching between albedo and normal textures
   - Error handling for missing or corrupted files

2. **Rendering Pipeline Tests**
   - Light direction normalization accuracy
   - Dot product calculation correctness with known inputs
   - Shading value clamping and range validation
   - Pixel composition and broadcasting verification

3. **Quality Metrics Tests**
   - PSNR calculation accuracy with known reference values
   - SSIM computation using test images with known similarity
   - Edge case handling (identical images, completely different images)
   - Metric range validation and consistency

4. **Experiment Logic Tests**
   - Complete legitimate vs attack experiment workflow
   - Delta calculation accuracy and interpretation
   - Result structure and data integrity
   - Performance with various texture sizes

### Integration Testing

1. **End-to-End Validation**
   - Complete render-to-metrics pipeline with sample textures
   - Binding effectiveness detection with known good/bad pairs
   - Performance testing with various image sizes and formats

2. **Cross-Component Integration**
   - Texture loading to rendering pipeline integration
   - Rendering output to quality metrics pipeline
   - Experiment orchestration across all components

### Performance Considerations

- **Memory Efficiency**: Process large textures without excessive memory usage
- **Computational Optimization**: Vectorized operations using NumPy
- **I/O Performance**: Efficient image loading and minimal file operations
- **Scalability**: Handle various texture resolutions efficiently