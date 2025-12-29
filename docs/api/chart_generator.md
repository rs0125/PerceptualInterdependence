# ChartGenerator API Reference

The `ChartGenerator` class provides comprehensive visualization capabilities for demonstrating perceptual interdependence effects and generating publication-quality analysis charts.

## Class: ChartGenerator

### Constructor

```python
ChartGenerator()
```

Initialize ChartGenerator with core components and default configuration.

**Example:**
```python
from perceptual_interdependence.utils.chart_generator import ChartGenerator

generator = ChartGenerator()
```

### Configuration Properties

- `figure_size`: Tuple (15, 10) - Figure size in inches
- `dpi`: int 300 - Resolution for high-quality output

### Main Methods

#### generate_demonstration_chart()

```python
generate_demonstration_chart(
    albedo_path: str,
    normal_path: str,
    victim_id: int,
    attacker_id: int,
    output_path: str
) -> str
```

Generate comprehensive demonstration chart showing binding effects.

**Parameters:**
- `albedo_path` (str): Path to original albedo texture
- `normal_path` (str): Path to original normal map
- `victim_id` (int): User ID for legitimate binding scenario
- `attacker_id` (int): User ID for attack scenario
- `output_path` (str): Output path for generated chart

**Returns:**
- `str`: Absolute path to saved chart file

**Chart Layout:**
```
[Original Albedo]    [Original Normal]    [Original Render]
[Legitimate Render]  [Attack Render]      [Difference Map]
```

**Example:**
```python
chart_path = generator.generate_demonstration_chart(
    albedo_path="texture.png",
    normal_path="normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="demo_chart.png"
)
print(f"Chart saved: {chart_path}")
```

#### generate_zoomed_demonstration_chart()

```python
generate_zoomed_demonstration_chart(
    albedo_path: str,
    normal_path: str,
    victim_id: int,
    attacker_id: int,
    output_path: str,
    zoom_factor: float = 10.0,
    zoom_region: Optional[Tuple[int, int, int, int]] = None
) -> str
```

Generate high-magnification chart for noise pattern analysis.

**Parameters:**
- `albedo_path` (str): Path to original albedo texture
- `normal_path` (str): Path to original normal map
- `victim_id` (int): User ID for legitimate binding scenario
- `attacker_id` (int): User ID for attack scenario
- `output_path` (str): Output path for generated chart
- `zoom_factor` (float): Magnification factor (default: 10.0)
- `zoom_region` (Tuple[int, int, int, int], optional): (x, y, width, height) region

**Returns:**
- `str`: Absolute path to saved zoomed chart file

**Performance Optimization:**
- Automatically crops regions before processing for 10-40x speed improvement
- Processes 300×300 to 600×600 pixel regions instead of full resolution

**Example:**
```python
# Auto-selected zoom region
zoomed_path = generator.generate_zoomed_demonstration_chart(
    albedo_path="texture.png",
    normal_path="normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="zoomed_analysis.png",
    zoom_factor=15.0
)

# Custom zoom region
custom_zoomed_path = generator.generate_zoomed_demonstration_chart(
    albedo_path="texture.png",
    normal_path="normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="custom_zoom.png",
    zoom_factor=20.0,
    zoom_region=(500, 400, 200, 200)  # x, y, width, height
)
```

### Difference Map Methods

#### _calculate_difference_map()

```python
_calculate_difference_map(
    reference: np.ndarray,
    comparison: np.ndarray,
    method: str = 'l2'
) -> np.ndarray
```

Calculate pixel-wise differences with proper scaling and normalization.

**Parameters:**
- `reference` (np.ndarray): Reference image [0.0, 1.0]
- `comparison` (np.ndarray): Comparison image [0.0, 1.0]
- `method` (str): Difference method ('l1', 'l2', 'perceptual')

**Returns:**
- `np.ndarray`: Normalized difference map [0.0, 1.0]

**Methods:**
- `'l1'`: Manhattan distance |ref - comp|
- `'l2'`: Euclidean distance (ref - comp)²
- `'perceptual'`: Luminance-weighted difference

#### _enhance_difference_visualization()

```python
_enhance_difference_visualization(
    difference_map: np.ndarray,
    threshold: float = 0.1,
    colormap: str = 'hot'
) -> Tuple[np.ndarray, str]
```

Apply visual enhancements to difference maps.

**Parameters:**
- `difference_map` (np.ndarray): Input difference map [0.0, 1.0]
- `threshold` (float): Threshold for highlighting significant changes
- `colormap` (str): Matplotlib colormap name

**Returns:**
- `Tuple[np.ndarray, str]`: Enhanced difference map and colormap name

**Enhancements:**
- Threshold-based highlighting
- Adaptive contrast stretching
- Histogram equalization
- Optional smoothing

#### _create_difference_colormap()

```python
_create_difference_colormap(
    difference_map: np.ndarray
) -> np.ndarray
```

Create custom RGB colormap for difference visualization.

**Parameters:**
- `difference_map` (np.ndarray): Normalized difference map [0.0, 1.0]

**Returns:**
- `np.ndarray`: RGB image array for visualization

**Color Mapping:**
- Black (0.0) → Red (0.33) → Yellow (0.66) → White (1.0)
- Provides excellent contrast for academic presentations

### Zoom and Region Methods

#### _select_optimal_zoom_region()

```python
_select_optimal_zoom_region(
    assets: AssetBundle,
    renders: RenderResults
) -> Tuple[int, int, int, int]
```

Automatically select optimal region for zooming based on difference analysis.

**Parameters:**
- `assets` (AssetBundle): Processed texture assets
- `renders` (RenderResults): Rendered images

**Returns:**
- `Tuple[int, int, int, int]`: (x, y, width, height) for zoom region

**Algorithm:**
- Calculates difference maps to find interesting regions
- Uses morphological operations to find connected regions
- Selects largest region with significant differences
- Falls back to center region if no differences found

#### _extract_zoom_region()

```python
_extract_zoom_region(
    image: np.ndarray,
    zoom_region: Tuple[int, int, int, int],
    zoom_factor: float
) -> np.ndarray
```

Extract and interpolate zoom region with high-quality resampling.

**Parameters:**
- `image` (np.ndarray): Source image array
- `zoom_region` (Tuple[int, int, int, int]): Region coordinates
- `zoom_factor` (float): Magnification factor

**Returns:**
- `np.ndarray`: High-quality zoomed image region

**Quality Features:**
- Lanczos interpolation for crisp results
- Proper handling of RGB and grayscale images
- Maintains aspect ratio and quality

### Utility Methods

#### _prepare_normal_for_display()

```python
_prepare_normal_for_display(
    normal_array: np.ndarray
) -> np.ndarray
```

Prepare normal map for display by ensuring [0,1] range.

#### _add_metrics_overlay()

```python
_add_metrics_overlay(
    ax: plt.Axes,
    text: str,
    color: str = 'white'
) -> None
```

Add text overlay with metrics to subplot.

#### _save_chart()

```python
_save_chart(
    fig: plt.Figure,
    output_path: str
) -> str
```

Save chart with high-resolution PNG output and proper cleanup.

## Usage Patterns

### Basic Chart Generation

```python
from perceptual_interdependence.utils.chart_generator import ChartGenerator

generator = ChartGenerator()

# Standard demonstration chart
chart_path = generator.generate_demonstration_chart(
    albedo_path="materials/brick_albedo.png",
    normal_path="materials/brick_normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="brick_analysis.png"
)
```

### High-Magnification Analysis

```python
# Generate multiple zoom levels
zoom_factors = [5.0, 10.0, 15.0, 25.0]

for zoom in zoom_factors:
    zoomed_path = generator.generate_zoomed_demonstration_chart(
        albedo_path="texture.png",
        normal_path="normal.png",
        victim_id=42,
        attacker_id=99,
        output_path=f"analysis_{zoom:.0f}x.png",
        zoom_factor=zoom
    )
    print(f"Generated {zoom}x magnification chart")
```

### Batch Processing for Research

```python
import os
from pathlib import Path

# Process multiple texture pairs
texture_pairs = [
    ("brick_albedo.png", "brick_normal.png"),
    ("wood_albedo.png", "wood_normal.png"),
    ("metal_albedo.png", "metal_normal.png")
]

for i, (albedo, normal) in enumerate(texture_pairs):
    # Standard chart
    generator.generate_demonstration_chart(
        albedo_path=albedo,
        normal_path=normal,
        victim_id=42,
        attacker_id=99,
        output_path=f"analysis_{i:02d}_standard.png"
    )
    
    # Zoomed chart
    generator.generate_zoomed_demonstration_chart(
        albedo_path=albedo,
        normal_path=normal,
        victim_id=42,
        attacker_id=99,
        output_path=f"analysis_{i:02d}_zoomed.png",
        zoom_factor=15.0
    )
```

### Custom Region Analysis

```python
# Analyze specific regions of interest
regions_of_interest = [
    (100, 100, 300, 300),  # Top-left region
    (500, 500, 200, 200),  # Center region
    (800, 600, 400, 400)   # Bottom-right region
]

for i, region in enumerate(regions_of_interest):
    generator.generate_zoomed_demonstration_chart(
        albedo_path="large_texture.png",
        normal_path="large_normal.png",
        victim_id=42,
        attacker_id=99,
        output_path=f"region_{i}_analysis.png",
        zoom_factor=20.0,
        zoom_region=region
    )
```

## Performance Considerations

### Standard Charts
- **2048×2048 textures**: ~45 seconds generation time
- **Memory usage**: ~6GB peak for full processing pipeline
- **Output size**: 10-15MB PNG files at 300 DPI

### Zoomed Charts
- **Cropped processing**: 30-60 seconds (10-40x faster than full resolution)
- **Memory usage**: ~500MB-2GB depending on crop size
- **Output size**: 1-8MB PNG files

### Optimization Tips

```python
# Reuse generator instance for multiple charts
generator = ChartGenerator()

# Process multiple charts with same generator
for texture_pair in texture_list:
    chart_path = generator.generate_demonstration_chart(*texture_pair)
    # Generator reuses internal components
```

## Error Handling

```python
try:
    chart_path = generator.generate_demonstration_chart(
        albedo_path="texture.png",
        normal_path="normal.png",
        victim_id=42,
        attacker_id=99,
        output_path="chart.png"
    )
except FileNotFoundError as e:
    print(f"Input file not found: {e}")
except ValueError as e:
    print(f"Processing error: {e}")
except IOError as e:
    print(f"Output error: {e}")
finally:
    # Generator automatically cleans up temporary files
    pass
```

## Output Quality

- **Resolution**: 300 DPI for publication quality
- **Format**: PNG with lossless compression
- **Color space**: sRGB for consistent display
- **File size**: Optimized for quality vs. size balance
- **Compatibility**: Works with all major image viewers and document systems