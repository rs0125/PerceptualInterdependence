# Chart Generation Tutorial

This tutorial covers advanced chart generation techniques for creating publication-quality visualizations and detailed analysis charts.

## Overview

The ChartGenerator system provides two main types of visualizations:
1. **Standard Charts**: Comprehensive 2×3 layouts showing all processing stages
2. **Zoomed Charts**: High-magnification analysis for noise pattern examination

## Standard Demonstration Charts

### Basic Chart Generation

```bash
# Generate a standard demonstration chart
perceptual-interdependence chart \
  --albedo materials/brick_albedo.png \
  --normal materials/brick_normal.png \
  --victim-id 42 \
  --attacker-id 99 \
  --output-name brick_analysis.png
```

### Chart Layout Understanding

The standard chart uses a 2×3 layout:

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Original Albedo │ Original Normal │ Original Render │
├─────────────────┼─────────────────┼─────────────────┤
│ Legitimate      │ Attack Render   │ Difference Map  │
│ Render          │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

Each panel shows:
- **Original Albedo**: Input albedo texture
- **Original Normal**: Input normal map
- **Original Render**: Baseline photometric rendering
- **Legitimate Render**: Result when legitimate user has both bound textures
- **Attack Render**: Result when attacker uses victim's albedo with their own normal
- **Difference Map**: Enhanced visualization of differences between original and attack

### Quality Metrics Overlays

Charts automatically include quality metrics:
- **PSNR (dB)**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Delta values**: Difference between legitimate and attack scenarios

## High-Magnification Analysis

### Auto-Selected Zoom Regions

```bash
# Let the system automatically select interesting regions
perceptual-interdependence zoom-chart \
  --albedo texture.png \
  --normal normal.png \
  --zoom-factor 10.0 \
  --output-name auto_zoom_10x.png
```

The system automatically:
1. Processes the full textures
2. Calculates difference maps
3. Identifies regions with significant differences
4. Selects the most interesting region for zooming

### Custom Zoom Regions

```bash
# Specify exact region coordinates
perceptual-interdependence zoom-chart \
  --albedo texture.png \
  --normal normal.png \
  --zoom-factor 15.0 \
  --zoom-region 500 400 200 200 \
  --output-name custom_zoom_15x.png
```

Region format: `X Y WIDTH HEIGHT`
- **X, Y**: Top-left corner coordinates
- **WIDTH, HEIGHT**: Region dimensions in pixels

### Multiple Magnification Levels

```bash
# Generate multiple zoom levels for comparison
for zoom in 5 10 15 20 25; do
    perceptual-interdependence zoom-chart \
      --albedo texture.png \
      --normal normal.png \
      --zoom-factor $zoom \
      --output-name analysis_${zoom}x.png
done
```

## Advanced Chart Customization

### Programmatic Chart Generation

```python
from perceptual_interdependence.utils.chart_generator import ChartGenerator

generator = ChartGenerator()

# Generate standard chart
standard_path = generator.generate_demonstration_chart(
    albedo_path="materials/wood_albedo.png",
    normal_path="materials/wood_normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="wood_analysis.png"
)

# Generate zoomed chart with custom parameters
zoomed_path = generator.generate_zoomed_demonstration_chart(
    albedo_path="materials/wood_albedo.png",
    normal_path="materials/wood_normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="wood_zoom_analysis.png",
    zoom_factor=20.0,
    zoom_region=(600, 500, 300, 300)
)
```

### Batch Chart Generation

```python
import os
from pathlib import Path

# Process multiple material types
materials = [
    ("brick", "materials/brick_albedo.png", "materials/brick_normal.png"),
    ("wood", "materials/wood_albedo.png", "materials/wood_normal.png"),
    ("metal", "materials/metal_albedo.png", "materials/metal_normal.png"),
    ("fabric", "materials/fabric_albedo.png", "materials/fabric_normal.png")
]

generator = ChartGenerator()

for name, albedo_path, normal_path in materials:
    # Standard chart
    generator.generate_demonstration_chart(
        albedo_path=albedo_path,
        normal_path=normal_path,
        victim_id=42,
        attacker_id=99,
        output_path=f"analysis_{name}_standard.png"
    )
    
    # Zoomed chart
    generator.generate_zoomed_demonstration_chart(
        albedo_path=albedo_path,
        normal_path=normal_path,
        victim_id=42,
        attacker_id=99,
        output_path=f"analysis_{name}_zoomed.png",
        zoom_factor=15.0
    )
    
    print(f"Generated charts for {name}")
```

## Research-Quality Visualizations

### Publication Settings

For academic publications, use these settings:

```bash
# High-quality chart for papers
perceptual-interdependence chart \
  --albedo research_texture.png \
  --normal research_normal.png \
  --victim-id 1001 \
  --attacker-id 2002 \
  --output-name paper_figure_1.png

# High-magnification detail for supplementary material
perceptual-interdependence zoom-chart \
  --albedo research_texture.png \
  --normal research_normal.png \
  --victim-id 1001 \
  --attacker-id 2002 \
  --zoom-factor 25.0 \
  --output-name paper_figure_2_detail.png
```

### Multi-Scenario Analysis

```python
# Compare different attack scenarios
scenarios = [
    (42, 99, "scenario_A"),
    (42, 123, "scenario_B"),
    (42, 456, "scenario_C")
]

for victim_id, attacker_id, scenario_name in scenarios:
    generator.generate_demonstration_chart(
        albedo_path="test_texture.png",
        normal_path="test_normal.png",
        victim_id=victim_id,
        attacker_id=attacker_id,
        output_path=f"comparison_{scenario_name}.png"
    )
```

### Statistical Analysis Charts

```python
# Generate charts for statistical analysis
import numpy as np

# Test multiple poison strengths
poison_strengths = [0.1, 0.15, 0.2, 0.25, 0.3]
results = []

for strength in poison_strengths:
    # Bind with custom strength
    from perceptual_interdependence import AssetBinder
    binder = AssetBinder(output_dir=f"./strength_{strength}")
    
    result = binder.bind_textures(
        albedo_path="test_texture.png",
        normal_path="test_normal.png",
        user_id=42,
        poison_strength=strength,
        output_prefix=f"strength_{strength}"
    )
    
    # Generate chart for this strength
    generator.generate_demonstration_chart(
        albedo_path=result['output_paths']['albedo'],
        normal_path=result['output_paths']['normal'],
        victim_id=42,
        attacker_id=99,
        output_path=f"strength_analysis_{strength}.png"
    )
    
    results.append((strength, result['statistics']))
```

## Difference Map Analysis

### Understanding Difference Maps

The difference map uses a custom colormap:
- **Black**: No difference (identical pixels)
- **Red**: Small differences
- **Yellow**: Moderate differences  
- **White**: Large differences

### Custom Difference Analysis

```python
# Access difference map calculation directly
from perceptual_interdependence.utils.chart_generator import ChartGenerator

generator = ChartGenerator()

# Load and process images
original_render = generator._load_texture_as_array("original_render.png")
attack_render = generator._load_texture_as_array("attack_render.png")

# Calculate different types of difference maps
l1_diff = generator._calculate_difference_map(original_render, attack_render, method='l1')
l2_diff = generator._calculate_difference_map(original_render, attack_render, method='l2')
perceptual_diff = generator._calculate_difference_map(original_render, attack_render, method='perceptual')

# Apply enhancements
enhanced_l2, _ = generator._enhance_difference_visualization(l2_diff, threshold=0.05)
rgb_diff = generator._create_difference_colormap(enhanced_l2)

# Save custom difference map
from PIL import Image
diff_image = Image.fromarray((rgb_diff * 255).astype(np.uint8))
diff_image.save("custom_difference_map.png")
```

## Performance Optimization

### Memory-Efficient Processing

For large textures, the zoomed chart generation automatically optimizes memory usage:

```python
# The system automatically crops regions before processing
# Original: 4096×4096 texture (67M pixels)
# Cropped: 600×600 region (360K pixels) - 186x less memory

generator.generate_zoomed_demonstration_chart(
    albedo_path="large_4k_texture.png",
    normal_path="large_4k_normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="large_texture_zoom.png",
    zoom_factor=10.0
)
```

### Parallel Chart Generation

```python
import concurrent.futures
from pathlib import Path

def generate_chart_pair(texture_info):
    name, albedo_path, normal_path = texture_info
    generator = ChartGenerator()
    
    # Generate both standard and zoomed charts
    standard_path = generator.generate_demonstration_chart(
        albedo_path=albedo_path,
        normal_path=normal_path,
        victim_id=42,
        attacker_id=99,
        output_path=f"{name}_standard.png"
    )
    
    zoomed_path = generator.generate_zoomed_demonstration_chart(
        albedo_path=albedo_path,
        normal_path=normal_path,
        victim_id=42,
        attacker_id=99,
        output_path=f"{name}_zoomed.png",
        zoom_factor=15.0
    )
    
    return name, standard_path, zoomed_path

# Process multiple textures in parallel
texture_list = [
    ("brick", "brick_albedo.png", "brick_normal.png"),
    ("wood", "wood_albedo.png", "wood_normal.png"),
    ("metal", "metal_albedo.png", "metal_normal.png")
]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(generate_chart_pair, texture) for texture in texture_list]
    
    for future in concurrent.futures.as_completed(futures):
        name, standard_path, zoomed_path = future.result()
        print(f"Completed charts for {name}")
```

## Quality Assessment

### Chart Quality Metrics

Generated charts include automatic quality assessment:

```python
# Access quality metrics from chart generation
results = generator.generate_demonstration_chart(
    albedo_path="texture.png",
    normal_path="normal.png",
    victim_id=42,
    attacker_id=99,
    output_path="quality_test.png"
)

# Quality metrics are displayed in the chart overlays:
# - Legitimate PSNR/SSIM (should be high)
# - Attack PSNR/SSIM (should be lower)
# - Delta values (should show significant difference)
```

### Validation Checklist

For research publications, validate your charts:

1. **Resolution**: 300 DPI for print quality
2. **Metrics**: PSNR delta > 5dB, SSIM delta > 0.1
3. **Visibility**: Difference maps show clear patterns
4. **Consistency**: Multiple runs produce similar results
5. **Documentation**: Include generation parameters

## Troubleshooting

### Common Issues

**Charts appear blurry:**
```bash
# Ensure high DPI output (automatically set to 300 DPI)
# Check that input textures have sufficient resolution
```

**Difference maps show no patterns:**
```bash
# Try higher zoom factors
perceptual-interdependence zoom-chart --zoom-factor 25.0 ...

# Or adjust poison strength
perceptual-interdependence bind --strength 0.3 ...
```

**Memory errors with large textures:**
```bash
# Use smaller zoom regions
perceptual-interdependence zoom-chart \
  --zoom-region 0 0 512 512 \
  --zoom-factor 10.0 ...
```

### Performance Tips

- **Reuse ChartGenerator instances** for multiple charts
- **Use appropriate zoom factors** (10-25x typically sufficient)
- **Process in batches** rather than individually
- **Monitor memory usage** with large textures

## Best Practices

### For Academic Papers

1. **Consistent Parameters**: Use same victim/attacker IDs across figures
2. **Multiple Examples**: Show results on different material types
3. **Clear Captions**: Explain what each panel demonstrates
4. **High Quality**: Use 300 DPI output for publications
5. **Reproducibility**: Document all generation parameters

### For Presentations

1. **High Contrast**: Zoomed charts work well for presentations
2. **Clear Labels**: Ensure text is readable at presentation size
3. **Progressive Disclosure**: Start with standard, then show zoomed details
4. **Consistent Style**: Use same color schemes across slides

### For Research Analysis

1. **Systematic Approach**: Generate charts for all test cases
2. **Statistical Validation**: Include multiple random seeds
3. **Comparative Analysis**: Show before/after or different methods
4. **Quantitative Metrics**: Include numerical results alongside visuals