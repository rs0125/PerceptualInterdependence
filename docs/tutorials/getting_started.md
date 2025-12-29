# Getting Started Tutorial

This tutorial will guide you through the basic usage of the Perceptual Interdependence system, from installation to generating your first demonstration charts.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large textures)
- Basic familiarity with command-line interfaces

## Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/research/perceptual-interdependence.git
cd perceptual-interdependence

# Install the package
pip install -e .
```

### Verify Installation

```bash
# Check if the system is working
perceptual-interdependence --help

# Run a quick system check
python -c "
from perceptual_interdependence.utils.validation import ValidationSuite
validator = ValidationSuite()
results = validator.validate_system_integrity()
print('System status:', 'READY' if results['valid'] else 'ERROR')
"
```

## Your First Binding Operation

Let's start with a simple texture binding operation using the sample textures.

### Step 1: Examine Sample Data

```bash
# List available sample textures
ls data/samples/

# You should see files like:
# - original_albedo.png (sample albedo texture)
# - original_normal.png (sample normal map)
# - church_bricks_03_diff_2k.jpg (church bricks albedo)
# - church_bricks_03_nor_gl_2k.jpg (church bricks normal)
```

### Step 2: Bind Textures for a User

```bash
# Bind the sample textures for user ID 42
perceptual-interdependence bind \
  --albedo data/samples/original_albedo.png \
  --normal data/samples/original_normal.png \
  --user-id 42 \
  --output-dir ./my_first_binding

# This creates:
# - my_first_binding/bound_albedo_42.png
# - my_first_binding/bound_normal_42.png
```

**Expected Output:**
```
Binding textures for user 42
AssetBinder: CPU acceleration (Numba JIT) enabled
Starting Analytically Safe One-Way Binding (user_id: 42)...
  Loaded textures: (1971, 2048, 3)
  Generated poison map: range [0.0, 0.2]
  Applied poison to albedo (with saturation handling)
  Calculated analytical antidote normal map
  Saved bound assets: bound_albedo_42.png, bound_normal_42.png
Assets bound successfully!
   Saturation ratio: 0.0%
```

### Step 3: Understand What Happened

The binding operation:
1. **Loaded** your input textures
2. **Generated** a user-specific poison map (based on user ID 42)
3. **Applied poison** to the albedo texture (brightening effect)
4. **Calculated antidote** for the normal map (steepening effect)
5. **Saved** the bound textures

The mathematical relationship ensures: `Poisoned_Albedo × Antidote_Normal = Original_Albedo × Original_Normal`

## Generate Your First Demonstration Chart

Now let's create a visualization to see the binding effects.

### Step 4: Create a Standard Chart

```bash
# Generate a demonstration chart
perceptual-interdependence chart \
  --albedo data/samples/original_albedo.png \
  --normal data/samples/original_normal.png \
  --victim-id 42 \
  --attacker-id 99 \
  --output-name my_first_chart.png
```

This creates a comprehensive 2×3 chart showing:
- **Top row**: Original albedo, original normal, original render
- **Bottom row**: Legitimate render (user 42), attack render (user 99), difference map

**Expected Output:**
```
Generating demonstration chart
Processing legitimate binding scenario (victim_id: 42)...
Processing attack binding scenario (attacker_id: 99)...
Rendering original scenario...
Rendering legitimate scenario...
Rendering attack scenario...
Quality metrics calculated:
  Legitimate: PSNR=44.22dB, SSIM=0.9827
  Attack: PSNR=27.85dB, SSIM=0.7269
  Deltas: PSNR=16.38dB, SSIM=0.2558
Chart generated successfully!
```

### Step 5: Examine the Results

Open `my_first_chart.png` in an image viewer. You should see:

1. **High quality preservation** for legitimate user (PSNR ~44dB)
2. **Significant degradation** for attacker (PSNR ~28dB)
3. **Clear difference map** showing where the security mechanism activates

## High-Magnification Analysis

Let's examine the noise patterns in detail.

### Step 6: Generate a Zoomed Chart

```bash
# Create a 15x magnification chart
perceptual-interdependence zoom-chart \
  --albedo data/samples/original_albedo.png \
  --normal data/samples/original_normal.png \
  --zoom-factor 15.0 \
  --output-name my_first_zoom_chart.png
```

This automatically:
- Selects an interesting region with differences
- Crops the region for faster processing
- Generates a 15x magnified view
- Shows the subtle noise patterns that provide security

## Forensic Analysis

Now let's see how the system can detect unauthorized usage.

### Step 7: Perform Forensic Detection

```bash
# Use the bound texture we created earlier for forensic analysis
perceptual-interdependence forensic \
  --suspicious my_first_binding/bound_albedo_42.png \
  --original data/samples/original_albedo.png \
  --max-users 50 \
  --generate-spike-chart \
  --output-chart my_first_forensic_chart.png
```

**Expected Output:**
```
Extracting noise signature from suspicious texture...
Running correlation analysis across 50 users...
Detected traitor: User ID 42
Detection score: 4180521.750000 (Z-score: 12.20)
```

The system correctly identifies user 42 as the source of the bound texture!

## Understanding the Results

### Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
  - Legitimate users: ~40-45dB (excellent quality)
  - Attackers: ~25-30dB (noticeable degradation)

- **SSIM (Structural Similarity Index)**: Closer to 1.0 is better
  - Legitimate users: ~0.95-0.99 (nearly identical)
  - Attackers: ~0.70-0.85 (visible differences)

### Security Properties

- **Perfect Restoration**: Legitimate users get mathematically perfect results
- **Guaranteed Degradation**: Attackers always experience quality loss
- **Forensic Traceability**: Unauthorized usage can be traced back to specific users

## Next Steps

Now that you've completed the basic tutorial, you can:

1. **Try Different Textures**: Use your own albedo and normal map pairs
2. **Experiment with Parameters**: Adjust poison strength, user IDs, zoom factors
3. **Batch Processing**: Process multiple textures or users
4. **Advanced Analysis**: Explore the research workflows and custom algorithms

### Try with Your Own Textures

```bash
# Replace with your texture files
perceptual-interdependence bind \
  --albedo your_albedo.png \
  --normal your_normal.png \
  --user-id 123 \
  --strength 0.25

perceptual-interdependence chart \
  --albedo your_albedo.png \
  --normal your_normal.png \
  --victim-id 123 \
  --attacker-id 456
```

### Batch Process Multiple Users

```bash
# Create a simple batch script
for user_id in {1..10}; do
    perceptual-interdependence bind \
      --albedo data/samples/original_albedo.png \
      --normal data/samples/original_normal.png \
      --user-id $user_id \
      --prefix user_$user_id \
      --output-dir ./batch_results
done
```

## Troubleshooting

### Common Issues

**"Command not found"**
```bash
# Make sure the package is installed
pip install -e .

# Or use python -m
python -m perceptual_interdependence.cli.main --help
```

**"File not found"**
```bash
# Check file paths
ls -la data/samples/
# Use absolute paths if needed
perceptual-interdependence bind --albedo $(pwd)/data/samples/original_albedo.png ...
```

**Memory issues with large textures**
```bash
# Use smaller zoom regions for large textures
perceptual-interdependence zoom-chart \
  --albedo large_texture.png \
  --normal large_normal.png \
  --zoom-region 0 0 512 512
```

### Performance Tips

- **Reuse bound textures**: Save bound textures and reuse them for multiple charts
- **Use appropriate zoom factors**: Higher magnification requires more processing
- **Monitor memory usage**: Large textures (>4K) may require significant RAM

### Getting Help

```bash
# Command-specific help
perceptual-interdependence bind --help
perceptual-interdependence chart --help

# System validation
python -c "
from perceptual_interdependence.utils.validation import ValidationSuite
validator = ValidationSuite()
results = validator.validate_system_integrity()
print('Validation results:', results)
"
```

## What's Next?

Continue with these tutorials:

- **[Advanced Binding](advanced_binding.md)**: Custom binding scenarios and parameters
- **[Chart Generation](chart_generation.md)**: Advanced visualization techniques
- **[Performance Optimization](performance.md)**: Optimizing for large-scale processing
- **[Research Workflows](research.md)**: Using the system for academic research

Or explore the API documentation for programmatic usage:

- **[AssetBinder API](../api/asset_binder.md)**: Core binding functionality
- **[ChartGenerator API](../api/chart_generator.md)**: Visualization capabilities
- **[Forensics API](../api/forensics.md)**: Traitor detection and analysis