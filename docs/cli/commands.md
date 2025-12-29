# CLI Reference

The Perceptual Interdependence system provides a comprehensive command-line interface for all operations including binding, visualization, forensic analysis, and performance testing.

## Global Usage

```bash
perceptual-interdependence <command> [options]
```

## Available Commands

- [`bind`](#bind) - Bind texture assets for specific users
- [`chart`](#chart) - Generate demonstration charts
- [`zoom-chart`](#zoom-chart) - Generate high-magnification analysis charts
- [`forensic`](#forensic) - Perform forensic traitor detection
- [`experiment`](#experiment) - Run comprehensive experimental pipeline
- [`benchmark`](#benchmark) - Run performance benchmarks
- [`gui`](#gui) - Launch interactive web interface
- [`demo`](#demo) - Run demonstration with sample textures

## Command Details

### bind

Bind albedo and normal textures for a specific user using Analytically Safe One-Way Binding.

```bash
perceptual-interdependence bind --albedo ALBEDO --normal NORMAL --user-id USER_ID [options]
```

**Required Arguments:**
- `--albedo ALBEDO` - Path to albedo texture file
- `--normal NORMAL` - Path to normal map file  
- `--user-id USER_ID` - User ID for binding (used as random seed)

**Optional Arguments:**
- `--strength STRENGTH` - Poison strength [0.0-1.0] (default: 0.2)
- `--output-dir OUTPUT_DIR` - Output directory (default: current directory)
- `--prefix PREFIX` - Output filename prefix (default: "bound")

**Examples:**
```bash
# Basic binding
perceptual-interdependence bind --albedo texture.png --normal normal.png --user-id 42

# Custom strength and output
perceptual-interdependence bind --albedo brick.png --normal brick_n.png \
  --user-id 123 --strength 0.25 --output-dir ./results --prefix user123

# High security binding
perceptual-interdependence bind --albedo sensitive.png --normal sensitive_n.png \
  --user-id 999 --strength 0.3
```

**Output:**
- `{prefix}_albedo_{user_id}.png` - Bound albedo texture
- `{prefix}_normal_{user_id}.png` - Bound normal map
- Console output with binding statistics

---

### chart

Generate comprehensive demonstration charts showing the effects of binding operations.

```bash
perceptual-interdependence chart --albedo ALBEDO --normal NORMAL [options]
```

**Required Arguments:**
- `--albedo ALBEDO` - Path to albedo texture file
- `--normal NORMAL` - Path to normal map file

**Optional Arguments:**
- `--victim-id VICTIM_ID` - User ID for legitimate binding scenario (default: 42)
- `--attacker-id ATTACKER_ID` - User ID for attack scenario (default: 99)
- `--output-dir OUTPUT_DIR` - Output directory (default: current directory)
- `--output-name OUTPUT_NAME` - Chart filename (default: "demonstration_chart.png")

**Examples:**
```bash
# Basic demonstration chart
perceptual-interdependence chart --albedo texture.png --normal normal.png

# Custom user IDs and output
perceptual-interdependence chart --albedo brick.png --normal brick_n.png \
  --victim-id 123 --attacker-id 456 --output-name brick_analysis.png

# Research chart with specific scenarios
perceptual-interdependence chart --albedo material.png --normal material_n.png \
  --victim-id 1001 --attacker-id 2002 --output-dir ./research_results
```

**Output:**
- High-resolution PNG chart (300 DPI) with 2×3 layout:
  - Original Albedo, Original Normal, Original Render
  - Legitimate Render, Attack Render, Difference Map
- Quality metrics overlays (PSNR, SSIM)
- Console output with processing statistics

---

### zoom-chart

Generate high-magnification charts for detailed noise pattern analysis.

```bash
perceptual-interdependence zoom-chart --albedo ALBEDO --normal NORMAL [options]
```

**Required Arguments:**
- `--albedo ALBEDO` - Path to albedo texture file
- `--normal NORMAL` - Path to normal map file

**Optional Arguments:**
- `--victim-id VICTIM_ID` - User ID for legitimate binding scenario (default: 42)
- `--attacker-id ATTACKER_ID` - User ID for attack scenario (default: 99)
- `--zoom-factor ZOOM_FACTOR` - Magnification factor (default: 10.0)
- `--zoom-region X Y WIDTH HEIGHT` - Specific region coordinates
- `--output-dir OUTPUT_DIR` - Output directory (default: current directory)
- `--output-name OUTPUT_NAME` - Chart filename (default: "zoomed_demonstration_chart.png")

**Examples:**
```bash
# Auto-selected zoom region at 10x
perceptual-interdependence zoom-chart --albedo texture.png --normal normal.png

# High magnification analysis
perceptual-interdependence zoom-chart --albedo brick.png --normal brick_n.png \
  --zoom-factor 25.0 --output-name brick_25x_analysis.png

# Custom region analysis
perceptual-interdependence zoom-chart --albedo large_texture.png --normal large_normal.png \
  --zoom-factor 15.0 --zoom-region 500 400 200 200

# Research-grade analysis
perceptual-interdependence zoom-chart --albedo material.png --normal material_n.png \
  --victim-id 1001 --attacker-id 2002 --zoom-factor 20.0
```

**Performance:**
- Optimized processing: 30-60 seconds (vs. several minutes for full resolution)
- Automatic region cropping for 10-40x speed improvement
- Memory efficient: processes 300×300 to 600×600 pixel regions

**Output:**
- High-magnification chart showing noise patterns
- Same 2×3 layout as standard charts but zoomed
- Region information in difference map overlay

---

### forensic

Perform forensic traitor detection analysis on suspicious textures.

```bash
perceptual-interdependence forensic --suspicious SUSPICIOUS --original ORIGINAL [options]
```

**Required Arguments:**
- `--suspicious SUSPICIOUS` - Path to suspicious albedo texture file
- `--original ORIGINAL` - Path to original clean texture file

**Optional Arguments:**
- `--max-users MAX_USERS` - Maximum number of users to test (default: 100)
- `--output-chart OUTPUT_CHART` - Output filename for spike chart (default: "forensic_spike_chart.png")
- `--generate-spike-chart` - Generate continuous spike chart visualization

**Examples:**
```bash
# Basic forensic analysis
perceptual-interdependence forensic --suspicious bound_texture.png --original clean_texture.png

# Extended user range with visualization
perceptual-interdependence forensic --suspicious suspicious.png --original original.png \
  --max-users 150 --generate-spike-chart --output-chart forensic_analysis.png

# High-precision detection
perceptual-interdependence forensic --suspicious evidence.png --original baseline.png \
  --max-users 500 --generate-spike-chart
```

**Output:**
- Detected traitor user ID
- Correlation score and Z-score statistics
- Optional continuous spike chart visualization
- Statistical significance assessment

**Performance:**
- Analysis time: ~30-60 seconds for 100 users
- Memory usage: ~2-4GB depending on texture size
- Scales linearly with max-users parameter

---

### experiment

Run comprehensive experimental pipeline for research purposes.

```bash
perceptual-interdependence experiment [options]
```

**Optional Arguments:**
- `--victim-id VICTIM_ID` - Victim user ID (default: 42)
- `--attacker-id ATTACKER_ID` - Attacker user ID (default: 99)
- `--max-users MAX_USERS` - Maximum users for forensics (default: 100)
- `--output-dir OUTPUT_DIR` - Results output directory (default: "results")

**Examples:**
```bash
# Basic experiment
perceptual-interdependence experiment

# Custom experimental parameters
perceptual-interdependence experiment --victim-id 1001 --attacker-id 2002 \
  --max-users 500 --output-dir ./research_data

# Large-scale experiment
perceptual-interdependence experiment --max-users 1000 --output-dir ./large_scale_results
```

**Output:**
- Comprehensive experimental results
- Performance metrics and analysis
- Quality assessment data
- Forensic analysis results

---

### benchmark

Run performance benchmarks to evaluate system performance.

```bash
perceptual-interdependence benchmark [options]
```

**Optional Arguments:**
- `--size WIDTH HEIGHT` - Image size for benchmarking (default: 2048 2048)
- `--iterations ITERATIONS` - Number of benchmark iterations (default: 5)

**Examples:**
```bash
# Standard benchmark
perceptual-interdependence benchmark

# Custom image size
perceptual-interdependence benchmark --size 4096 4096 --iterations 10

# Quick performance test
perceptual-interdependence benchmark --size 1024 1024 --iterations 3
```

**Output:**
- Processing time statistics (mean, min, max, std dev)
- Throughput measurements (Mpixels/sec)
- Performance comparison with baseline
- System configuration details

---

### gui

Launch interactive Streamlit-based web interface.

```bash
perceptual-interdependence gui [options]
```

**Optional Arguments:**
- `--port PORT` - Port for Streamlit server (default: 8501)
- `--host HOST` - Host for Streamlit server (default: "localhost")

**Examples:**
```bash
# Launch on default port
perceptual-interdependence gui

# Custom port and host
perceptual-interdependence gui --port 8080 --host 0.0.0.0

# Development server
perceptual-interdependence gui --port 3000
```

**Features:**
- Interactive texture upload and processing
- Real-time parameter adjustment
- Visual result comparison
- Chart generation interface
- Forensic analysis tools

---

### demo

Run demonstration with sample textures.

```bash
perceptual-interdependence demo [options]
```

**Optional Arguments:**
- `--texture-dir TEXTURE_DIR` - Directory containing sample textures (default: "data/samples")
- `--output-dir OUTPUT_DIR` - Demo output directory (default: "demo_results")

**Examples:**
```bash
# Basic demo
perceptual-interdependence demo

# Custom directories
perceptual-interdependence demo --texture-dir ./my_samples --output-dir ./demo_output
```

**Output:**
- Demonstration results with sample textures
- Example charts and visualizations
- Performance metrics
- Tutorial-style output

## Global Options

All commands support these global options:

- `-h, --help` - Show help message and exit
- `--version` - Show version information

## Environment Variables

Configure system behavior with environment variables:

```bash
# Disable Numba JIT compilation
export PERCEPTUAL_NUMBA_DISABLE=1

# Set default output directory
export PERCEPTUAL_OUTPUT_DIR=/path/to/output

# Set logging level
export PERCEPTUAL_LOG_LEVEL=DEBUG
```

## Exit Codes

- `0` - Success
- `1` - General error (invalid arguments, processing failure)
- `2` - File not found error
- `3` - Permission error
- `130` - Interrupted by user (Ctrl+C)

## Performance Tips

### Memory Management
```bash
# For large textures, monitor memory usage
perceptual-interdependence bind --albedo large_texture.png --normal large_normal.png --user-id 42
```

### Batch Processing
```bash
# Process multiple textures efficiently
for i in {1..100}; do
    perceptual-interdependence bind --albedo texture.png --normal normal.png --user-id $i --prefix batch_$i
done
```

### Parallel Execution
```bash
# Run multiple commands in parallel (be mindful of memory usage)
perceptual-interdependence chart --albedo tex1.png --normal norm1.png --output-name chart1.png &
perceptual-interdependence chart --albedo tex2.png --normal norm2.png --output-name chart2.png &
wait
```

## Troubleshooting

### Common Issues

**File Not Found:**
```bash
# Ensure file paths are correct and files exist
ls -la texture.png normal.png
perceptual-interdependence bind --albedo texture.png --normal normal.png --user-id 42
```

**Memory Issues:**
```bash
# For large textures, use smaller zoom regions
perceptual-interdependence zoom-chart --albedo large.png --normal large_n.png \
  --zoom-region 0 0 512 512
```

**Permission Errors:**
```bash
# Ensure write permissions for output directory
mkdir -p ./results
chmod 755 ./results
perceptual-interdependence bind --albedo texture.png --normal normal.png \
  --user-id 42 --output-dir ./results
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
export PERCEPTUAL_LOG_LEVEL=DEBUG
perceptual-interdependence bind --albedo texture.png --normal normal.png --user-id 42
```

### Performance Issues

Check system requirements and optimization:

```bash
# Run benchmark to check performance
perceptual-interdependence benchmark

# Check if Numba JIT is working
python -c "from perceptual_interdependence.algorithms.cpu_math import get_cpu_math; print('Numba available:', get_cpu_math().numba_available)"
```