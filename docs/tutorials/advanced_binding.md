# Advanced Binding Tutorial

This tutorial covers advanced binding scenarios, custom parameters, and specialized use cases for the Perceptual Interdependence system.

## Custom Poison Strength

### Understanding Poison Strength

Poison strength controls the trade-off between security and quality:
- **Lower values (0.1-0.2)**: Better quality preservation, moderate security
- **Higher values (0.25-0.3)**: Stronger security, more noticeable effects

### Adaptive Poison Strength

```bash
# Test different poison strengths
for strength in 0.1 0.15 0.2 0.25 0.3; do
    perceptual-interdependence bind \
      --albedo texture.png \
      --normal normal.png \
      --user-id 42 \
      --strength $strength \
      --prefix "strength_${strength}" \
      --output-dir ./strength_analysis
done
```

### Programmatic Strength Selection

```python
from perceptual_interdependence import AssetBinder
import numpy as np

def select_optimal_strength(albedo_path, normal_path, target_psnr=35.0):
    """Select poison strength to achieve target PSNR."""
    binder = AssetBinder()
    
    strengths = np.linspace(0.1, 0.3, 20)
    results = []
    
    for strength in strengths:
        result = binder.bind_textures(
            albedo_path=albedo_path,
            normal_path=normal_path,
            user_id=42,
            poison_strength=strength
        )
        
        # Calculate quality metrics (simplified)
        saturation_ratio = result['statistics']['saturation_ratio']
        estimated_psnr = 50 - (strength * 100)  # Rough estimate
        
        results.append((strength, estimated_psnr, saturation_ratio))
        
        if estimated_psnr <= target_psnr:
            return strength
    
    return strengths[-1]  # Return maximum if target not reached

# Usage
optimal_strength = select_optimal_strength("texture.png", "normal.png", target_psnr=30.0)
print(f"Optimal poison strength: {optimal_strength:.3f}")
```

## Batch Processing Workflows

### Multi-User Binding

```bash
#!/bin/bash
# Batch process for multiple users

ALBEDO="materials/brick_albedo.png"
NORMAL="materials/brick_normal.png"
OUTPUT_DIR="./multi_user_results"

mkdir -p $OUTPUT_DIR

# Process users 1-100
for user_id in {1..100}; do
    echo "Processing user $user_id..."
    
    perceptual-interdependence bind \
      --albedo $ALBEDO \
      --normal $NORMAL \
      --user-id $user_id \
      --strength 0.2 \
      --prefix "user_$(printf "%03d" $user_id)" \
      --output-dir $OUTPUT_DIR
    
    # Generate individual chart
    perceptual-interdependence chart \
      --albedo $ALBEDO \
      --normal $NORMAL \
      --victim-id $user_id \
      --attacker-id 999 \
      --output-name "${OUTPUT_DIR}/chart_user_$(printf "%03d" $user_id).png"
done

echo "Batch processing complete: $OUTPUT_DIR"
```

### Multi-Texture Processing

```python
import os
from pathlib import Path
from perceptual_interdependence import AssetBinder

def process_texture_library(texture_dir, output_dir, user_ids):
    """Process entire texture library for multiple users."""
    
    texture_dir = Path(texture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all texture pairs
    albedo_files = list(texture_dir.glob("*_albedo.*"))
    texture_pairs = []
    
    for albedo_file in albedo_files:
        # Look for corresponding normal map
        base_name = albedo_file.stem.replace("_albedo", "")
        normal_candidates = [
            texture_dir / f"{base_name}_normal{albedo_file.suffix}",
            texture_dir / f"{base_name}_n{albedo_file.suffix}",
            texture_dir / f"{base_name}_norm{albedo_file.suffix}"
        ]
        
        for normal_file in normal_candidates:
            if normal_file.exists():
                texture_pairs.append((base_name, albedo_file, normal_file))
                break
    
    print(f"Found {len(texture_pairs)} texture pairs")
    
    # Process each texture for each user
    binder = AssetBinder()
    results = {}
    
    for texture_name, albedo_path, normal_path in texture_pairs:
        print(f"Processing texture: {texture_name}")
        texture_results = {}
        
        for user_id in user_ids:
            user_output_dir = output_dir / texture_name / f"user_{user_id:03d}"
            user_output_dir.mkdir(parents=True, exist_ok=True)
            
            result = binder.bind_textures(
                albedo_path=str(albedo_path),
                normal_path=str(normal_path),
                user_id=user_id,
                poison_strength=0.2,
                output_prefix=f"{texture_name}_user_{user_id:03d}"
            )
            
            texture_results[user_id] = result
        
        results[texture_name] = texture_results
    
    return results

# Usage
texture_library_results = process_texture_library(
    texture_dir="./texture_library",
    output_dir="./processed_library", 
    user_ids=range(1, 51)  # Process users 1-50
)
```

## Advanced Security Scenarios

### Multi-Layer Binding

```python
def multi_layer_binding(albedo_path, normal_path, user_layers):
    """Apply multiple binding layers for enhanced security."""
    
    current_albedo = albedo_path
    current_normal = normal_path
    
    binder = AssetBinder()
    layer_results = []
    
    for layer_idx, (user_id, strength) in enumerate(user_layers):
        print(f"Applying layer {layer_idx + 1}: User {user_id}, Strength {strength}")
        
        result = binder.bind_textures(
            albedo_path=current_albedo,
            normal_path=current_normal,
            user_id=user_id,
            poison_strength=strength,
            output_prefix=f"layer_{layer_idx + 1}"
        )
        
        # Use output as input for next layer
        current_albedo = result['output_paths']['albedo']
        current_normal = result['output_paths']['normal']
        
        layer_results.append(result)
    
    return layer_results

# Usage: Apply three security layers
layers = [
    (42, 0.15),   # Primary user, moderate strength
    (123, 0.1),   # Secondary user, light strength  
    (999, 0.05)   # Tertiary user, minimal strength
]

multi_layer_results = multi_layer_binding(
    "sensitive_texture.png",
    "sensitive_normal.png", 
    layers
)
```

### Hierarchical User Management

```python
class HierarchicalBinding:
    """Manage hierarchical user binding with inheritance."""
    
    def __init__(self):
        self.user_hierarchy = {}
        self.binding_cache = {}
    
    def add_user(self, user_id, parent_id=None, access_level=1.0):
        """Add user to hierarchy."""
        self.user_hierarchy[user_id] = {
            'parent': parent_id,
            'access_level': access_level,
            'children': []
        }
        
        if parent_id and parent_id in self.user_hierarchy:
            self.user_hierarchy[parent_id]['children'].append(user_id)
    
    def calculate_effective_strength(self, user_id, base_strength=0.2):
        """Calculate effective poison strength based on hierarchy."""
        if user_id not in self.user_hierarchy:
            return base_strength
        
        access_level = self.user_hierarchy[user_id]['access_level']
        return base_strength * access_level
    
    def bind_for_user(self, albedo_path, normal_path, user_id):
        """Bind textures considering user hierarchy."""
        effective_strength = self.calculate_effective_strength(user_id)
        
        binder = AssetBinder()
        return binder.bind_textures(
            albedo_path=albedo_path,
            normal_path=normal_path,
            user_id=user_id,
            poison_strength=effective_strength,
            output_prefix=f"hierarchical_user_{user_id}"
        )

# Usage
hierarchy = HierarchicalBinding()

# Add users with different access levels
hierarchy.add_user(1, access_level=1.0)      # Full access
hierarchy.add_user(2, parent_id=1, access_level=0.8)  # 80% access
hierarchy.add_user(3, parent_id=1, access_level=0.6)  # 60% access
hierarchy.add_user(4, parent_id=2, access_level=0.4)  # 40% access (inherited)

# Bind for different users
for user_id in [1, 2, 3, 4]:
    result = hierarchy.bind_for_user("texture.png", "normal.png", user_id)
    print(f"User {user_id}: Effective strength = {hierarchy.calculate_effective_strength(user_id):.3f}")
```

## Custom Algorithms

### Texture-Adaptive Binding

```python
def texture_adaptive_binding(albedo_path, normal_path, user_id):
    """Adapt binding parameters based on texture characteristics."""
    
    from perceptual_interdependence.utils.texture_processing import TextureProcessor
    
    # Analyze texture properties
    processor = TextureProcessor()
    albedo_array = processor.load_texture(albedo_path)
    
    # Calculate texture complexity metrics
    complexity = calculate_texture_complexity(albedo_array)
    contrast = calculate_local_contrast(albedo_array)
    frequency_content = analyze_frequency_content(albedo_array)
    
    # Adapt poison strength based on texture properties
    base_strength = 0.2
    
    # Increase strength for low-complexity textures (easier to hide noise)
    if complexity < 0.3:
        strength_multiplier = 1.2
    elif complexity > 0.7:
        strength_multiplier = 0.8  # Reduce for high-complexity textures
    else:
        strength_multiplier = 1.0
    
    # Adjust for contrast
    if contrast < 0.2:
        strength_multiplier *= 1.1  # Increase for low-contrast textures
    
    adaptive_strength = base_strength * strength_multiplier
    adaptive_strength = np.clip(adaptive_strength, 0.1, 0.3)
    
    print(f"Texture analysis:")
    print(f"  Complexity: {complexity:.3f}")
    print(f"  Contrast: {contrast:.3f}")
    print(f"  Adaptive strength: {adaptive_strength:.3f}")
    
    # Perform binding with adaptive parameters
    binder = AssetBinder()
    return binder.bind_textures(
        albedo_path=albedo_path,
        normal_path=normal_path,
        user_id=user_id,
        poison_strength=adaptive_strength,
        output_prefix="adaptive"
    )

def calculate_texture_complexity(texture_array):
    """Calculate texture complexity using gradient magnitude."""
    from scipy import ndimage
    
    # Convert to grayscale
    if len(texture_array.shape) == 3:
        gray = np.mean(texture_array, axis=2)
    else:
        gray = texture_array
    
    # Calculate gradients
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize complexity measure
    complexity = np.mean(gradient_magnitude)
    return np.clip(complexity, 0, 1)

def calculate_local_contrast(texture_array):
    """Calculate local contrast using standard deviation."""
    if len(texture_array.shape) == 3:
        gray = np.mean(texture_array, axis=2)
    else:
        gray = texture_array
    
    return np.std(gray)

def analyze_frequency_content(texture_array):
    """Analyze frequency content using FFT."""
    if len(texture_array.shape) == 3:
        gray = np.mean(texture_array, axis=2)
    else:
        gray = texture_array
    
    # Compute 2D FFT
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft)
    
    # Analyze frequency distribution
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Calculate energy in different frequency bands
    low_freq = np.sum(fft_magnitude[center_y-h//8:center_y+h//8, center_x-w//8:center_x+w//8])
    total_energy = np.sum(fft_magnitude)
    
    low_freq_ratio = low_freq / total_energy if total_energy > 0 else 0
    
    return {
        'low_frequency_ratio': low_freq_ratio,
        'total_energy': total_energy
    }
```

### Spatially-Varying Poison Maps

```python
def generate_spatially_varying_poison(shape, user_id, base_strength=0.2):
    """Generate poison map with spatial variation."""
    
    height, width = shape
    
    # Create base random pattern
    np.random.seed(user_id)
    base_noise = np.random.uniform(0, 1, shape)
    
    # Create spatial variation pattern
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, height),
        np.linspace(0, 1, width),
        indexing='ij'
    )
    
    # Example: Radial variation from center
    center_y, center_x = height // 2, width // 2
    distance_from_center = np.sqrt(
        ((y_coords - 0.5) * height)**2 + 
        ((x_coords - 0.5) * width)**2
    )
    max_distance = np.sqrt((height/2)**2 + (width/2)**2)
    normalized_distance = distance_from_center / max_distance
    
    # Vary strength based on distance (stronger at edges)
    spatial_multiplier = 0.5 + 0.5 * normalized_distance
    
    # Apply spatial variation
    spatially_varying_poison = base_noise * base_strength * spatial_multiplier
    
    return spatially_varying_poison

# Custom binding with spatially-varying poison
def custom_spatial_binding(albedo_path, normal_path, user_id):
    """Perform binding with spatially-varying poison map."""
    
    # Load textures
    from perceptual_interdependence.utils.texture_processing import TextureProcessor
    processor = TextureProcessor()
    
    albedo_array = processor.load_texture(albedo_path)
    normal_array = processor.load_texture(normal_path)
    
    # Generate custom poison map
    poison_map = generate_spatially_varying_poison(
        albedo_array.shape[:2], 
        user_id, 
        base_strength=0.2
    )
    
    # Apply poison to albedo
    poisoned_albedo = albedo_array * (1.0 + poison_map[:, :, np.newaxis])
    poisoned_albedo = np.clip(poisoned_albedo, 0.0, 1.0)
    
    # Calculate antidote for normal map
    antidote_normal = normal_array.copy()
    antidote_normal[:, :, 2] = normal_array[:, :, 2] / (1.0 + poison_map)
    antidote_normal = np.clip(antidote_normal, 0.0, 1.0)
    
    # Save results
    processor.save_texture(poisoned_albedo, f"custom_spatial_albedo_{user_id}.png")
    processor.save_texture(antidote_normal, f"custom_spatial_normal_{user_id}.png")
    
    return {
        'albedo_path': f"custom_spatial_albedo_{user_id}.png",
        'normal_path': f"custom_spatial_normal_{user_id}.png",
        'poison_map': poison_map
    }
```

## Quality Control and Validation

### Automated Quality Assessment

```python
def validate_binding_quality(original_albedo, original_normal, bound_albedo, bound_normal):
    """Comprehensive quality validation for bound textures."""
    
    validation_results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'metrics': {}
    }
    
    # Check value ranges
    if np.any(bound_albedo < 0) or np.any(bound_albedo > 1):
        validation_results['errors'].append("Bound albedo values out of range [0,1]")
        validation_results['passed'] = False
    
    if np.any(bound_normal < 0) or np.any(bound_normal > 1):
        validation_results['errors'].append("Bound normal values out of range [0,1]")
        validation_results['passed'] = False
    
    # Check for excessive saturation
    saturation_ratio = np.mean(bound_albedo >= 0.99)
    validation_results['metrics']['saturation_ratio'] = saturation_ratio
    
    if saturation_ratio > 0.05:  # More than 5% saturated
        validation_results['warnings'].append(f"High saturation ratio: {saturation_ratio:.1%}")
    
    # Check normal map Z-component validity
    z_component = bound_normal[:, :, 2]
    if np.any(z_component <= 0):
        validation_results['errors'].append("Invalid Z-component in normal map (non-positive values)")
        validation_results['passed'] = False
    
    # Calculate quality metrics
    mse_albedo = np.mean((original_albedo - bound_albedo)**2)
    mse_normal = np.mean((original_normal - bound_normal)**2)
    
    validation_results['metrics']['albedo_mse'] = mse_albedo
    validation_results['metrics']['normal_mse'] = mse_normal
    
    # Estimate PSNR
    if mse_albedo > 0:
        psnr_albedo = 10 * np.log10(1.0 / mse_albedo)
        validation_results['metrics']['albedo_psnr'] = psnr_albedo
        
        if psnr_albedo < 20:
            validation_results['warnings'].append(f"Low albedo PSNR: {psnr_albedo:.1f}dB")
    
    return validation_results

# Usage in binding workflow
def validated_binding(albedo_path, normal_path, user_id, poison_strength=0.2):
    """Perform binding with automatic quality validation."""
    
    # Load original textures
    processor = TextureProcessor()
    original_albedo = processor.load_texture(albedo_path)
    original_normal = processor.load_texture(normal_path)
    
    # Perform binding
    binder = AssetBinder()
    result = binder.bind_textures(
        albedo_path=albedo_path,
        normal_path=normal_path,
        user_id=user_id,
        poison_strength=poison_strength
    )
    
    # Load bound textures for validation
    bound_albedo = processor.load_texture(result['output_paths']['albedo'])
    bound_normal = processor.load_texture(result['output_paths']['normal'])
    
    # Validate quality
    validation = validate_binding_quality(
        original_albedo, original_normal,
        bound_albedo, bound_normal
    )
    
    # Add validation results to output
    result['validation'] = validation
    
    # Print validation summary
    if validation['passed']:
        print(f"✓ Binding validation PASSED for user {user_id}")
    else:
        print(f"✗ Binding validation FAILED for user {user_id}")
        for error in validation['errors']:
            print(f"  Error: {error}")
    
    for warning in validation['warnings']:
        print(f"  Warning: {warning}")
    
    return result
```

## Performance Optimization

### Memory-Efficient Batch Processing

```python
def memory_efficient_batch_binding(texture_pairs, user_ids, max_memory_gb=8):
    """Process large batches with memory constraints."""
    
    import psutil
    import gc
    
    def get_memory_usage():
        return psutil.Process().memory_info().rss / 1024**3  # GB
    
    results = {}
    binder = AssetBinder()
    
    for texture_name, albedo_path, normal_path in texture_pairs:
        print(f"Processing texture: {texture_name}")
        texture_results = {}
        
        for i, user_id in enumerate(user_ids):
            # Check memory usage
            current_memory = get_memory_usage()
            
            if current_memory > max_memory_gb * 0.8:  # 80% threshold
                print(f"Memory usage high ({current_memory:.1f}GB), forcing cleanup...")
                gc.collect()
                current_memory = get_memory_usage()
                
                if current_memory > max_memory_gb * 0.9:  # 90% threshold
                    print("Memory usage critical, pausing processing...")
                    import time
                    time.sleep(5)  # Brief pause for system recovery
            
            # Process user
            result = binder.bind_textures(
                albedo_path=albedo_path,
                normal_path=normal_path,
                user_id=user_id,
                poison_strength=0.2,
                output_prefix=f"{texture_name}_user_{user_id:03d}"
            )
            
            texture_results[user_id] = result
            
            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
        
        results[texture_name] = texture_results
        
        # Major cleanup after each texture
        gc.collect()
    
    return results
```

This advanced binding tutorial provides the foundation for sophisticated usage scenarios, custom algorithms, and production-scale deployments of the Perceptual Interdependence system.