# Performance Optimization Tutorial

This tutorial covers performance optimization techniques for the Perceptual Interdependence system, from basic optimizations to large-scale deployment strategies.

## System Performance Overview

### Baseline Performance Metrics

**Standard Performance (2048×2048 textures)**:
- **Binding Operation**: ~0.1 seconds with Numba JIT
- **Chart Generation**: ~45 seconds (full pipeline)
- **Zoomed Charts**: ~35 seconds (optimized cropping)
- **Forensic Analysis**: ~30-60 seconds (100 users)

**Memory Usage**:
- **Peak Memory**: ~2.5× input texture size
- **2048×2048 RGB**: ~100MB peak usage
- **4096×4096 RGB**: ~400MB peak usage

## CPU Optimization

### Numba JIT Acceleration

The system automatically uses Numba JIT compilation when available:

```python
# Check Numba availability
from perceptual_interdependence.algorithms.cpu_math import get_cpu_math

cpu_math = get_cpu_math()
if cpu_math.numba_available:
    print("✓ Numba JIT acceleration enabled")
    print(f"Expected speedup: 10-50x for mathematical operations")
else:
    print("⚠ Using NumPy fallback (slower)")
    print("Install Numba for significant performance improvement:")
    print("pip install numba")
```

### Benchmarking Your System

```bash
# Run comprehensive performance benchmark
perceptual-interdependence benchmark --size 2048 2048 --iterations 10

# Test different image sizes
for size in 512 1024 2048 4096; do
    echo "Testing ${size}x${size}..."
    perceptual-interdependence benchmark --size $size $size --iterations 5
done
```

**Expected Results**:
```
Image Size    | Processing Time | Throughput
512×512       | ~0.008s        | 33 Mpx/s
1024×1024     | ~0.032s        | 33 Mpx/s  
2048×2048     | ~0.096s        | 44 Mpx/s
4096×4096     | ~0.384s        | 44 Mpx/s
```

### CPU-Specific Optimizations

```python
import numpy as np
from perceptual_interdependence.algorithms.cpu_math import CPUOptimizedMath

# Configure NumPy for optimal performance
def optimize_numpy_performance():
    """Configure NumPy for maximum performance."""
    
    # Use all available CPU cores
    import os
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    # Enable fast math operations
    np.seterr(all='ignore')  # Ignore floating point warnings for speed
    
    print(f"Configured NumPy to use {os.cpu_count()} CPU cores")

# Custom CPU math with optimizations
class OptimizedCPUMath(CPUOptimizedMath):
    """CPU math with additional optimizations."""
    
    def __init__(self):
        super().__init__()
        optimize_numpy_performance()
    
    def generate_poison_map_vectorized(self, shape, seed, poison_strength):
        """Highly optimized poison map generation."""
        
        # Use NumPy's fastest random number generator
        rng = np.random.default_rng(seed)
        
        # Generate in chunks for better cache performance
        chunk_size = 1024 * 1024  # 1M elements per chunk
        total_elements = shape[0] * shape[1]
        
        if total_elements <= chunk_size:
            # Small image: generate directly
            return rng.uniform(0, poison_strength, shape, dtype=np.float32)
        else:
            # Large image: generate in chunks
            poison_map = np.empty(shape, dtype=np.float32)
            flat_view = poison_map.ravel()
            
            for start in range(0, total_elements, chunk_size):
                end = min(start + chunk_size, total_elements)
                chunk_shape = (end - start,)
                flat_view[start:end] = rng.uniform(0, poison_strength, chunk_shape)
            
            return poison_map

# Usage
optimized_cpu_math = OptimizedCPUMath()
```

## Memory Optimization

### Memory-Efficient Processing

```python
def memory_efficient_binding(albedo_path, normal_path, user_id, max_memory_mb=4000):
    """Bind textures with memory constraints."""
    
    import psutil
    from perceptual_interdependence import AssetBinder
    
    def get_memory_usage_mb():
        return psutil.Process().memory_info().rss / 1024**2
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / 1024**2
    print(f"Available memory: {available_memory:.0f}MB")
    
    if available_memory < max_memory_mb:
        print(f"Warning: Limited memory available ({available_memory:.0f}MB < {max_memory_mb}MB)")
    
    # Load texture metadata first
    from PIL import Image
    with Image.open(albedo_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    
    # Estimate memory requirements
    estimated_memory = (width * height * channels * 4 * 6) / 1024**2  # 6 arrays, 4 bytes each
    print(f"Estimated memory usage: {estimated_memory:.0f}MB")
    
    if estimated_memory > max_memory_mb:
        # Process in tiles for large images
        return process_in_tiles(albedo_path, normal_path, user_id, max_memory_mb)
    else:
        # Standard processing
        binder = AssetBinder()
        return binder.bind_textures(
            albedo_path=albedo_path,
            normal_path=normal_path,
            user_id=user_id,
            poison_strength=0.2
        )

def process_in_tiles(albedo_path, normal_path, user_id, max_memory_mb, tile_size=1024):
    """Process large textures in tiles to manage memory usage."""
    
    from PIL import Image
    import numpy as np
    
    # Load image metadata
    with Image.open(albedo_path) as img:
        width, height = img.size
    
    print(f"Processing {width}×{height} texture in {tile_size}×{tile_size} tiles")
    
    # Calculate number of tiles
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    
    print(f"Total tiles: {tiles_x}×{tiles_y} = {tiles_x * tiles_y}")
    
    # Process each tile
    binder = AssetBinder()
    tile_results = []
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Calculate tile bounds
            x_start = tx * tile_size
            y_start = ty * tile_size
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            
            print(f"Processing tile ({tx},{ty}): {x_start},{y_start} to {x_end},{y_end}")
            
            # Extract and process tile
            tile_result = process_tile(
                albedo_path, normal_path, user_id,
                (x_start, y_start, x_end, y_end)
            )
            
            tile_results.append(tile_result)
            
            # Force garbage collection after each tile
            import gc
            gc.collect()
    
    # Combine tile results
    return combine_tile_results(tile_results, width, height)
```

### Streaming Processing for Large Datasets

```python
def streaming_batch_processor(texture_list, user_ids, batch_size=10):
    """Process large datasets with streaming to manage memory."""
    
    import gc
    from collections import deque
    
    # Create processing queue
    processing_queue = deque(texture_list)
    results = {}
    
    while processing_queue:
        # Process batch
        current_batch = []
        for _ in range(min(batch_size, len(processing_queue))):
            if processing_queue:
                current_batch.append(processing_queue.popleft())
        
        print(f"Processing batch of {len(current_batch)} textures...")
        
        # Process current batch
        batch_results = {}
        for texture_name, albedo_path, normal_path in current_batch:
            texture_results = {}
            
            for user_id in user_ids:
                result = memory_efficient_binding(
                    albedo_path, normal_path, user_id, max_memory_mb=2000
                )
                texture_results[user_id] = result
            
            batch_results[texture_name] = texture_results
        
        # Merge results
        results.update(batch_results)
        
        # Aggressive cleanup between batches
        del batch_results
        gc.collect()
        
        print(f"Completed batch. Remaining: {len(processing_queue)} textures")
    
    return results
```

## Parallel Processing

### Multi-Threading for Independent Operations

```python
import concurrent.futures
import threading
from pathlib import Path

def parallel_user_binding(albedo_path, normal_path, user_ids, max_workers=4):
    """Bind textures for multiple users in parallel."""
    
    def bind_single_user(user_id):
        """Bind textures for a single user."""
        from perceptual_interdependence import AssetBinder
        
        # Each thread gets its own AssetBinder instance
        binder = AssetBinder(output_dir=f"./parallel_results/user_{user_id:03d}")
        
        result = binder.bind_textures(
            albedo_path=albedo_path,
            normal_path=normal_path,
            user_id=user_id,
            poison_strength=0.2,
            output_prefix=f"user_{user_id:03d}"
        )
        
        return user_id, result
    
    # Process users in parallel
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_user = {
            executor.submit(bind_single_user, user_id): user_id 
            for user_id in user_ids
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_user):
            user_id = future_to_user[future]
            try:
                result_user_id, result = future.result()
                results[result_user_id] = result
                print(f"✓ Completed user {result_user_id}")
            except Exception as e:
                print(f"✗ Failed user {user_id}: {e}")
    
    return results

# Usage
user_list = list(range(1, 101))  # Users 1-100
parallel_results = parallel_user_binding(
    "texture.png", 
    "normal.png", 
    user_list, 
    max_workers=8
)
```

### Parallel Chart Generation

```python
def parallel_chart_generation(texture_pairs, max_workers=2):
    """Generate charts for multiple textures in parallel."""
    
    def generate_charts_for_texture(texture_info):
        """Generate both standard and zoomed charts for one texture."""
        texture_name, albedo_path, normal_path = texture_info
        
        from perceptual_interdependence.utils.chart_generator import ChartGenerator
        
        generator = ChartGenerator()
        
        # Generate standard chart
        standard_path = generator.generate_demonstration_chart(
            albedo_path=albedo_path,
            normal_path=normal_path,
            victim_id=42,
            attacker_id=99,
            output_path=f"charts/{texture_name}_standard.png"
        )
        
        # Generate zoomed chart
        zoomed_path = generator.generate_zoomed_demonstration_chart(
            albedo_path=albedo_path,
            normal_path=normal_path,
            victim_id=42,
            attacker_id=99,
            output_path=f"charts/{texture_name}_zoomed.png",
            zoom_factor=15.0
        )
        
        return texture_name, standard_path, zoomed_path
    
    # Process textures in parallel (limited workers due to memory usage)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_texture = {
            executor.submit(generate_charts_for_texture, texture): texture[0]
            for texture in texture_pairs
        }
        
        for future in concurrent.futures.as_completed(future_to_texture):
            texture_name = future_to_texture[future]
            try:
                name, standard_path, zoomed_path = future.result()
                results[name] = {
                    'standard': standard_path,
                    'zoomed': zoomed_path
                }
                print(f"✓ Generated charts for {name}")
            except Exception as e:
                print(f"✗ Failed charts for {texture_name}: {e}")
    
    return results
```

## Large-Scale Deployment

### Distributed Processing Architecture

```python
class DistributedProcessor:
    """Distributed processing coordinator for large-scale operations."""
    
    def __init__(self, worker_nodes):
        self.worker_nodes = worker_nodes
        self.task_queue = []
        self.results = {}
    
    def add_task(self, task_type, **kwargs):
        """Add task to processing queue."""
        task = {
            'id': len(self.task_queue),
            'type': task_type,
            'params': kwargs,
            'status': 'pending'
        }
        self.task_queue.append(task)
        return task['id']
    
    def distribute_tasks(self):
        """Distribute tasks across worker nodes."""
        
        # Simple round-robin distribution
        for i, task in enumerate(self.task_queue):
            worker_id = i % len(self.worker_nodes)
            worker = self.worker_nodes[worker_id]
            
            print(f"Assigning task {task['id']} to worker {worker_id}")
            
            # In a real implementation, this would use network communication
            result = self.execute_task_on_worker(worker, task)
            self.results[task['id']] = result
    
    def execute_task_on_worker(self, worker, task):
        """Execute task on specific worker (simplified)."""
        
        if task['type'] == 'bind':
            return self.worker_bind_texture(worker, task['params'])
        elif task['type'] == 'chart':
            return self.worker_generate_chart(worker, task['params'])
        elif task['type'] == 'forensic':
            return self.worker_forensic_analysis(worker, task['params'])
    
    def worker_bind_texture(self, worker, params):
        """Execute binding task on worker."""
        # Simulate distributed execution
        from perceptual_interdependence import AssetBinder
        
        binder = AssetBinder()
        return binder.bind_textures(**params)

# Usage for large-scale processing
def large_scale_processing_example():
    """Example of large-scale distributed processing."""
    
    # Define worker nodes (in practice, these would be remote machines)
    workers = [
        {'id': 0, 'host': 'worker-01', 'cores': 16, 'memory': '64GB'},
        {'id': 1, 'host': 'worker-02', 'cores': 16, 'memory': '64GB'},
        {'id': 2, 'host': 'worker-03', 'cores': 16, 'memory': '64GB'},
        {'id': 3, 'host': 'worker-04', 'cores': 16, 'memory': '64GB'}
    ]
    
    processor = DistributedProcessor(workers)
    
    # Add binding tasks for 1000 users
    for user_id in range(1, 1001):
        processor.add_task(
            'bind',
            albedo_path='dataset/texture.png',
            normal_path='dataset/normal.png',
            user_id=user_id,
            poison_strength=0.2
        )
    
    # Add chart generation tasks
    for i in range(100):
        processor.add_task(
            'chart',
            albedo_path=f'dataset/texture_{i:03d}.png',
            normal_path=f'dataset/normal_{i:03d}.png',
            victim_id=42,
            attacker_id=99
        )
    
    # Execute all tasks
    print(f"Processing {len(processor.task_queue)} tasks across {len(workers)} workers...")
    processor.distribute_tasks()
    
    return processor.results
```

### Performance Monitoring

```python
import time
import psutil
import threading
from collections import defaultdict

class PerformanceMonitor:
    """Monitor system performance during processing."""
    
    def __init__(self, log_interval=5.0):
        self.log_interval = log_interval
        self.monitoring = False
        self.metrics = defaultdict(list)
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_percent'].append((timestamp, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / 1024**3
            self.metrics['memory_percent'].append((timestamp, memory_percent))
            self.metrics['memory_used_gb'].append((timestamp, memory_used_gb))
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.metrics['disk_read_mb'].append((timestamp, disk_io.read_bytes / 1024**2))
                self.metrics['disk_write_mb'].append((timestamp, disk_io.write_bytes / 1024**2))
            
            time.sleep(self.log_interval)
    
    def get_summary(self):
        """Get performance summary."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                metric_values = [v[1] for v in values]
                summary[metric_name] = {
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'mean': sum(metric_values) / len(metric_values),
                    'samples': len(metric_values)
                }
        
        return summary
    
    def save_metrics(self, filename):
        """Save metrics to file for analysis."""
        import json
        
        with open(filename, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        
        print(f"Metrics saved to {filename}")

# Usage with performance monitoring
def monitored_batch_processing(texture_list, user_ids):
    """Run batch processing with performance monitoring."""
    
    monitor = PerformanceMonitor(log_interval=2.0)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run processing
        start_time = time.time()
        results = streaming_batch_processor(texture_list, user_ids, batch_size=5)
        end_time = time.time()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Print summary
        processing_time = end_time - start_time
        print(f"\nProcessing completed in {processing_time:.1f} seconds")
        
        summary = monitor.get_summary()
        print("\nPerformance Summary:")
        for metric, stats in summary.items():
            print(f"  {metric}: min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}")
        
        # Save detailed metrics
        monitor.save_metrics(f"performance_metrics_{int(time.time())}.json")
        
        return results
        
    except Exception as e:
        monitor.stop_monitoring()
        raise e
```

## Optimization Best Practices

### Configuration Tuning

```python
def optimize_system_configuration():
    """Apply optimal system configuration for performance."""
    
    import os
    import numpy as np
    
    # NumPy optimizations
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    # Disable NumPy warnings for performance
    np.seterr(all='ignore')
    
    # PIL optimizations
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Remove PIL size limits
    
    # Matplotlib optimizations
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for better performance
    
    print("System configuration optimized for performance")

# Apply optimizations at startup
optimize_system_configuration()
```

### Performance Testing Framework

```bash
#!/bin/bash
# Performance regression testing script

echo "Running performance regression tests..."

# Test different image sizes
sizes=(512 1024 2048 4096)
baseline_file="performance_baseline.txt"

for size in "${sizes[@]}"; do
    echo "Testing ${size}x${size}..."
    
    # Run benchmark
    result=$(perceptual-interdependence benchmark --size $size $size --iterations 5 2>&1)
    
    # Extract timing
    timing=$(echo "$result" | grep "Mean time:" | awk '{print $3}' | sed 's/s//')
    
    echo "${size}x${size}: ${timing}s" >> "performance_results_$(date +%Y%m%d).txt"
    
    # Compare with baseline if available
    if [ -f "$baseline_file" ]; then
        baseline_timing=$(grep "${size}x${size}" "$baseline_file" | awk '{print $2}' | sed 's/s//')
        
        if [ ! -z "$baseline_timing" ]; then
            # Calculate performance change
            change=$(echo "scale=2; ($timing - $baseline_timing) / $baseline_timing * 100" | bc)
            echo "  Performance change: ${change}%"
            
            # Alert if significant regression (>10% slower)
            if (( $(echo "$change > 10" | bc -l) )); then
                echo "  ⚠️  PERFORMANCE REGRESSION DETECTED!"
            fi
        fi
    fi
done

echo "Performance testing complete"
```

This performance optimization tutorial provides comprehensive strategies for maximizing the efficiency of the Perceptual Interdependence system across different scales of deployment.