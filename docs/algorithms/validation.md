# Validation Methods

This document describes the comprehensive validation and quality assurance methods used in the Perceptual Interdependence system to ensure mathematical correctness, performance reliability, and system integrity.

## Mathematical Validation Framework

### Perfect Cancellation Verification

The core mathematical property of the system is perfect cancellation. This is validated through rigorous testing:

```python
def validate_perfect_cancellation(test_cases=1000, tolerance=1e-6):
    """Validate perfect mathematical cancellation property."""
    
    validation_results = {
        'passed_tests': 0,
        'failed_tests': 0,
        'max_error': 0.0,
        'mean_error': 0.0,
        'error_distribution': []
    }
    
    errors = []
    
    for i in range(test_cases):
        # Generate random test case
        shape = (64, 64, 3)  # Small size for fast testing
        
        # Random albedo and normal map
        albedo = np.random.uniform(0.1, 0.9, shape).astype(np.float32)
        normal = np.random.uniform(0.1, 0.9, shape).astype(np.float32)
        
        # Random poison parameters
        user_id = np.random.randint(1, 10000)
        poison_strength = np.random.uniform(0.05, 0.3)
        
        # Generate poison map
        poison_map = generate_poison_map(shape[:2], user_id, poison_strength)
        
        # Apply binding operations
        poisoned_albedo = albedo * (1.0 + poison_map[:, :, np.newaxis])
        antidote_normal = normal.copy()
        antidote_normal[:, :, 2] = normal[:, :, 2] / (1.0 + poison_map)
        
        # Calculate products
        original_product = albedo * normal
        bound_product = poisoned_albedo * antidote_normal
        
        # Measure error
        error = np.abs(original_product - bound_product)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        errors.append(max_error)
        
        # Check if within tolerance
        if max_error <= tolerance:
            validation_results['passed_tests'] += 1
        else:
            validation_results['failed_tests'] += 1
    
    # Calculate statistics
    validation_results['max_error'] = np.max(errors)
    validation_results['mean_error'] = np.mean(errors)
    validation_results['error_distribution'] = errors
    validation_results['success_rate'] = validation_results['passed_tests'] / test_cases
    
    return validation_results
```

### Geometric Constraint Validation

Normal maps must maintain valid geometric properties:

```python
def validate_geometric_constraints(normal_maps, tolerance=1e-3):
    """Validate geometric constraints of normal maps."""
    
    validation_results = {
        'valid_normal_maps': 0,
        'invalid_normal_maps': 0,
        'constraint_violations': {
            'negative_z': 0,
            'non_unit_vectors': 0,
            'invalid_range': 0
        }
    }
    
    for normal_map in normal_maps:
        is_valid = True
        
        # Check value range [0, 1]
        if np.any(normal_map < 0) or np.any(normal_map > 1):
            validation_results['constraint_violations']['invalid_range'] += 1
            is_valid = False
        
        # Check Z-component positivity (assuming Z is in channel 2)
        z_component = normal_map[:, :, 2]
        if np.any(z_component <= 0):
            validation_results['constraint_violations']['negative_z'] += 1
            is_valid = False
        
        # Check unit vector constraint (after converting to [-1,1] range)
        normal_tangent = normal_map * 2.0 - 1.0
        vector_lengths = np.sqrt(np.sum(normal_tangent**2, axis=2))
        
        # Allow some tolerance for unit length constraint
        unit_error = np.abs(vector_lengths - 1.0)
        if np.any(unit_error > tolerance):
            validation_results['constraint_violations']['non_unit_vectors'] += 1
            is_valid = False
        
        if is_valid:
            validation_results['valid_normal_maps'] += 1
        else:
            validation_results['invalid_normal_maps'] += 1
    
    return validation_results
```

## Performance Validation

### Regression Testing Framework

```python
class PerformanceRegressionTester:
    """Framework for detecting performance regressions."""
    
    def __init__(self, baseline_file="performance_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline_data = self.load_baseline()
        self.current_results = {}
    
    def load_baseline(self):
        """Load baseline performance data."""
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_baseline(self):
        """Save current results as new baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.current_results, f, indent=2)
    
    def benchmark_operation(self, operation_name, operation_func, *args, iterations=10):
        """Benchmark a specific operation."""
        
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = operation_func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        self.current_results[operation_name] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'iterations': iterations
        }
        
        return mean_time
    
    def check_regression(self, tolerance=0.1):
        """Check for performance regression against baseline."""
        
        regression_results = {
            'regressions_detected': [],
            'improvements_detected': [],
            'new_operations': [],
            'overall_regression': False
        }
        
        for operation_name, current_metrics in self.current_results.items():
            if operation_name in self.baseline_data:
                baseline_time = self.baseline_data[operation_name]['mean_time']
                current_time = current_metrics['mean_time']
                
                # Calculate relative change
                relative_change = (current_time - baseline_time) / baseline_time
                
                if relative_change > tolerance:
                    # Performance regression
                    regression_results['regressions_detected'].append({
                        'operation': operation_name,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'regression_percent': relative_change * 100
                    })
                    regression_results['overall_regression'] = True
                
                elif relative_change < -tolerance:
                    # Performance improvement
                    regression_results['improvements_detected'].append({
                        'operation': operation_name,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'improvement_percent': -relative_change * 100
                    })
            else:
                # New operation
                regression_results['new_operations'].append(operation_name)
        
        return regression_results
```

## System Integration Testing

### End-to-End Workflow Validation

```python
def validate_end_to_end_workflow(texture_paths, user_ids, poison_strengths):
    """Validate complete system workflow."""
    
    validation_results = {
        'successful_workflows': 0,
        'failed_workflows': 0,
        'error_details': [],
        'performance_metrics': {}
    }
    
    for albedo_path, normal_path in texture_paths:
        for user_id in user_ids:
            for poison_strength in poison_strengths:
                
                workflow_start = time.time()
                
                try:
                    # Step 1: Binding
                    binder = AssetBinder()
                    binding_result = binder.bind_textures(
                        albedo_path=albedo_path,
                        normal_path=normal_path,
                        user_id=user_id,
                        poison_strength=poison_strength
                    )
                    
                    # Step 2: Chart Generation
                    generator = ChartGenerator()
                    chart_path = generator.generate_demonstration_chart(
                        albedo_path=albedo_path,
                        normal_path=normal_path,
                        victim_id=user_id,
                        attacker_id=user_id + 1,
                        output_path=f"test_chart_{user_id}_{poison_strength}.png"
                    )
                    
                    # Step 3: Forensic Analysis
                    forensics = RGBForensics()
                    signature = forensics.extract_signature(
                        binding_result['output_paths']['albedo'],
                        albedo_path
                    )
                    detected_user = forensics.find_traitor(signature, max_users=10)
                    
                    # Validate results
                    workflow_valid = True
                    
                    # Check binding results
                    if binding_result['statistics']['saturation_ratio'] > 0.1:
                        workflow_valid = False
                        validation_results['error_details'].append(
                            f"High saturation ratio: {binding_result['statistics']['saturation_ratio']}"
                        )
                    
                    # Check forensic detection
                    if detected_user != user_id:
                        workflow_valid = False
                        validation_results['error_details'].append(
                            f"Forensic detection failed: expected {user_id}, got {detected_user}"
                        )
                    
                    # Check file outputs
                    if not os.path.exists(chart_path):
                        workflow_valid = False
                        validation_results['error_details'].append(f"Chart not generated: {chart_path}")
                    
                    if workflow_valid:
                        validation_results['successful_workflows'] += 1
                    else:
                        validation_results['failed_workflows'] += 1
                    
                    # Record performance
                    workflow_time = time.time() - workflow_start
                    key = f"{os.path.basename(albedo_path)}_{user_id}_{poison_strength}"
                    validation_results['performance_metrics'][key] = workflow_time
                
                except Exception as e:
                    validation_results['failed_workflows'] += 1
                    validation_results['error_details'].append(f"Workflow exception: {str(e)}")
    
    return validation_results
```

This validation framework ensures the system maintains mathematical correctness, performance standards, and operational reliability across all components and use cases.