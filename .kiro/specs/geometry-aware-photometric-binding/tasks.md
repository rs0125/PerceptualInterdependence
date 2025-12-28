# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create AssetBinderComplex class file with proper imports (numpy, PIL)
  - Define class structure with method stubs and docstrings
  - Set up mathematical constants (EPSILON, BLOCK_SIZE, RGB_MAX)
  - _Requirements: 1.1, 2.1_

- [x] 2. Implement input processing and validation
  - [x] 2.1 Create texture loading functionality
    - Implement _load_and_validate_inputs method to load albedo and normal textures
    - Convert albedo from [0, 255] to float [0.0, 1.0] range
    - Unpack normal maps from [0, 255] to [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
    - _Requirements: 1.2, 1.3, 1.5_
  
  - [x] 2.2 Add input validation and error handling
    - Validate file existence and image format compatibility
    - Normalize normal vectors to ensure unit length
    - Validate poison_strength parameter range [0.0, 1.0]
    - _Requirements: 1.4, 1.5, 2.3_

- [x] 3. Implement block-based noise generation
  - [x] 3.1 Create _generate_block_noise method
    - Initialize np.random.RandomState with user_seed for deterministic results
    - Generate noise in 4x4 pixel blocks for BC7 compression survival
    - Produce scalar values in range [1.0, 1.0 + poison_strength]
    - _Requirements: 2.2, 2.4, 3.1, 3.2, 3.3, 3.5_

- [x] 4. Implement albedo poisoning algorithm
  - [x] 4.1 Create _apply_poison_to_albedo method
    - Apply poison by multiplying Albedo_Old by Noise_S values
    - Clip results to valid range [0.0, 1.0] to prevent overflow
    - Calculate effective noise as S_effective = Albedo_New / (Albedo_Old + 1e-6)
    - Handle division by zero cases using epsilon protection
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Implement normal map antidote generation
  - [x] 5.1 Create _generate_antidote_normal method
    - Extract Z-component from original normal vectors
    - Calculate target Z values using Z_new = Z_old / S_effective
    - Preserve azimuth by maintaining X/Y ratio of original vectors
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 5.2 Implement vector bending mathematics
    - Calculate old lateral magnitude: Lat_old = sqrt(x_old^2 + y_old^2)
    - Calculate new lateral magnitude: Lat_new = sqrt(1 - Z_new^2)
    - Compute scale factor: k = Lat_new / (Lat_old + 1e-6)
    - Generate new components: X_new = X_old * k, Y_new = Y_old * k
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

- [x] 6. Implement output processing and file saving
  - [x] 6.1 Create _save_outputs method
    - Pack float albedo values [0.0, 1.0] back to RGB format [0, 255]
    - Pack normal vectors [-1.0, 1.0] to RGB using formula (Vector + 1.0) / 2.0 * 255.0
    - Generate output filenames: bound_albedo_{seed}.png and bound_normal_{seed}.png
    - Save files in PNG format maintaining original dimensions and quality
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 7. Implement main binding orchestration method
  - [x] 7.1 Create bind_textures public method
    - Orchestrate the complete binding workflow from input to output
    - Call input processing, noise generation, poisoning, antidote generation, and output saving
    - Handle method parameter passing and coordinate between internal methods
    - _Requirements: 1.1, 2.1, 2.2_

- [ ]* 8. Create comprehensive test suite
  - [ ]* 8.1 Write unit tests for input processing
    - Test texture loading with various formats (PNG, JPEG, TIFF)
    - Verify normal vector unpacking and normalization accuracy
    - Test parameter validation for seed and poison_strength ranges
    - _Requirements: 1.2, 1.3, 1.4, 2.3_
  
  - [ ]* 8.2 Write unit tests for noise generation
    - Test deterministic output with same seed values
    - Verify block structure validation (4x4 patterns)
    - Test value range verification [1.0, 1.0 + poison_strength]
    - _Requirements: 3.1, 3.3, 3.5_
  
  - [ ]* 8.3 Write unit tests for mathematical algorithms
    - Test albedo poisoning accuracy with known inputs
    - Verify normal vector preservation (unit length maintenance)
    - Test azimuth preservation in vector bending operations
    - _Requirements: 4.1, 4.2, 5.2, 5.3, 5.7_
  
  - [ ]* 8.4 Write integration tests for complete workflow
    - Test end-to-end binding process with sample textures
    - Verify geometric consistency between albedo and normal outputs
    - Test edge cases with extreme poison_strength values
    - _Requirements: All requirements_

- [ ]* 9. Create example usage and documentation
  - [ ]* 9.1 Write example script demonstrating usage
    - Create sample script showing how to use AssetBinderComplex class
    - Include example with sample texture files and various parameter combinations
    - Document expected output and file naming conventions
    - _Requirements: 2.1, 2.2, 6.3_