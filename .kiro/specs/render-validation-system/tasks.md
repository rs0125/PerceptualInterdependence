# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create RenderSimulator class file with proper imports (numpy, PIL, skimage)
  - Define class structure with method stubs and docstrings
  - Set up mathematical constants (EPSILON, RGB_MAX, FLOAT_MAX)
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement texture loading and validation
  - [x] 2.1 Create texture loading functionality
    - Implement _load_textures_for_rendering method to load albedo and normal textures
    - Convert albedo from [0, 255] to float [0.0, 1.0] range
    - Unpack normal maps from [0, 255] to [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 2.2 Add input validation and error handling
    - Validate file existence and image format compatibility
    - Ensure albedo and normal textures have matching dimensions
    - Handle various image formats with appropriate error messages
    - _Requirements: 1.4, 2.4, 2.5_

- [x] 3. Implement light direction processing
  - [x] 3.1 Create _normalize_light_direction method
    - Accept light direction as list parameter with default [0, 0, 1]
    - Convert to numpy array and normalize to unit length
    - Handle zero-length vectors with appropriate error handling
    - _Requirements: 1.3, 3.5_

- [x] 4. Implement PBR-lite rendering pipeline
  - [x] 4.1 Create core lighting calculations
    - Implement vectorized dot product calculation between Normal and LightDir
    - Clamp shading values to range [0.0, 1.0] handling negative values
    - Ensure proper handling of vectorized operations across all pixels
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [x] 4.2 Implement pixel composition and output generation
    - Multiply Albedo by Shading values with proper broadcasting across RGB channels
    - Ensure final pixel values remain in range [0.0, 1.0]
    - Return output as float32 numpy array preserving original dimensions
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Implement main render method
  - [x] 5.1 Create render method orchestration
    - Orchestrate complete rendering pipeline from texture loading to final output
    - Call texture loading, light normalization, lighting calculation, and composition
    - Handle method parameter validation and coordinate between internal methods
    - _Requirements: 1.1, 1.2, 1.5_

- [x] 6. Implement quality metrics calculation
  - [x] 6.1 Create PSNR calculation method
    - Implement _calculate_psnr method computing Mean Squared Error
    - Apply PSNR formula: 20 * log10(MAX_VAL / sqrt(MSE))
    - Handle edge case of identical images (infinite PSNR)
    - _Requirements: 5.2_
  
  - [x] 6.2 Create SSIM calculation method
    - Implement _calculate_ssim method using skimage.metrics.structural_similarity
    - Handle luminance channel processing or RGB averaging for single score
    - Configure appropriate parameters for SSIM computation
    - _Requirements: 5.3, 5.4_

- [x] 7. Implement evaluate method for quality assessment
  - [x] 7.1 Create evaluate method
    - Load clean reference image and prepare for comparison
    - Call PSNR and SSIM calculation methods with rendered image
    - Return both metrics as tuple of numerical values
    - _Requirements: 5.1, 5.5_

- [x] 8. Implement binding validation experiment logic
  - [x] 8.1 Create run_binding_experiment method
    - Implement legitimate test: render bound_albedo_A with bound_normal_A
    - Implement attack test: render bound_albedo_A with bound_normal_B
    - Calculate quality metrics for both tests against clean reference
    - _Requirements: 6.1, 6.2_
  
  - [x] 8.2 Implement experiment result analysis
    - Calculate delta between legitimate and attack PSNR scores
    - Calculate delta between legitimate and attack SSIM scores
    - Return comprehensive experiment results as structured dictionary
    - _Requirements: 6.3, 6.4, 6.5_

- [ ]* 9. Create comprehensive test suite
  - [ ]* 9.1 Write unit tests for texture loading
    - Test image loading with various formats (PNG, JPEG, TIFF)
    - Verify normal vector unpacking and range validation accuracy
    - Test dimension matching validation between albedo and normal textures
    - _Requirements: 2.1, 2.2, 2.4, 2.5_
  
  - [ ]* 9.2 Write unit tests for rendering pipeline
    - Test light direction normalization with various input vectors
    - Verify dot product calculation accuracy with known inputs
    - Test shading value clamping and pixel composition correctness
    - _Requirements: 3.1, 3.2, 3.5, 4.1, 4.2_
  
  - [ ]* 9.3 Write unit tests for quality metrics
    - Test PSNR calculation accuracy with known reference values
    - Verify SSIM computation using test images with known similarity scores
    - Test edge case handling for identical and completely different images
    - _Requirements: 5.2, 5.3, 5.4_
  
  - [ ]* 9.4 Write integration tests for complete workflow
    - Test end-to-end render-to-metrics pipeline with sample textures
    - Verify binding effectiveness detection with known good/bad texture pairs
    - Test experiment logic with complete legitimate vs attack scenarios
    - _Requirements: All requirements_

- [x] 10. Create example usage and documentation
  - [x]* 10.1 Write example script demonstrating usage
    - Create sample script showing RenderSimulator class usage
    - Include examples of rendering, quality assessment, and binding experiments
    - Document expected outputs and interpretation of quality metrics
    - _Requirements: 1.1, 5.5, 6.5_