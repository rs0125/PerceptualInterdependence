# Requirements Document

## Introduction

The Geometry-Aware Photometric Binding system implements a sophisticated texture processing technique that applies controlled "poison" noise to RGB albedo textures while maintaining geometric consistency through corresponding normal map adjustments. This system enables texture artists to create bound texture pairs that survive compression while preserving visual fidelity.

## Glossary

- **AssetBinderComplex**: The main Python class implementing the geometry-aware photometric binding algorithm
- **RenderSimulator**: Python class for validating visual quality of bound assets through PBR-lite rendering and metric calculation
- **Albedo Texture**: RGB texture representing surface color information, stored as float values [0.0, 1.0]
- **Normal Map**: RGB texture encoding surface normal vectors, packed from [-1.0, 1.0] to [0, 255]
- **Brightening Poison**: Controlled noise applied to increase albedo brightness within specified strength parameters
- **Antidote Normal**: Adjusted normal map that compensates for albedo modifications to maintain geometric consistency
- **Block Noise**: Noise generated in 4x4 pixel blocks to survive BC7 compression algorithms
- **Headroom**: Available range for brightness adjustment before clipping occurs
- **Vector Bending**: Process of adjusting normal vector components while preserving azimuth direction
- **PBR-Lite Renderer**: Simplified physically-based rendering implementation for validation purposes
- **PSNR**: Peak Signal-to-Noise Ratio metric for measuring image quality differences
- **SSIM**: Structural Similarity Index metric for perceptual image quality assessment

## Requirements

### Requirement 1

**User Story:** As a texture artist, I want to load and process RGB albedo and normal map textures, so that I can apply geometry-aware photometric binding.

#### Acceptance Criteria

1. WHEN the AssetBinderComplex class is instantiated, THE system SHALL accept clean_albedo_path and original_normal_path as string parameters
2. WHEN loading albedo textures, THE system SHALL convert pixel values from [0, 255] to float range [0.0, 1.0]
3. WHEN loading normal maps, THE system SHALL unpack RGB values from [0, 255] to normal vectors [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
4. WHEN processing normal vectors, THE system SHALL normalize input normal vectors to ensure unit length
5. THE system SHALL validate that input files exist and are readable image formats

### Requirement 2

**User Story:** As a texture artist, I want to configure binding parameters, so that I can control the strength and reproducibility of the binding process.

#### Acceptance Criteria

1. WHEN initializing the binding process, THE system SHALL accept user_seed as integer parameter for reproducible results
2. WHEN configuring poison strength, THE system SHALL accept poison_strength as float parameter with default value 0.2
3. WHEN poison_strength is provided, THE system SHALL validate that the value is within acceptable range [0.0, 1.0]
4. THE system SHALL use the user_seed to initialize numpy random state for consistent noise generation

### Requirement 3

**User Story:** As a texture artist, I want the system to generate block-based noise, so that the binding survives BC7 compression.

#### Acceptance Criteria

1. WHEN generating noise, THE system SHALL implement _generate_block_noise method accepting shape and seed parameters
2. WHEN creating noise blocks, THE system SHALL generate noise in 4x4 pixel blocks
3. WHEN calculating noise values, THE system SHALL generate scalar values in range [1.0, 1.0 + poison_strength]
4. WHEN applying block noise, THE system SHALL ensure noise pattern is optimized for BC7 compression survival
5. THE system SHALL use np.random.RandomState with provided seed for deterministic noise generation

### Requirement 4

**User Story:** As a texture artist, I want the system to apply brightening poison to albedo textures, so that I can create bound texture pairs with controlled modifications.

#### Acceptance Criteria

1. WHEN applying poison to albedo, THE system SHALL multiply Albedo_Old by Noise_S values
2. WHEN calculating new albedo values, THE system SHALL clip results to valid range [0.0, 1.0]
3. WHEN clipping occurs, THE system SHALL calculate effective noise as S_effective = Albedo_New / (Albedo_Old + 1e-6)
4. THE system SHALL handle division by zero cases using epsilon value 1e-6
5. THE system SHALL preserve original albedo characteristics while applying controlled modifications

### Requirement 5

**User Story:** As a texture artist, I want the system to generate antidote normal maps, so that geometric consistency is maintained after albedo modifications.

#### Acceptance Criteria

1. WHEN processing normal maps, THE system SHALL extract Z-component from original normal vectors
2. WHEN calculating target Z values, THE system SHALL compute Z_new = Z_old / S_effective
3. WHEN preserving azimuth, THE system SHALL maintain X/Y ratio of original normal vectors
4. WHEN calculating lateral components, THE system SHALL compute Lat_old = sqrt(x_old^2 + y_old^2)
5. WHEN determining new lateral magnitude, THE system SHALL calculate Lat_new = sqrt(1 - Z_new^2)
6. WHEN scaling vectors, THE system SHALL compute scale factor k = Lat_new / (Lat_old + 1e-6)
7. WHEN generating new components, THE system SHALL calculate X_new = X_old * k and Y_new = Y_old * k

### Requirement 6

**User Story:** As a texture artist, I want the system to output bound texture pairs, so that I can use them in my rendering pipeline.

#### Acceptance Criteria

1. WHEN saving albedo textures, THE system SHALL pack float values [0.0, 1.0] back to RGB format [0, 255]
2. WHEN saving normal maps, THE system SHALL pack normal vectors [-1.0, 1.0] to RGB using formula (Vector + 1.0) / 2.0 * 255.0
3. WHEN generating output filenames, THE system SHALL create bound_albedo_{seed}.png and bound_normal_{seed}.png
4. THE system SHALL save output files in PNG format with appropriate bit depth
5. THE system SHALL ensure output files maintain original image dimensions and quality

### Requirement 7

**User Story:** As a texture artist, I want to validate the visual quality of bound assets through rendering simulation, so that I can verify the binding effectiveness and detect potential attacks.

#### Acceptance Criteria

1. WHEN the RenderSimulator class is instantiated, THE system SHALL provide a render method accepting albedo_path, normal_path, and light_dir parameters
2. WHEN loading textures for rendering, THE system SHALL load albedo images and unpack normal maps to [-1, 1] vectors
3. WHEN calculating lighting, THE system SHALL compute shading using dot product of Normal and LightDir vectors
4. WHEN applying shading, THE system SHALL clamp shading values to range [0.0, 1.0]
5. WHEN composing final pixels, THE system SHALL multiply Albedo by Shading and broadcast across RGB channels
6. THE system SHALL return rendered output as float array with values in range [0.0, 1.0]

### Requirement 8

**User Story:** As a texture artist, I want to calculate quality metrics between rendered images, so that I can quantify the visual impact of binding operations.

#### Acceptance Criteria

1. WHEN the RenderSimulator provides evaluate method, THE system SHALL accept clean_ref_path and rendered_img parameters
2. WHEN calculating PSNR, THE system SHALL compute Peak Signal-to-Noise Ratio between reference and rendered images
3. WHEN calculating SSIM, THE system SHALL use skimage.metrics to compute Structural Similarity
4. WHEN computing SSIM, THE system SHALL calculate for luminance channel or average across RGB to get single score
5. THE system SHALL return both PSNR and SSIM metrics as numerical values

### Requirement 9

**User Story:** As a texture artist, I want to run binding validation experiments, so that I can verify legitimate binding quality and detect attack scenarios.

#### Acceptance Criteria

1. WHEN running legitimate test, THE system SHALL render bound_albedo_A with bound_normal_A and compare against clean reference
2. WHEN running attack test, THE system SHALL render bound_albedo_A with bound_normal_B and compare against clean reference
3. WHEN calculating experiment results, THE system SHALL compute delta between legitimate and attack scores
4. THE system SHALL expect high PSNR for legitimate test and low PSNR for attack test
5. THE system SHALL output the quality metric differences to quantify binding effectiveness