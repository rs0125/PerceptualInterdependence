# Requirements Document

## Introduction

The Render Validation System implements a PBR-lite renderer and quality assessment framework for validating the visual quality of bound texture assets. This system enables texture artists to verify binding effectiveness through rendering simulation and quantitative quality metrics, including detection of potential attack scenarios.

## Glossary

- **RenderSimulator**: The main Python class implementing PBR-lite rendering and quality assessment
- **PBR-Lite Renderer**: Simplified physically-based rendering implementation for validation purposes
- **Albedo Texture**: RGB texture representing surface color information, stored as float values [0.0, 1.0]
- **Normal Map**: RGB texture encoding surface normal vectors, packed from [-1.0, 1.0] to [0, 255]
- **Light Direction**: 3D vector representing the direction of incoming light for shading calculations
- **Shading Value**: Computed lighting intensity based on normal-light dot product, clamped to [0.0, 1.0]
- **PSNR**: Peak Signal-to-Noise Ratio metric for measuring image quality differences
- **SSIM**: Structural Similarity Index metric for perceptual image quality assessment
- **Legitimate Test**: Validation test using matching bound albedo and normal pairs
- **Attack Test**: Validation test using mismatched bound albedo and normal pairs to detect vulnerabilities

## Requirements

### Requirement 1

**User Story:** As a texture artist, I want to render texture pairs using PBR-lite rendering, so that I can simulate how bound textures will appear under lighting conditions.

#### Acceptance Criteria

1. WHEN the RenderSimulator class is instantiated, THE system SHALL initialize without parameters
2. WHEN the render method is called, THE system SHALL accept albedo_path, normal_path, and optional light_dir parameters
3. WHEN light_dir is not provided, THE system SHALL use default light direction [0, 0, 1]
4. THE system SHALL validate that albedo_path and normal_path point to existing image files
5. THE system SHALL return rendered output as numpy array with float values in range [0.0, 1.0]

### Requirement 2

**User Story:** As a texture artist, I want the system to load and process textures for rendering, so that I can work with albedo and normal map data.

#### Acceptance Criteria

1. WHEN loading albedo textures, THE system SHALL convert pixel values from [0, 255] to float range [0.0, 1.0]
2. WHEN loading normal maps, THE system SHALL unpack RGB values from [0, 255] to normal vectors [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
3. WHEN processing normal vectors, THE system SHALL ensure vectors are properly formatted for dot product calculations
4. THE system SHALL handle image format compatibility and provide appropriate error messages
5. THE system SHALL ensure albedo and normal textures have matching dimensions

### Requirement 3

**User Story:** As a texture artist, I want the system to perform lighting calculations, so that I can simulate realistic shading effects.

#### Acceptance Criteria

1. WHEN calculating lighting, THE system SHALL compute shading using dot product of Normal and LightDir vectors
2. WHEN computing dot products, THE system SHALL handle vectorized operations across all pixels
3. WHEN shading values are calculated, THE system SHALL clamp results to range [0.0, 1.0]
4. THE system SHALL handle negative dot product values by clamping to zero
5. THE system SHALL ensure light direction vector is normalized before calculations

### Requirement 4

**User Story:** As a texture artist, I want the system to compose final rendered pixels, so that I can see the complete lighting effect on textures.

#### Acceptance Criteria

1. WHEN composing final pixels, THE system SHALL multiply Albedo by Shading values
2. WHEN broadcasting shading, THE system SHALL apply shading values across all RGB channels
3. WHEN generating output, THE system SHALL ensure final pixel values remain in range [0.0, 1.0]
4. THE system SHALL preserve original image dimensions in the rendered output
5. THE system SHALL return output as float32 numpy array for precision

### Requirement 5

**User Story:** As a texture artist, I want to calculate quality metrics between images, so that I can quantify visual differences and binding effectiveness.

#### Acceptance Criteria

1. WHEN the evaluate method is called, THE system SHALL accept clean_ref_path and rendered_img parameters
2. WHEN calculating PSNR, THE system SHALL compute Peak Signal-to-Noise Ratio between reference and rendered images
3. WHEN calculating SSIM, THE system SHALL use skimage.metrics.structural_similarity for computation
4. WHEN computing SSIM, THE system SHALL calculate for luminance channel or average across RGB channels to get single score
5. THE system SHALL return both PSNR and SSIM metrics as numerical values in a tuple

### Requirement 6

**User Story:** As a texture artist, I want to run binding validation experiments, so that I can verify legitimate binding quality and detect attack scenarios.

#### Acceptance Criteria

1. WHEN running legitimate test, THE system SHALL render bound_albedo_A with bound_normal_A and compare against clean reference
2. WHEN running attack test, THE system SHALL render bound_albedo_A with bound_normal_B and compare against clean reference
3. WHEN calculating experiment results, THE system SHALL compute delta between legitimate and attack PSNR scores
4. WHEN calculating experiment results, THE system SHALL compute delta between legitimate and attack SSIM scores
5. THE system SHALL expect high PSNR/SSIM for legitimate test and low PSNR/SSIM for attack test to demonstrate binding effectiveness