# Requirements Document

## Introduction

This system implements a traitor tracing mechanism for digital assets using RGB forensics and ratio test analysis. The system can identify unauthorized users who have stolen or misused texture assets by analyzing noise patterns embedded in the assets and correlating them with known user signatures.

## Glossary

- **RGB_Forensics_System**: The main forensics analysis system that extracts signatures and identifies traitors
- **Ratio_Test**: A mathematical technique that compares suspicious assets to original clean assets to extract noise patterns
- **Noise_Pattern**: The extracted signature from comparing suspicious and original assets using ratio analysis
- **Signature_Map**: A normalized 2D array representing the unique noise pattern for forensic analysis
- **Correlation_Matcher**: The component that matches extracted signatures against known user patterns
- **Research_Orchestrator**: The main experimental pipeline that coordinates asset generation, simulation, and forensic analysis
- **Asset_Binder**: The system component that generates user-specific asset variations with embedded signatures
- **Render_Simulator**: The component that creates rendered images from bound assets for testing scenarios

## Requirements

### Requirement 1

**User Story:** As a digital forensics investigator, I want to extract noise signatures from suspicious texture assets, so that I can identify which user's signature is embedded in stolen content.

#### Acceptance Criteria

1. WHEN provided with suspicious albedo and original clean texture paths, THE RGB_Forensics_System SHALL extract a noise pattern using ratio analysis
2. THE RGB_Forensics_System SHALL add epsilon value 1e-6 to denominators to prevent division by zero errors
3. WHEN original texture pixels are black (value 0), THE RGB_Forensics_System SHALL mask out those pixels or set ratio to 1.0
4. THE RGB_Forensics_System SHALL fuse RGB channels by averaging to create a single 2D noise map
5. THE RGB_Forensics_System SHALL normalize the signature using (Signature - Mean) / StdDev formula

### Requirement 2

**User Story:** As a digital forensics investigator, I want to correlate extracted signatures with known user patterns, so that I can identify the specific user who created the suspicious asset.

#### Acceptance Criteria

1. WHEN provided with an extracted signature and maximum user count, THE Correlation_Matcher SHALL iterate through all possible user IDs from 0 to max_users
2. THE Correlation_Matcher SHALL regenerate expected noise block maps for each user ID using Asset_Binder logic
3. THE Correlation_Matcher SHALL normalize each expected map before correlation analysis
4. THE Correlation_Matcher SHALL compute correlation scores using sum(Signature * Expected) formula
5. THE Correlation_Matcher SHALL return the user ID with the highest correlation score

### Requirement 3

**User Story:** As a digital forensics investigator, I want to visualize correlation analysis results, so that I can present evidence in a clear and understandable format.

#### Acceptance Criteria

1. THE RGB_Forensics_System SHALL generate matplotlib bar charts showing correlation scores for all tested user IDs
2. THE RGB_Forensics_System SHALL save visualization results as forensic_report.png file
3. THE RGB_Forensics_System SHALL display user IDs on x-axis and correlation scores on y-axis
4. THE RGB_Forensics_System SHALL highlight the detected traitor user ID in the visualization
5. THE RGB_Forensics_System SHALL include chart title and axis labels for clarity

### Requirement 4

**User Story:** As a researcher, I want to run complete experimental pipelines, so that I can validate the entire traitor tracing system with controlled test scenarios.

#### Acceptance Criteria

1. THE Research_Orchestrator SHALL create dummy original albedo using skimage.data.astronaut() function
2. THE Research_Orchestrator SHALL generate dummy original normal maps using flat blue [128, 128, 255] values or gaussian filtered variations
3. THE Research_Orchestrator SHALL generate bound assets for victim user (ID 42) and attacker user (ID 99)
4. THE Research_Orchestrator SHALL create legitimate rendering using victim user's complete asset pair
5. THE Research_Orchestrator SHALL create attack scenario rendering using victim's albedo with attacker's normal map

### Requirement 5

**User Story:** As a researcher, I want to validate forensic detection accuracy, so that I can verify the system correctly identifies the victim user from stolen assets.

#### Acceptance Criteria

1. WHEN analyzing victim user's bound albedo asset, THE RGB_Forensics_System SHALL correctly detect the victim's user ID (42)
2. THE Research_Orchestrator SHALL assert that forensic analysis returns the expected victim user ID
3. THE Research_Orchestrator SHALL generate PSNR and SSIM quality metrics for both legitimate and attack scenarios
4. THE Research_Orchestrator SHALL output formatted LaTeX-compatible table with scenario names and quality metrics
5. THE Research_Orchestrator SHALL save all intermediate results including bound assets and rendered images

### Requirement 6

**User Story:** As a researcher, I want to generate comprehensive experimental reports, so that I can document system performance and present results in academic publications.

#### Acceptance Criteria

1. THE Research_Orchestrator SHALL save legitimate scenario rendering as fig_legit.png
2. THE Research_Orchestrator SHALL save attack scenario rendering as fig_attack.png
3. THE Research_Orchestrator SHALL calculate and display PSNR values in decibels for quality assessment
4. THE Research_Orchestrator SHALL calculate and display SSIM values for structural similarity assessment
5. THE Research_Orchestrator SHALL format output table with columns for Scenario, PSNR (dB), and SSIM values