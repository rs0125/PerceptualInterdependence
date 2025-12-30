"""
RGB Forensics Traitor Tracing System

This module implements the RGBForensics class for extracting noise signatures from
suspicious texture assets and identifying traitors through correlation matching.
The system uses ratio test analysis to extract unique noise patterns and correlates
them with known user signatures to detect unauthorized asset usage.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class RGBForensics:
    """
    RGB Forensics system for traitor tracing in digital texture assets.
    
    This class implements signature extraction using ratio test analysis and
    correlation matching to identify users who have stolen or misused texture assets.
    The system extracts noise patterns from suspicious assets and matches them
    against known user signatures to detect traitors.
    """
    
    # Mathematical constants
    EPSILON = 1e-6      # Division by zero protection
    RGB_MAX = 255.0     # Maximum RGB value for conversion
    
    def __init__(self):
        """
        Initialize the RGBForensics class.
        
        The class is designed to be instantiated without parameters and configured
        through method calls for signature extraction and traitor identification.
        """
        pass
    
    def extract_signature(self, suspicious_albedo_path: str, original_clean_path: str) -> np.ndarray:
        """
        Extract noise signature from suspicious albedo using ratio test analysis.
        
        Implements the complete signature extraction pipeline:
        1. Load suspicious and original albedo textures
        2. Apply ratio test with epsilon protection
        3. Handle black pixels by masking or setting ratio to 1.0
        4. Fuse RGB channels by averaging to create 2D signature
        5. Normalize signature using (Signature - Mean) / StdDev formula
        
        Args:
            suspicious_albedo_path (str): Path to suspicious albedo texture file
            original_clean_path (str): Path to original clean texture file
            
        Returns:
            np.ndarray: Normalized 2D signature array representing noise pattern
            
        Raises:
            FileNotFoundError: If input texture files do not exist
            ValueError: If textures have mismatched dimensions or invalid format
            TypeError: If paths are not strings
        """
        # Validate input types
        if not isinstance(suspicious_albedo_path, str):
            raise TypeError(f"suspicious_albedo_path must be a string, got {type(suspicious_albedo_path)}")
        if not isinstance(original_clean_path, str):
            raise TypeError(f"original_clean_path must be a string, got {type(original_clean_path)}")
        
        # Step 1: Load and validate input textures
        suspicious_array, original_array = self._load_and_validate_textures(
            suspicious_albedo_path, original_clean_path
        )
        
        # Step 2: Apply ratio test with epsilon protection
        ratio_array = self._apply_ratio_test(suspicious_array, original_array)
        
        # Step 3: Handle black pixels by masking or setting ratio to 1.0
        ratio_processed = self._handle_black_pixels(ratio_array, original_array)
        
        # Step 4: Fuse RGB channels by averaging to create 2D signature
        signature_2d = self._fuse_rgb_channels(ratio_processed)
        
        # Step 5: Normalize signature using (Signature - Mean) / StdDev formula
        normalized_signature = self._normalize_signature(signature_2d)
        
        return normalized_signature
    
    def _load_and_validate_textures(self, suspicious_path: str, original_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate input textures for signature extraction.
        
        Args:
            suspicious_path (str): Path to suspicious albedo texture
            original_path (str): Path to original clean texture
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Loaded suspicious and original texture arrays
            
        Raises:
            FileNotFoundError: If texture files do not exist
            ValueError: If textures have mismatched dimensions or invalid format
        """
        # Validate file existence
        import os
        if not os.path.exists(suspicious_path):
            raise FileNotFoundError(f"Suspicious albedo texture file not found: {suspicious_path}")
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original clean texture file not found: {original_path}")
        
        try:
            # Load suspicious albedo texture
            suspicious_img = Image.open(suspicious_path).convert('RGB')
            suspicious_array = np.array(suspicious_img, dtype=np.float32)
            
            # Convert from [0, 255] to float [0.0, 1.0] range if needed
            if suspicious_array.max() > 1.0:
                suspicious_array = suspicious_array / self.RGB_MAX
            
            # Load original clean texture
            original_img = Image.open(original_path).convert('RGB')
            original_array = np.array(original_img, dtype=np.float32)
            
            # Convert from [0, 255] to float [0.0, 1.0] range if needed
            if original_array.max() > 1.0:
                original_array = original_array / self.RGB_MAX
            
            # Ensure textures have matching dimensions
            if suspicious_array.shape != original_array.shape:
                raise ValueError(f"Texture dimension mismatch: suspicious {suspicious_array.shape} vs "
                               f"original {original_array.shape}. Both textures must have identical dimensions.")
            
            # Ensure textures are 3-channel (RGB)
            if len(suspicious_array.shape) != 3 or suspicious_array.shape[2] != 3:
                raise ValueError(f"Suspicious texture must be RGB (3 channels), got shape: {suspicious_array.shape}")
            if len(original_array.shape) != 3 or original_array.shape[2] != 3:
                raise ValueError(f"Original texture must be RGB (3 channels), got shape: {original_array.shape}")
            
            # Validate textures are not empty
            if suspicious_array.size == 0:
                raise ValueError(f"Suspicious texture is empty: {suspicious_path}")
            if original_array.size == 0:
                raise ValueError(f"Original texture is empty: {original_path}")
            
            return suspicious_array, original_array
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Failed to load or process texture files: {str(e)}")
    
    def _apply_ratio_test(self, suspicious: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Apply ratio test analysis with epsilon protection.
        
        Implements the ratio test formula: Noise_Pattern = Suspicious / (Original + epsilon)
        
        Args:
            suspicious (np.ndarray): Suspicious albedo texture array [0.0, 1.0]
            original (np.ndarray): Original clean texture array [0.0, 1.0]
            
        Returns:
            np.ndarray: Ratio array representing noise pattern
        """
        # Add epsilon protection to prevent division by zero
        original_protected = original + self.EPSILON
        
        # Apply ratio test: Noise_Pattern = Suspicious / (Original + epsilon)
        ratio_array = suspicious / original_protected
        
        return ratio_array.astype(np.float32)
    
    def _handle_black_pixels(self, ratio_array: np.ndarray, original_array: np.ndarray) -> np.ndarray:
        """
        Handle edge cases for black pixels by masking or setting ratio to 1.0.
        
        Args:
            ratio_array (np.ndarray): Ratio array from ratio test
            original_array (np.ndarray): Original texture array for black pixel detection
            
        Returns:
            np.ndarray: Processed ratio array with black pixels handled
        """
        # Create a copy to avoid modifying the input
        ratio_processed = ratio_array.copy()
        
        # Detect black pixels (values close to 0.0)
        black_pixel_mask = original_array < self.EPSILON
        
        # Set ratio to 1.0 for black pixels to avoid noise artifacts
        ratio_processed[black_pixel_mask] = 1.0
        
        return ratio_processed
    
    def _fuse_rgb_channels(self, ratio_array: np.ndarray) -> np.ndarray:
        """
        Fuse RGB channels by averaging to convert 3D noise to 2D signature.
        
        Args:
            ratio_array (np.ndarray): 3D ratio array with shape (height, width, 3)
            
        Returns:
            np.ndarray: 2D signature array with shape (height, width)
        """
        # Average across RGB channels (axis=2) to create 2D signature
        signature_2d = np.mean(ratio_array, axis=2)
        
        return signature_2d.astype(np.float32)
    
    def _normalize_signature(self, signature: np.ndarray) -> np.ndarray:
        """
        Normalize signature using (Signature - Mean) / StdDev formula.
        
        Args:
            signature (np.ndarray): 2D signature array to normalize
            
        Returns:
            np.ndarray: Normalized signature with zero mean and unit variance
        """
        # Calculate mean and standard deviation
        signature_mean = np.mean(signature)
        signature_std = np.std(signature)
        
        # Handle edge case where standard deviation is zero (constant signature)
        if signature_std < self.EPSILON:
            # Return zero array if signature is constant
            return np.zeros_like(signature, dtype=np.float32)
        
        # Apply normalization: (Signature - Mean) / StdDev
        normalized_signature = (signature - signature_mean) / signature_std
        
        return normalized_signature.astype(np.float32)
    
    def find_traitor(self, signature: np.ndarray, max_users: int = 100) -> int:
        """
        Find traitor by correlating extracted signature with known user patterns.
        
        Iterates through user IDs from 0 to max_users, regenerates expected noise
        block maps for each user using AssetBinder, normalizes each expected
        map, computes correlation scores, and returns the user ID with highest score.
        
        Args:
            signature (np.ndarray): Extracted signature array to match
            max_users (int): Maximum number of users to test (default: 100)
            
        Returns:
            int: User ID with highest correlation score (detected traitor)
            
        Raises:
            ValueError: If signature is invalid or max_users is not positive
            TypeError: If parameters have incorrect types
        """
        # Validate input types
        if not isinstance(signature, np.ndarray):
            raise TypeError(f"signature must be a numpy array, got {type(signature)}")
        if not isinstance(max_users, int):
            raise TypeError(f"max_users must be an integer, got {type(max_users)}")
        
        # Validate parameters
        if max_users <= 0:
            raise ValueError(f"max_users must be positive, got {max_users}")
        if signature.size == 0:
            raise ValueError("Signature array cannot be empty")
        if len(signature.shape) != 2:
            raise ValueError(f"Signature must be 2D array, got shape: {signature.shape}")
        
        # Initialize correlation tracking
        correlation_scores = np.zeros(max_users, dtype=np.float32)
        best_user_id = 0
        highest_correlation = -np.inf
        
        # Initialize AssetBinder for noise generation
        from .asset_binder import AssetBinder
        asset_binder = AssetBinder()
        
        # Iterate through all possible user IDs from 0 to max_users
        for user_id in range(max_users):
            try:
                # Generate expected noise block map for this user ID
                expected_noise = self._generate_expected_noise_map(
                    signature.shape, user_id, asset_binder
                )
                
                # Normalize expected map using same algorithm as signature normalization
                normalized_expected = self._normalize_signature(expected_noise)
                
                # Compute correlation score using sum(Signature * Expected) formula
                correlation_score = self._compute_correlation_score(signature, normalized_expected)
                
                # Store correlation score
                correlation_scores[user_id] = correlation_score
                
                # Track user ID with highest correlation
                if correlation_score > highest_correlation:
                    highest_correlation = correlation_score
                    best_user_id = user_id
                    
            except Exception as e:
                # Handle individual user processing errors gracefully
                print(f"Warning: Error processing user {user_id}: {e}")
                correlation_scores[user_id] = -np.inf
        
        # Store correlation scores for visualization
        self._last_correlation_scores = correlation_scores
        self._last_detected_user = best_user_id
        
        return best_user_id
    
    def _generate_expected_noise_map(self, shape: Tuple[int, int], user_seed: int, 
                                   asset_binder) -> np.ndarray:
        """
        Generate expected noise block map for a specific user using AssetBinder logic.
        
        Args:
            shape (Tuple[int, int]): Shape of the noise map (height, width)
            user_seed (int): User seed for noise generation
            asset_binder: AssetBinder instance for noise generation
            
        Returns:
            np.ndarray: Expected noise map for the user
        """
        # Use AssetBinder's poison map generation method
        # Default poison_strength of 0.2 matches typical binding parameters
        expected_noise = asset_binder._generate_poison_map(shape, user_seed, poison_strength=0.2)
        
        return expected_noise
    
    def _compute_correlation_score(self, signature: np.ndarray, expected: np.ndarray) -> float:
        """
        Computes the Theoretical Z-Score using Pearson Correlation scaled by sqrt(N).
        This is the standard 'Watermark Detection Statistic'.
        
        Args:
            signature (np.ndarray): Normalized signature array
            expected (np.ndarray): Normalized expected noise array
            
        Returns:
            float: Theoretical Z-Score (rho * sqrt(N))
        """
        # Ensure arrays have matching shapes
        if signature.shape != expected.shape:
            raise ValueError(f"Shape mismatch: signature {signature.shape} vs expected {expected.shape}")
        
        # 1. Flatten arrays to 1D
        sig_flat = signature.flatten()
        exp_flat = expected.flatten()
        
        # 2. Centering (Subtract Mean)
        # We do this explicitly to ensure robustness against lighting shifts
        sig_mean = np.mean(sig_flat)
        exp_mean = np.mean(exp_flat)
        
        sig_centered = sig_flat - sig_mean
        exp_centered = exp_flat - exp_mean
        
        # 3. Pearson Correlation Coefficient (rho)
        # Formula: sum(a*b) / sqrt(sum(a^2) * sum(b^2))
        numerator = np.sum(sig_centered * exp_centered)
        denominator = np.sqrt(np.sum(sig_centered**2) * np.sum(exp_centered**2))
        
        if denominator == 0:
            return 0.0
        
        rho = numerator / denominator
        
        # 4. Theoretical Z-Score Calculation
        # Z = rho * sqrt(N)
        # For 1k texture, N = 1,048,576. sqrt(N) = 1024.
        # A tiny correlation of 0.01 becomes Z = 10.24.
        n_pixels = len(sig_flat)
        z_score = rho * np.sqrt(n_pixels)
        
        return float(z_score)
    
    def visualize_results(self, scores: Optional[np.ndarray] = None, 
                         detected_uid: Optional[int] = None) -> None:
        """
        Generate matplotlib bar chart visualization of correlation analysis results.
        
        Creates a bar chart showing correlation scores for all tested user IDs,
        highlights the detected traitor user ID, and saves the result as forensic_report.png.
        
        Args:
            scores (np.ndarray, optional): Array of correlation scores. Uses last computed if None.
            detected_uid (int, optional): Detected user ID to highlight. Uses last detected if None.
            
        Raises:
            ValueError: If no correlation data is available and none provided
        """
        # Use last computed results if not provided
        if scores is None:
            if not hasattr(self, '_last_correlation_scores'):
                raise ValueError("No correlation scores available. Run find_traitor() first or provide scores.")
            scores = self._last_correlation_scores
        
        if detected_uid is None:
            if not hasattr(self, '_last_detected_user'):
                raise ValueError("No detected user available. Run find_traitor() first or provide detected_uid.")
            detected_uid = self._last_detected_user
        
        # Create matplotlib figure and axis
        plt.figure(figsize=(12, 6))
        
        # Generate user IDs for x-axis
        user_ids = np.arange(len(scores))
        
        # Create bar chart with user IDs on x-axis and correlation scores on y-axis
        bars = plt.bar(user_ids, scores, alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
        
        # Highlight the detected traitor user ID
        if 0 <= detected_uid < len(bars):
            bars[detected_uid].set_color('red')
            bars[detected_uid].set_alpha(1.0)
            bars[detected_uid].set_linewidth(2.0)
        
        # Add chart title and axis labels for clarity
        plt.title('RGB Forensics Correlation Analysis Results', fontsize=16, fontweight='bold')
        plt.xlabel('User ID', fontsize=12)
        plt.ylabel('Correlation Score', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        if 0 <= detected_uid < len(bars):
            plt.legend(['Other Users', f'Detected Traitor (ID: {detected_uid})'], 
                      loc='upper right', fontsize=10)
        
        # Add annotation for detected traitor
        if 0 <= detected_uid < len(scores):
            max_score = scores[detected_uid]
            plt.annotate(f'Traitor ID: {detected_uid}\nScore: {max_score:.4f}',
                        xy=(detected_uid, max_score),
                        xytext=(detected_uid + len(scores) * 0.1, max_score + (max(scores) - min(scores)) * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Adjust layout and save as forensic_report.png with high resolution
        plt.tight_layout()
        plt.savefig('forensic_report.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Forensic visualization saved as 'forensic_report.png'")
        print(f"Detected traitor: User ID {detected_uid} with correlation score {scores[detected_uid]:.6f}")
    
    def generate_continuous_spike_chart(self, signature: np.ndarray, max_users: int = 100, 
                                      output_path: str = 'forensic_spike_chart.png') -> int:
        """
        Generate continuous spike chart for forensic detection demonstration.
        
        Creates a professional spike chart showing correlation scores across all users,
        with clear visualization of the detection spike for the traitor user.
        
        Args:
            signature (np.ndarray): Extracted signature array to analyze
            max_users (int): Maximum number of users to test (default: 100)
            output_path (str): Output path for the spike chart
            
        Returns:
            int: Detected traitor user ID
        """
        # Run traitor detection to get correlation scores
        detected_user = self.find_traitor(signature, max_users)
        scores = self._last_correlation_scores
        
        # Create professional spike chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Generate user IDs for x-axis
        user_ids = np.arange(len(scores))
        
        # Create continuous line plot with markers
        ax.plot(user_ids, scores, 'b-', linewidth=1.5, alpha=0.7, label='Correlation Scores')
        ax.scatter(user_ids, scores, c='lightblue', s=20, alpha=0.6, edgecolors='navy', linewidth=0.5)
        
        # Highlight the detection spike
        if 0 <= detected_user < len(scores):
            ax.scatter(detected_user, scores[detected_user], c='red', s=200, 
                      marker='^', edgecolors='darkred', linewidth=2, 
                      label=f'Detected Traitor (ID: {detected_user})', zorder=5)
            
            # Add vertical line at detection point
            ax.axvline(x=detected_user, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add horizontal line at detection score
            ax.axhline(y=scores[detected_user], color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # Styling for professional appearance
        ax.set_title('Forensic Detection: Continuous Correlation Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('User ID', fontsize=14, fontweight='bold')
        ax.set_ylabel('Theoretical Z-Score (ρ × √N)', fontsize=14, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set axis limits with padding
        ax.set_xlim(-2, max_users + 2)
        score_range = max(scores) - min(scores)
        ax.set_ylim(min(scores) - score_range * 0.1, max(scores) + score_range * 0.2)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add statistical annotations
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        detection_score = scores[detected_user]
        
        # Add text box with statistics
        stats_text = f'Detection Statistics:\n'
        stats_text += f'Mean Z-Score: {mean_score:.4f}\n'
        stats_text += f'Std Dev: {std_score:.4f}\n'
        stats_text += f'Detection Z-Score: {detection_score:.4f}\n'
        stats_text += f'Relative Strength: {(detection_score - mean_score) / std_score:.2f}σ'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightgray', alpha=0.8))
        
        # Add detection threshold line (mean + 3*std)
        threshold = mean_score + 3 * std_score
        ax.axhline(y=threshold, color='orange', linestyle='-.', alpha=0.7, 
                  linewidth=2, label=f'Detection Threshold (μ+3σ)')
        
        # Update legend to include threshold
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Professional styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Continuous spike chart saved as '{output_path}'")
        print(f"Detected traitor: User ID {detected_user}")
        print(f"Detection score: {detection_score:.6f} (Z-score: {(detection_score - mean_score) / std_score:.2f})")
        
        return detected_user