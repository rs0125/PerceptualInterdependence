"""
Render Validation System - PBR-lite Renderer and Quality Assessment Framework

This module implements a simplified physically-based rendering system for validating
bound texture assets through rendering simulation and quantitative quality metrics.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# Mathematical constants
EPSILON = 1e-6      # Division by zero protection
RGB_MAX = 255.0     # Maximum RGB value for conversion
FLOAT_MAX = 1.0     # Maximum float value for PSNR calculation


class RenderSimulator:
    """
    PBR-lite renderer and quality assessment framework for texture binding validation.
    
    This class provides simplified physically-based rendering to simulate how textures
    appear under lighting conditions, combined with quantitative quality metrics
    (PSNR and SSIM) to measure binding effectiveness and detect attack scenarios.
    """
    
    def __init__(self):
        """
        Initialize the RenderSimulator without parameters.
        
        The simulator is ready to use immediately after instantiation.
        """
        pass
    
    def render(self, albedo_path: str, normal_path: str, light_dir: list = None) -> np.ndarray:
        """
        Main rendering method implementing PBR-lite pipeline.
        
        Orchestrates complete rendering pipeline from texture loading to final output,
        coordinating between texture loading, light normalization, lighting calculation,
        and pixel composition methods.
        
        Args:
            albedo_path (str): Path to albedo texture file
            normal_path (str): Path to normal map texture file
            light_dir (list, optional): Light direction vector [x, y, z]. 
                                      Defaults to [0, 0, 1] if not provided.
        
        Returns:
            np.ndarray: Rendered image as float32 array with values in range [0.0, 1.0]
        
        Raises:
            FileNotFoundError: If texture files don't exist
            ValueError: If textures have mismatched dimensions or invalid parameters
            TypeError: If parameters have incorrect types
        """
        # Validate method parameters
        if not isinstance(albedo_path, str):
            raise TypeError(f"albedo_path must be a string, got {type(albedo_path)}")
        if not isinstance(normal_path, str):
            raise TypeError(f"normal_path must be a string, got {type(normal_path)}")
        if light_dir is not None and not isinstance(light_dir, (list, tuple, np.ndarray)):
            raise TypeError(f"light_dir must be list, tuple, or numpy array, got {type(light_dir)}")
        
        # Step 1: Load and prepare textures for rendering
        albedo_array, normal_array = self._load_textures_for_rendering(albedo_path, normal_path)
        
        # Step 2: Normalize light direction vector (defaults to [0, 0, 1] if None)
        light_direction = self._normalize_light_direction(light_dir)
        
        # Step 3: Calculate lighting using vectorized dot product
        shading_values = self._calculate_lighting(normal_array, light_direction)
        
        # Step 4: Compose final pixels by multiplying albedo by shading
        final_rendered = self._compose_final_pixels(albedo_array, shading_values)
        
        return final_rendered
    
    def evaluate(self, clean_ref_path: str, rendered_img: np.ndarray) -> tuple:
        """
        Calculate PSNR and SSIM quality metrics between reference and rendered images.
        
        Args:
            clean_ref_path (str): Path to clean reference image
            rendered_img (np.ndarray): Rendered image array to compare
        
        Returns:
            tuple: (psnr_value, ssim_value) as numerical values
        
        Raises:
            FileNotFoundError: If reference image doesn't exist
            ValueError: If images have incompatible dimensions or invalid values
            TypeError: If parameters have incorrect types
        """
        # Validate input types
        if not isinstance(clean_ref_path, str):
            raise TypeError(f"clean_ref_path must be a string, got {type(clean_ref_path)}")
        if not isinstance(rendered_img, np.ndarray):
            raise TypeError(f"rendered_img must be a numpy array, got {type(rendered_img)}")
        
        # Validate file existence
        import os
        if not os.path.exists(clean_ref_path):
            raise FileNotFoundError(f"Clean reference image file not found: {clean_ref_path}")
        
        # Load clean reference image and prepare for comparison
        try:
            # Load reference image with format validation
            try:
                ref_img = Image.open(clean_ref_path)
                # Convert to RGB if needed for consistency
                if ref_img.mode not in ['RGB', 'RGBA', 'L']:
                    ref_img = ref_img.convert('RGB')
                elif ref_img.mode == 'RGBA':
                    ref_img = ref_img.convert('RGB')
                elif ref_img.mode == 'L':
                    ref_img = ref_img.convert('RGB')
                    
            except Exception as e:
                raise ValueError(f"Invalid reference image format or corrupted file '{clean_ref_path}': {str(e)}")
            
            # Convert to numpy array and normalize to [0.0, 1.0] range
            ref_array = np.array(ref_img, dtype=np.float32)
            if ref_array.max() > 1.0:  # Check if values are in [0, 255] range
                ref_array = ref_array / RGB_MAX
            
            # Validate rendered image format and range
            if rendered_img.min() < 0.0 or rendered_img.max() > 1.0 + EPSILON:
                raise ValueError(f"Rendered image values must be in range [0.0, 1.0], "
                               f"got range [{rendered_img.min():.6f}, {rendered_img.max():.6f}]")
            
            # Ensure both images have compatible dimensions
            if ref_array.shape != rendered_img.shape:
                raise ValueError(f"Image dimension mismatch: reference {ref_array.shape} vs rendered {rendered_img.shape}. "
                               f"Both images must have identical dimensions for comparison.")
            
            # Ensure images are 3-channel (RGB) or 2-channel (grayscale)
            if len(ref_array.shape) == 3 and ref_array.shape[2] != 3:
                raise ValueError(f"Reference image must be RGB (3 channels), got shape: {ref_array.shape}")
            if len(rendered_img.shape) == 3 and rendered_img.shape[2] != 3:
                raise ValueError(f"Rendered image must be RGB (3 channels), got shape: {rendered_img.shape}")
            
            # Validate images are not empty
            if ref_array.size == 0:
                raise ValueError(f"Reference image is empty: {clean_ref_path}")
            if rendered_img.size == 0:
                raise ValueError(f"Rendered image is empty")
            
            # Call PSNR and SSIM calculation methods with rendered image
            psnr_value = self._calculate_psnr(ref_array, rendered_img)
            ssim_value = self._calculate_ssim(ref_array, rendered_img)
            
            # Return both metrics as tuple of numerical values
            return (psnr_value, ssim_value)
            
        except (FileNotFoundError, ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error during quality evaluation: {str(e)}")
    
    def run_binding_experiment(self, clean_ref_path: str, bound_albedo_a: str, 
                              bound_normal_a: str, bound_normal_b: str, 
                              light_dir: list = None) -> dict:
        """
        Run legitimate vs attack validation experiment.
        
        Performs both legitimate test (bound_albedo_A + bound_normal_A) and
        attack test (bound_albedo_A + bound_normal_B) against clean reference.
        
        Args:
            clean_ref_path (str): Path to clean reference image
            bound_albedo_a (str): Path to bound albedo texture A
            bound_normal_a (str): Path to bound normal texture A (legitimate pair)
            bound_normal_b (str): Path to bound normal texture B (attack pair)
            light_dir (list, optional): Light direction vector. Defaults to [0, 0, 1].
        
        Returns:
            dict: Comprehensive experiment results with PSNR/SSIM deltas
        
        Raises:
            FileNotFoundError: If any texture or reference files don't exist
            ValueError: If textures have incompatible dimensions or invalid parameters
            TypeError: If parameters have incorrect types
        """
        # Validate input types
        if not isinstance(clean_ref_path, str):
            raise TypeError(f"clean_ref_path must be a string, got {type(clean_ref_path)}")
        if not isinstance(bound_albedo_a, str):
            raise TypeError(f"bound_albedo_a must be a string, got {type(bound_albedo_a)}")
        if not isinstance(bound_normal_a, str):
            raise TypeError(f"bound_normal_a must be a string, got {type(bound_normal_a)}")
        if not isinstance(bound_normal_b, str):
            raise TypeError(f"bound_normal_b must be a string, got {type(bound_normal_b)}")
        if light_dir is not None and not isinstance(light_dir, (list, tuple, np.ndarray)):
            raise TypeError(f"light_dir must be list, tuple, or numpy array, got {type(light_dir)}")
        
        # Validate file existence
        import os
        if not os.path.exists(clean_ref_path):
            raise FileNotFoundError(f"Clean reference image file not found: {clean_ref_path}")
        if not os.path.exists(bound_albedo_a):
            raise FileNotFoundError(f"Bound albedo texture A file not found: {bound_albedo_a}")
        if not os.path.exists(bound_normal_a):
            raise FileNotFoundError(f"Bound normal texture A file not found: {bound_normal_a}")
        if not os.path.exists(bound_normal_b):
            raise FileNotFoundError(f"Bound normal texture B file not found: {bound_normal_b}")
        
        try:
            # Implement legitimate test: render bound_albedo_A with bound_normal_A
            legitimate_rendered = self.render(bound_albedo_a, bound_normal_a, light_dir)
            
            # Implement attack test: render bound_albedo_A with bound_normal_B
            attack_rendered = self.render(bound_albedo_a, bound_normal_b, light_dir)
            
            # Calculate quality metrics for both tests against clean reference
            legitimate_psnr, legitimate_ssim = self.evaluate(clean_ref_path, legitimate_rendered)
            attack_psnr, attack_ssim = self.evaluate(clean_ref_path, attack_rendered)
            
            # Calculate delta between legitimate and attack PSNR scores
            psnr_delta = legitimate_psnr - attack_psnr
            
            # Calculate delta between legitimate and attack SSIM scores
            ssim_delta = legitimate_ssim - attack_ssim
            
            # Return comprehensive experiment results as structured dictionary
            experiment_results = {
                'legitimate_psnr': float(legitimate_psnr),
                'legitimate_ssim': float(legitimate_ssim),
                'attack_psnr': float(attack_psnr),
                'attack_ssim': float(attack_ssim),
                'psnr_delta': float(psnr_delta),
                'ssim_delta': float(ssim_delta)
            }
            
            return experiment_results
            
        except (FileNotFoundError, ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error during binding experiment: {str(e)}")
    
    def _load_textures_for_rendering(self, albedo_path: str, normal_path: str) -> tuple:
        """
        Load and prepare textures for rendering.
        
        Args:
            albedo_path (str): Path to albedo texture
            normal_path (str): Path to normal map texture
        
        Returns:
            tuple: (albedo_array, normal_array) prepared for rendering
        
        Raises:
            FileNotFoundError: If texture files don't exist
            ValueError: If textures have mismatched dimensions or invalid format
            TypeError: If paths are not strings
        """
        # Validate input types
        if not isinstance(albedo_path, str):
            raise TypeError(f"albedo_path must be a string, got {type(albedo_path)}")
        if not isinstance(normal_path, str):
            raise TypeError(f"normal_path must be a string, got {type(normal_path)}")
        
        # Validate file existence
        import os
        if not os.path.exists(albedo_path):
            raise FileNotFoundError(f"Albedo texture file not found: {albedo_path}")
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal map texture file not found: {normal_path}")
        
        try:
            # Load albedo texture with format validation
            try:
                albedo_img = Image.open(albedo_path)
                # Verify image can be loaded and converted to RGB
                if albedo_img.mode not in ['RGB', 'RGBA', 'L']:
                    albedo_img = albedo_img.convert('RGB')
                elif albedo_img.mode == 'RGBA':
                    albedo_img = albedo_img.convert('RGB')
                elif albedo_img.mode == 'L':
                    albedo_img = albedo_img.convert('RGB')
                    
            except Exception as e:
                raise ValueError(f"Invalid albedo image format or corrupted file '{albedo_path}': {str(e)}")
            
            albedo_array = np.array(albedo_img, dtype=np.float32)
            
            # Convert albedo from [0, 255] to float [0.0, 1.0] range
            if albedo_array.max() > 1.0:  # Check if values are in [0, 255] range
                albedo_array = albedo_array / RGB_MAX
            
            # Load normal map texture with format validation
            try:
                normal_img = Image.open(normal_path)
                # Verify image can be loaded and converted to RGB
                if normal_img.mode not in ['RGB', 'RGBA', 'L']:
                    normal_img = normal_img.convert('RGB')
                elif normal_img.mode == 'RGBA':
                    normal_img = normal_img.convert('RGB')
                elif normal_img.mode == 'L':
                    normal_img = normal_img.convert('RGB')
                    
            except Exception as e:
                raise ValueError(f"Invalid normal map image format or corrupted file '{normal_path}': {str(e)}")
            
            normal_array = np.array(normal_img, dtype=np.float32)
            
            # Unpack normal maps from [0, 255] to [-1.0, 1.0] using formula (Image / 255.0) * 2.0 - 1.0
            if normal_array.max() > 1.0:  # Check if values are in [0, 255] range
                normal_array = (normal_array / RGB_MAX) * 2.0 - 1.0
            
            # Ensure albedo and normal textures have matching dimensions
            if albedo_array.shape != normal_array.shape:
                raise ValueError(f"Texture dimension mismatch: albedo {albedo_array.shape} vs normal {normal_array.shape}. "
                               f"Both textures must have identical dimensions.")
            
            # Ensure textures are 3-channel (RGB)
            if len(albedo_array.shape) != 3 or albedo_array.shape[2] != 3:
                raise ValueError(f"Albedo texture must be RGB (3 channels), got shape: {albedo_array.shape}")
            if len(normal_array.shape) != 3 or normal_array.shape[2] != 3:
                raise ValueError(f"Normal texture must be RGB (3 channels), got shape: {normal_array.shape}")
            
            # Validate texture dimensions are reasonable (not empty)
            if albedo_array.size == 0:
                raise ValueError(f"Albedo texture is empty: {albedo_path}")
            if normal_array.size == 0:
                raise ValueError(f"Normal texture is empty: {normal_path}")
            
            return albedo_array, normal_array
            
        except (FileNotFoundError, ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error loading textures: {str(e)}")
    
    def _normalize_light_direction(self, light_dir: list = None) -> np.ndarray:
        """
        Normalize light direction vector to unit length.
        
        Args:
            light_dir (list, optional): Light direction vector [x, y, z]. 
                                      Defaults to [0, 0, 1] if not provided.
        
        Returns:
            np.ndarray: Normalized light direction vector as float32 array
        
        Raises:
            ValueError: If light direction is zero-length vector
            TypeError: If light_dir is not a list or array-like
        """
        # Use default light direction if not provided
        if light_dir is None:
            light_dir = [0, 0, 1]
        
        # Validate input type
        try:
            # Convert to numpy array
            light_vector = np.array(light_dir, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise TypeError(f"light_dir must be array-like (list, tuple, or numpy array), got {type(light_dir)}: {str(e)}")
        
        # Validate vector has exactly 3 components
        if light_vector.shape != (3,):
            raise ValueError(f"Light direction must be a 3D vector [x, y, z], got shape: {light_vector.shape}")
        
        # Calculate vector magnitude
        magnitude = np.linalg.norm(light_vector)
        
        # Handle zero-length vectors with appropriate error handling
        if magnitude < EPSILON:
            raise ValueError(f"Light direction vector cannot be zero-length. "
                           f"Magnitude {magnitude} is below threshold {EPSILON}. "
                           f"Provided vector: {light_dir}")
        
        # Normalize to unit length
        normalized_vector = light_vector / magnitude
        
        # Verify normalization (magnitude should be 1.0)
        final_magnitude = np.linalg.norm(normalized_vector)
        if abs(final_magnitude - 1.0) > EPSILON:
            raise ValueError(f"Normalization failed. Expected magnitude 1.0, got {final_magnitude}")
        
        return normalized_vector
    
    def _compose_final_pixels(self, albedo_array: np.ndarray, shading_values: np.ndarray) -> np.ndarray:
        """
        Compose final rendered pixels by multiplying Albedo by Shading values.
        
        Implements pixel composition with proper broadcasting across RGB channels
        and ensures final pixel values remain in range [0.0, 1.0].
        
        Args:
            albedo_array (np.ndarray): Albedo texture with shape (height, width, 3)
                                     Values should be in range [0.0, 1.0]
            shading_values (np.ndarray): Shading values with shape (height, width)
                                       Values should be in range [0.0, 1.0]
        
        Returns:
            np.ndarray: Final rendered image as float32 array with shape (height, width, 3)
                       Values in range [0.0, 1.0] preserving original dimensions
        
        Raises:
            ValueError: If input arrays have incompatible shapes or invalid values
        """
        # Validate input shapes
        if len(albedo_array.shape) != 3 or albedo_array.shape[2] != 3:
            raise ValueError(f"Albedo array must have shape (height, width, 3), got {albedo_array.shape}")
        
        expected_shading_shape = (albedo_array.shape[0], albedo_array.shape[1])
        if shading_values.shape != expected_shading_shape:
            raise ValueError(f"Shading values must have shape {expected_shading_shape}, got {shading_values.shape}")
        
        # Validate input value ranges
        if albedo_array.min() < 0.0 or albedo_array.max() > 1.0 + EPSILON:
            raise ValueError(f"Albedo values must be in range [0.0, 1.0], "
                           f"got range [{albedo_array.min():.6f}, {albedo_array.max():.6f}]")
        
        if shading_values.min() < 0.0 or shading_values.max() > 1.0 + EPSILON:
            raise ValueError(f"Shading values must be in range [0.0, 1.0], "
                           f"got range [{shading_values.min():.6f}, {shading_values.max():.6f}]")
        
        # Multiply Albedo by Shading values with proper broadcasting across RGB channels
        # albedo_array shape: (height, width, 3)
        # shading_values shape: (height, width)
        # Need to broadcast shading_values to (height, width, 1) for element-wise multiplication
        
        # Expand shading dimensions to enable broadcasting: (height, width) -> (height, width, 1)
        shading_broadcast = shading_values[:, :, np.newaxis]
        
        # Perform element-wise multiplication: Final_Pixel = Albedo * Shading
        final_pixels = albedo_array * shading_broadcast
        
        # Ensure final pixel values remain in range [0.0, 1.0]
        # Since both inputs are in [0.0, 1.0], the product should naturally be in [0.0, 1.0]
        # But we clamp to handle any floating-point precision issues
        final_pixels = np.clip(final_pixels, 0.0, 1.0)
        
        # Return output as float32 numpy array preserving original dimensions
        final_pixels = final_pixels.astype(np.float32)
        
        # Validate output
        if final_pixels.shape != albedo_array.shape:
            raise ValueError(f"Output shape mismatch: expected {albedo_array.shape}, got {final_pixels.shape}")
        
        if final_pixels.min() < 0.0 or final_pixels.max() > 1.0:
            raise ValueError(f"Final pixel values out of range [0.0, 1.0]: "
                           f"[{final_pixels.min():.6f}, {final_pixels.max():.6f}]")
        
        return final_pixels

    def _calculate_lighting(self, normal_array: np.ndarray, light_direction: np.ndarray) -> np.ndarray:
        """
        Calculate lighting using vectorized dot product between Normal and LightDir.
        
        Implements core lighting calculations with proper vectorized operations
        across all pixels and clamping of shading values to [0.0, 1.0] range.
        
        Args:
            normal_array (np.ndarray): Normal vectors array with shape (height, width, 3)
                                     Values should be in range [-1.0, 1.0]
            light_direction (np.ndarray): Normalized light direction vector with shape (3,)
        
        Returns:
            np.ndarray: Shading values array with shape (height, width)
                       Values clamped to range [0.0, 1.0]
        
        Raises:
            ValueError: If input arrays have invalid shapes or values
        """
        # Validate input shapes
        if len(normal_array.shape) != 3 or normal_array.shape[2] != 3:
            raise ValueError(f"Normal array must have shape (height, width, 3), got {normal_array.shape}")
        
        if light_direction.shape != (3,):
            raise ValueError(f"Light direction must have shape (3,), got {light_direction.shape}")
        
        # Validate normal array values are in expected range [-1.0, 1.0]
        if normal_array.min() < -1.0 - EPSILON or normal_array.max() > 1.0 + EPSILON:
            raise ValueError(f"Normal array values must be in range [-1.0, 1.0], "
                           f"got range [{normal_array.min():.6f}, {normal_array.max():.6f}]")
        
        # Implement vectorized dot product calculation between Normal and LightDir
        # normal_array shape: (height, width, 3)
        # light_direction shape: (3,)
        # Result shape: (height, width)
        
        # Use numpy's einsum for efficient vectorized dot product
        # 'hwc,c->hw' means: for each pixel (h,w), dot product the 3 channels (c) with light vector (c)
        shading_values = np.einsum('hwc,c->hw', normal_array, light_direction)
        
        # Alternative implementation using np.sum for clarity:
        # shading_values = np.sum(normal_array * light_direction[np.newaxis, np.newaxis, :], axis=2)
        
        # Clamp shading values to range [0.0, 1.0] handling negative values
        # Negative dot product values indicate surfaces facing away from light
        shading_values = np.clip(shading_values, 0.0, 1.0)
        
        # Ensure output is float32 for consistency
        shading_values = shading_values.astype(np.float32)
        
        # Validate output range
        if shading_values.min() < 0.0 or shading_values.max() > 1.0:
            raise ValueError(f"Shading values out of range [0.0, 1.0]: "
                           f"[{shading_values.min():.6f}, {shading_values.max():.6f}]")
        
        return shading_values

    def _calculate_psnr(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two images.
        
        Implements PSNR calculation using Mean Squared Error and the formula:
        PSNR = 20 * log10(MAX_VAL / sqrt(MSE))
        
        Args:
            reference (np.ndarray): Reference image array with values in [0.0, 1.0]
            rendered (np.ndarray): Rendered image array to compare with values in [0.0, 1.0]
        
        Returns:
            float: PSNR value in decibels. Returns float('inf') for identical images.
        
        Raises:
            ValueError: If input arrays have different shapes or invalid values
        """
        # Validate input shapes match
        if reference.shape != rendered.shape:
            raise ValueError(f"Image shapes must match: reference {reference.shape} vs rendered {rendered.shape}")
        
        # Validate input value ranges
        if reference.min() < 0.0 or reference.max() > 1.0 + EPSILON:
            raise ValueError(f"Reference image values must be in range [0.0, 1.0], "
                           f"got range [{reference.min():.6f}, {reference.max():.6f}]")
        
        if rendered.min() < 0.0 or rendered.max() > 1.0 + EPSILON:
            raise ValueError(f"Rendered image values must be in range [0.0, 1.0], "
                           f"got range [{rendered.min():.6f}, {rendered.max():.6f}]")
        
        # Compute Mean Squared Error
        mse = np.mean((reference - rendered) ** 2)
        
        # Handle edge case of identical images (infinite PSNR)
        if mse < EPSILON:
            return float('inf')
        
        # Apply PSNR formula: 20 * log10(MAX_VAL / sqrt(MSE))
        # For float images with values in [0.0, 1.0], MAX_VAL = 1.0
        psnr_value = 20.0 * np.log10(FLOAT_MAX / np.sqrt(mse))
        
        return float(psnr_value)
    
    def _calculate_ssim(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        """
        Compute Structural Similarity Index using skimage.metrics.structural_similarity.
        
        Handles luminance channel processing or RGB averaging for single score
        and configures appropriate parameters for SSIM computation.
        
        Args:
            reference (np.ndarray): Reference image array with values in [0.0, 1.0]
            rendered (np.ndarray): Rendered image array to compare with values in [0.0, 1.0]
        
        Returns:
            float: SSIM value between 0 and 1, where 1 indicates identical images
        
        Raises:
            ValueError: If input arrays have different shapes or invalid values
        """
        # Validate input shapes match
        if reference.shape != rendered.shape:
            raise ValueError(f"Image shapes must match: reference {reference.shape} vs rendered {rendered.shape}")
        
        # Validate input value ranges
        if reference.min() < 0.0 or reference.max() > 1.0 + EPSILON:
            raise ValueError(f"Reference image values must be in range [0.0, 1.0], "
                           f"got range [{reference.min():.6f}, {reference.max():.6f}]")
        
        if rendered.min() < 0.0 or rendered.max() > 1.0 + EPSILON:
            raise ValueError(f"Rendered image values must be in range [0.0, 1.0], "
                           f"got range [{rendered.min():.6f}, {rendered.max():.6f}]")
        
        # Handle different image formats
        if len(reference.shape) == 3 and reference.shape[2] == 3:
            # RGB image - calculate SSIM for each channel and average
            # Use multichannel=True to handle RGB images properly
            ssim_value = structural_similarity(
                reference, 
                rendered, 
                multichannel=True,
                channel_axis=2,  # RGB channels are on axis 2
                data_range=1.0,  # Data range is [0.0, 1.0] for float images
                win_size=7       # Default window size, must be odd and >= 7
            )
        elif len(reference.shape) == 2:
            # Grayscale image
            ssim_value = structural_similarity(
                reference, 
                rendered, 
                data_range=1.0,  # Data range is [0.0, 1.0] for float images
                win_size=7       # Default window size, must be odd and >= 7
            )
        else:
            raise ValueError(f"Unsupported image format: shape {reference.shape}. "
                           f"Expected 2D grayscale or 3D RGB images.")
        
        # Ensure SSIM value is in valid range [-1, 1] (SSIM can be negative)
        if ssim_value < -1.0 - EPSILON or ssim_value > 1.0 + EPSILON:
            raise ValueError(f"SSIM value out of expected range [-1.0, 1.0]: {ssim_value}")
        
        return float(ssim_value)