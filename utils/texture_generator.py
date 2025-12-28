#!/usr/bin/env python3
"""
Texture Downloader and Normal Map Generator for Research Demonstration
Downloads wood texture from internet and generates corresponding normal map
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import urllib.request
import urllib.error

class TextureGenerator:
    def __init__(self, size=(512, 512)):
        self.size = size
        
    def download_wood_texture(self):
        """Download a wood texture from the internet"""
        # List of free wood texture URLs (CC0 or public domain)
        texture_urls = [
            "https://cdn.pixabay.com/photo/2017/10/10/07/48/wood-2835775_1280.jpg",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&h=512&fit=crop",
            "https://cdn.pixabay.com/photo/2016/11/29/05/45/wood-1867667_1280.jpg",
            "https://images.pexels.com/photos/172276/pexels-photo-172276.jpeg?w=512&h=512&fit=crop"
        ]
        
        for i, url in enumerate(texture_urls):
            try:
                print(f"Attempting to download texture from URL {i+1}...")
                
                # Set user agent to avoid blocking
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    image_data = response.read()
                
                # Save the downloaded image
                with open("demo_textures/downloaded_wood.jpg", "wb") as f:
                    f.write(image_data)
                
                # Load and process the image
                img = Image.open("demo_textures/downloaded_wood.jpg")
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img = img.resize(self.size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array (0-1 range)
                wood_texture = np.array(img) / 255.0
                
                print(f"Successfully downloaded and processed wood texture!")
                return wood_texture
                
            except Exception as e:
                print(f"Failed to download from URL {i+1}: {e}")
                continue
        
        # If all downloads fail, create a simple wood-like texture
        print("All downloads failed, creating fallback texture...")
        return self.create_fallback_texture()
    
    def create_fallback_texture(self):
        """Create a simple wood-like texture as fallback"""
        width, height = self.size
        
        # Create wood grain pattern
        x = np.linspace(0, 8, width)
        y = np.linspace(0, 8, height)
        X, Y = np.meshgrid(x, y)
        
        # Wood grain
        grain = np.sin(X * 1.5 + np.sin(Y * 0.3) * 3) * 0.3
        grain += np.sin(X * 3 + np.sin(Y * 0.2) * 2) * 0.15
        
        # Normalize
        grain = (grain - grain.min()) / (grain.max() - grain.min())
        
        # Wood colors
        dark_brown = np.array([0.4, 0.25, 0.15])
        light_brown = np.array([0.7, 0.5, 0.3])
        
        # Create RGB texture
        wood_rgb = np.zeros((height, width, 3))
        for i in range(3):
            wood_rgb[:, :, i] = dark_brown[i] + (light_brown[i] - dark_brown[i]) * grain
        
        return np.clip(wood_rgb, 0, 1)
    
    
    def generate_normal_map_from_albedo(self, albedo_texture):
        """Generate normal map from albedo texture using edge detection"""
        # Convert to grayscale for height map
        if len(albedo_texture.shape) == 3:
            grayscale = np.dot(albedo_texture, [0.299, 0.587, 0.114])
        else:
            grayscale = albedo_texture
        
        # Apply Gaussian blur to smooth the height map
        height_map = ndimage.gaussian_filter(grayscale, sigma=1.0)
        
        # Calculate gradients
        grad_x = np.gradient(height_map, axis=1)
        grad_y = np.gradient(height_map, axis=0)
        
        # Scale gradients for normal map intensity
        scale_factor = 2.0
        grad_x *= scale_factor
        grad_y *= scale_factor
        
        # Calculate normal vectors
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(normal_x)
        
        # Normalize the vectors
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= length
        normal_y /= length
        normal_z /= length
        
        # Convert to normal map format (0-1 range)
        normal_map = np.zeros((*height_map.shape, 3))
        normal_map[:, :, 0] = (normal_x + 1) * 0.5  # R channel
        normal_map[:, :, 1] = (normal_y + 1) * 0.5  # G channel  
        normal_map[:, :, 2] = (normal_z + 1) * 0.5  # B channel
        
        return np.clip(normal_map, 0, 1)
    
    def save_texture(self, texture, filename):
        """Save texture as PNG image"""
        texture_8bit = (texture * 255).astype(np.uint8)
        img = Image.fromarray(texture_8bit)
        img.save(filename)
        print(f"Saved texture: {filename}")
    
    def generate_demo_textures(self):
        """Generate demonstration textures for the research paper"""
        print("Downloading wood texture and generating normal map...")
        
        # Create output directory
        os.makedirs("demo_textures", exist_ok=True)
        
        # Download wood texture
        wood_albedo = self.download_wood_texture()
        
        # Generate normal map from the albedo
        wood_normal = self.generate_normal_map_from_albedo(wood_albedo)
        
        # Save original textures
        self.save_texture(wood_albedo, "demo_textures/wood_albedo_original.png")
        self.save_texture(wood_normal, "demo_textures/wood_normal_original.png")
        
        return wood_albedo, wood_normal

if __name__ == "__main__":
    generator = TextureGenerator()
    wood_albedo, wood_normal = generator.generate_demo_textures()
    print("Demo textures generated successfully!")