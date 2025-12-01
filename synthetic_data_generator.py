"""
Synthetic Data Generator for Water Body Classification
Generates 1000 RGB images (512x512) for 6 water body classes with realistic patterns

Author: B.Tech Research Team
Classes: swamp, river, estuary, tidal_pool, shallow_water, flood_plain
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import albumentations as A
from sklearn.cluster import KMeans
import os
import json
from tqdm import tqdm


class WaterBodySyntheticGenerator:
    """
    Generates synthetic water body images for data augmentation
    Focuses on 6 classes relevant to Sundarbans and Indian wetlands
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.height, self.width = image_size
        
        # 6 water body classes for research
        self.water_classes = {
            'swamp': 0,
            'river': 1, 
            'estuary': 2,
            'tidal_pool': 3,
            'shallow_water': 4,
            'flood_plain': 5
        }
        
        # Realistic color palettes based on Sundarbans analysis
        self.color_palettes = {
            'swamp': {
                'water': [(45, 85, 65), (55, 95, 75), (35, 75, 55)],  # Dark green-brown
                'vegetation': [(85, 107, 47), (124, 252, 0), (107, 142, 35)],  # Mangrove green
                'mud': [(139, 69, 19), (160, 82, 45), (205, 133, 63)]  # Muddy brown
            },
            'river': {
                'water': [(64, 128, 128), (70, 130, 180), (100, 149, 237)],  # Flowing blue
                'bank': [(160, 82, 45), (205, 133, 63), (222, 184, 135)],  # Sandy banks
                'vegetation': [(34, 139, 34), (85, 107, 47), (107, 142, 35)]  # Riparian green
            },
            'estuary': {
                'water': [(120, 140, 160), (135, 155, 175), (150, 170, 190)],  # Brackish blue-gray
                'sediment': [(210, 180, 140), (222, 184, 135), (245, 222, 179)],  # Sediment plume
                'vegetation': [(107, 142, 35), (85, 107, 47), (124, 252, 0)]  # Salt-tolerant vegetation
            },
            'tidal_pool': {
                'water': [(100, 120, 140), (120, 140, 160), (140, 160, 180)],  # Shallow blue-gray
                'exposed': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],  # Exposed mud
                'algae': [(85, 107, 47), (107, 142, 35), (124, 252, 0)]  # Algal mats
            },
            'shallow_water': {
                'water': [(180, 200, 220), (200, 220, 240), (220, 235, 250)],  # Very light blue
                'bottom': [(222, 184, 135), (245, 222, 179), (255, 228, 196)],  # Visible bottom
                'vegetation': [(124, 252, 0), (144, 238, 144), (152, 251, 152)]  # Aquatic vegetation
            },
            'flood_plain': {
                'water': [(100, 149, 237), (135, 206, 235), (173, 216, 230)],  # Flood water
                'grass': [(107, 142, 35), (124, 252, 0), (144, 238, 144)],  # Flooded grass
                'soil': [(160, 82, 45), (205, 133, 63), (139, 69, 19)]  # Exposed soil
            }
        }
        
        # Augmentation pipeline optimized for satellite imagery
        self.augmentations = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
        ])
    
    def generate_perlin_noise(self, scale: float = 0.1, octaves: int = 4) -> np.ndarray:
        """
        Generate Perlin-like noise for realistic terrain
        
        Args:
            scale: Noise scale factor
            octaves: Number of noise octaves
            
        Returns:
            2D noise array
        """
        
        noise = np.zeros((self.height, self.width))
        
        for octave in range(octaves):
            # Generate noise at different scales
            octave_scale = scale * (2 ** octave)
            octave_height = max(4, self.height // (2 ** octave))
            octave_width = max(4, self.width // (2 ** octave))
            
            # Create random noise
            octave_noise = np.random.rand(octave_height, octave_width)
            
            # Upscale to full size
            octave_noise = cv2.resize(octave_noise, (self.width, self.height), 
                                    interpolation=cv2.INTER_CUBIC)
            
            # Add to final noise with decreasing amplitude
            amplitude = 1.0 / (2 ** octave)
            noise += amplitude * octave_noise
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        return noise
    
    def generate_base_terrain(self, water_class: str) -> np.ndarray:
        """
        Generate base terrain appropriate for water body type
        
        Args:
            water_class: Type of water body
            
        Returns:
            RGB base terrain image
        """
        
        terrain = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Generate elevation map
        elevation = self.generate_perlin_noise(scale=0.08, octaves=3)
        
        # Generate vegetation density map
        vegetation = self.generate_perlin_noise(scale=0.12, octaves=2)
        
        # Get color palette for this water class
        palette = self.color_palettes[water_class]
        
        # Assign colors based on elevation and vegetation
        for i in range(self.height):
            for j in range(self.width):
                elev = elevation[i, j]
                veg = vegetation[i, j]
                
                if water_class == 'swamp':
                    # Dense vegetation with muddy areas
                    if veg > 0.6:
                        color = random.choice(palette['vegetation'])
                    elif elev < 0.3:
                        color = random.choice(palette['mud'])
                    else:
                        color = random.choice(palette['vegetation'])
                        
                elif water_class in ['river', 'estuary']:
                    # Banks with vegetation
                    if elev < 0.4:
                        color = random.choice(palette.get('bank', palette['vegetation']))
                    else:
                        color = random.choice(palette['vegetation'])
                        
                elif water_class == 'flood_plain':
                    # Mix of grass and exposed soil
                    if veg > 0.5:
                        color = random.choice(palette['grass'])
                    else:
                        color = random.choice(palette['soil'])
                        
                else:  # tidal_pool, shallow_water
                    # Exposed areas with sparse vegetation
                    if veg > 0.7:
                        color = random.choice(palette.get('vegetation', palette.get('algae', [(107, 142, 35)])))
                    else:
                        color = random.choice(palette.get('exposed', palette.get('bottom', [(222, 184, 135)])))
                
                # Add noise for realism
                noise = np.random.randint(-15, 15, 3)
                final_color = np.clip(np.array(color) + noise, 0, 255)
                terrain[i, j] = final_color
        
        return terrain
    
    def generate_water_body_mask(self, water_class: str, complexity: str = 'medium') -> np.ndarray:
        """
        Generate water body mask for specific class
        
        Args:
            water_class: Type of water body
            complexity: Complexity level
            
        Returns:
            Binary mask for water body
        """
        
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        if water_class == 'swamp':
            # Fragmented water areas typical of mangrove swamps
            num_fragments = random.randint(8, 15) if complexity == 'complex' else random.randint(4, 8)
            
            for _ in range(num_fragments):
                # Create irregular water patches
                center_x = random.randint(50, self.width - 50)
                center_y = random.randint(50, self.height - 50)
                
                # Generate irregular shape using random walk
                points = self._generate_irregular_shape(center_x, center_y, 
                                                      base_radius=random.randint(20, 50))
                
                # Fill polygon
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        
        elif water_class == 'river':
            # Meandering river channel
            river_width = random.randint(15, 35)
            
            # Generate river path
            control_points = []
            num_controls = 6 if complexity == 'simple' else 10
            
            for i in range(num_controls):
                x = i * self.width // (num_controls - 1)
                y = self.height // 2 + random.randint(-self.height // 4, self.height // 4)
                control_points.append((x, y))
            
            # Draw river path
            for i in range(len(control_points) - 1):
                cv2.line(mask, control_points[i], control_points[i+1], 255, river_width)
            
            # Add meandering details
            if complexity != 'simple':
                for point in control_points[1:-1]:
                    # Add oxbow lakes or side channels
                    if random.random() < 0.3:
                        side_radius = random.randint(10, 25)
                        cv2.circle(mask, point, side_radius, 255, -1)
        
        elif water_class == 'estuary':
            # Funnel-shaped estuary with distributaries
            mouth_width = random.randint(80, 150)
            river_width = random.randint(20, 40)
            
            # Main estuary channel
            points = [
                (0, self.height // 2 - river_width // 2),
                (self.width // 3, self.height // 2 - mouth_width // 4),
                (2 * self.width // 3, self.height // 2 - mouth_width // 3),
                (self.width, self.height // 2 - mouth_width // 2),
                (self.width, self.height // 2 + mouth_width // 2),
                (2 * self.width // 3, self.height // 2 + mouth_width // 3),
                (self.width // 3, self.height // 2 + mouth_width // 4),
                (0, self.height // 2 + river_width // 2)
            ]
            
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            
            # Add distributary channels
            if complexity != 'simple':
                num_distributaries = random.randint(2, 4)
                for _ in range(num_distributaries):
                    start_x = random.randint(self.width // 4, 3 * self.width // 4)
                    start_y = self.height // 2 + random.randint(-mouth_width // 4, mouth_width // 4)
                    end_x = self.width
                    end_y = start_y + random.randint(-20, 20)
                    
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, 
                            random.randint(8, 15))
        
        elif water_class == 'tidal_pool':
            # Small scattered pools typical of tidal areas
            num_pools = random.randint(6, 12) if complexity == 'complex' else random.randint(3, 6)
            
            for _ in range(num_pools):
                center_x = random.randint(30, self.width - 30)
                center_y = random.randint(30, self.height - 30)
                
                # Irregular pool shape
                if random.random() < 0.7:
                    # Elliptical pool
                    radius_x = random.randint(8, 25)
                    radius_y = random.randint(8, 25)
                    angle = random.randint(0, 180)
                    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 
                              angle, 0, 360, 255, -1)
                else:
                    # Irregular polygon pool
                    points = self._generate_irregular_shape(center_x, center_y, 
                                                          base_radius=random.randint(10, 20))
                    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        
        elif water_class == 'shallow_water':
            # Large shallow area with irregular edges
            # Create main water body covering 40-70% of image
            coverage = random.uniform(0.4, 0.7)
            
            # Generate irregular boundary
            boundary_points = []
            num_points = random.randint(12, 20)
            
            center_x, center_y = self.width // 2, self.height // 2
            base_radius = int(np.sqrt(coverage) * min(self.width, self.height) / 2)
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                radius = base_radius + random.randint(-base_radius // 3, base_radius // 3)
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                boundary_points.append((int(x), int(y)))
            
            cv2.fillPoly(mask, [np.array(boundary_points, dtype=np.int32)], 255)
        
        elif water_class == 'flood_plain':
            # Wide flooded area with irregular water level
            flood_level = random.randint(self.height // 4, 3 * self.height // 4)
            
            # Create irregular flood boundary
            boundary_points = []
            
            # Top boundary (irregular)
            for x in range(0, self.width, 20):
                y = flood_level + random.randint(-30, 30)
                boundary_points.append((x, max(0, min(self.height, y))))
            
            # Right edge
            boundary_points.append((self.width, self.height))
            
            # Bottom edge
            boundary_points.append((0, self.height))
            
            cv2.fillPoly(mask, [np.array(boundary_points, dtype=np.int32)], 255)
            
            # Add isolated flooded patches above main flood level
            if complexity != 'simple':
                num_patches = random.randint(3, 8)
                for _ in range(num_patches):
                    patch_x = random.randint(50, self.width - 50)
                    patch_y = random.randint(50, flood_level - 50)
                    patch_radius = random.randint(15, 35)
                    
                    cv2.circle(mask, (patch_x, patch_y), patch_radius, 255, -1)
        
        return mask
    
    def _generate_irregular_shape(self, center_x: int, center_y: int, 
                                 base_radius: int, irregularity: float = 0.3) -> List[Tuple[int, int]]:
        """Generate irregular shape using random walk around center"""
        
        num_points = random.randint(8, 16)
        points = []
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            # Add irregularity
            radius = base_radius * (1 + irregularity * (random.random() - 0.5))
            angle_offset = irregularity * (random.random() - 0.5)
            
            x = center_x + radius * np.cos(angle + angle_offset)
            y = center_y + radius * np.sin(angle + angle_offset)
            
            points.append((int(x), int(y)))
        
        return points
    
    def apply_water_to_terrain(self, terrain: np.ndarray, water_mask: np.ndarray, 
                              water_class: str) -> np.ndarray:
        """
        Apply water body to terrain with realistic colors and effects
        
        Args:
            terrain: Base terrain image
            water_mask: Binary water mask
            water_class: Type of water body
            
        Returns:
            Image with water body applied
        """
        
        result = terrain.copy()
        palette = self.color_palettes[water_class]
        
        # Apply water colors
        water_pixels = water_mask > 0
        
        if water_pixels.any():
            # Choose base water color
            base_water_color = random.choice(palette['water'])
            
            # Add spatial variation to water color
            water_coords = np.where(water_pixels)
            
            for i, (y, x) in enumerate(zip(water_coords[0], water_coords[1])):
                # Add subtle color variation based on position
                position_factor = (x / self.width + y / self.height) / 2
                color_variation = np.array([
                    random.randint(-10, 10),
                    random.randint(-10, 10), 
                    random.randint(-10, 10)
                ])
                
                # Apply depth effect for shallow water
                if water_class == 'shallow_water':
                    # Mix water color with bottom color based on "depth"
                    depth_factor = random.uniform(0.3, 0.8)
                    bottom_color = random.choice(palette['bottom'])
                    final_color = (depth_factor * np.array(base_water_color) + 
                                 (1 - depth_factor) * np.array(bottom_color))
                else:
                    final_color = np.array(base_water_color)
                
                final_color = np.clip(final_color + color_variation, 0, 255)
                result[y, x] = final_color.astype(np.uint8)
        
        # Add class-specific environmental effects
        if water_class == 'estuary':
            # Add sediment plumes
            self._add_sediment_plumes(result, water_mask, palette['sediment'])
        elif water_class == 'tidal_pool':
            # Add algal mats
            self._add_algal_mats(result, water_mask, palette['algae'])
        elif water_class == 'swamp':
            # Add floating vegetation
            self._add_floating_vegetation(result, water_mask, palette['vegetation'])
        
        return result
    
    def _add_sediment_plumes(self, image: np.ndarray, water_mask: np.ndarray, 
                           sediment_colors: List[Tuple[int, int, int]]):
        """Add sediment plumes to estuary water"""
        
        # Create sediment concentration map
        sediment_noise = self.generate_perlin_noise(scale=0.15, octaves=2)
        
        water_coords = np.where(water_mask > 0)
        
        for y, x in zip(water_coords[0], water_coords[1]):
            sediment_concentration = sediment_noise[y, x]
            
            if sediment_concentration > 0.6:  # High sediment area
                sediment_color = random.choice(sediment_colors)
                # Blend with existing water color
                blend_factor = 0.4
                current_color = image[y, x]
                blended_color = (blend_factor * np.array(sediment_color) + 
                               (1 - blend_factor) * current_color)
                image[y, x] = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    def _add_algal_mats(self, image: np.ndarray, water_mask: np.ndarray,
                       algae_colors: List[Tuple[int, int, int]]):
        """Add algal mats to tidal pools"""
        
        # Create algae distribution map
        algae_noise = self.generate_perlin_noise(scale=0.2, octaves=1)
        
        water_coords = np.where(water_mask > 0)
        
        for y, x in zip(water_coords[0], water_coords[1]):
            algae_density = algae_noise[y, x]
            
            if algae_density > 0.7:  # High algae area
                algae_color = random.choice(algae_colors)
                # Blend with water
                blend_factor = 0.5
                current_color = image[y, x]
                blended_color = (blend_factor * np.array(algae_color) + 
                               (1 - blend_factor) * current_color)
                image[y, x] = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    def _add_floating_vegetation(self, image: np.ndarray, water_mask: np.ndarray,
                               veg_colors: List[Tuple[int, int, int]]):
        """Add floating vegetation to swamp water"""
        
        # Create vegetation coverage map
        veg_noise = self.generate_perlin_noise(scale=0.18, octaves=2)
        
        water_coords = np.where(water_mask > 0)
        
        for y, x in zip(water_coords[0], water_coords[1]):
            veg_coverage = veg_noise[y, x]
            
            if veg_coverage > 0.5:  # Vegetation present
                veg_color = random.choice(veg_colors)
                # Partial coverage blend
                coverage_factor = min(0.7, veg_coverage)
                current_color = image[y, x]
                blended_color = (coverage_factor * np.array(veg_color) + 
                               (1 - coverage_factor) * current_color)
                image[y, x] = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    def add_satellite_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic satellite imagery artifacts"""
        
        # Atmospheric haze (slight blur)
        if random.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), 0.8)
        
        # Sensor noise
        noise = np.random.normal(0, 3, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # Compression artifacts (simulate JPEG compression)
        if random.random() < 0.2:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(75, 95)]
            _, encoded_img = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Slight color cast (atmospheric effects)
        if random.random() < 0.25:
            color_cast = np.random.randint(-5, 5, 3)
            image = np.clip(image.astype(int) + color_cast, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_single_image(self, water_class: str, 
                            complexity: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate single synthetic image with ground truth
        
        Args:
            water_class: Type of water body to generate
            complexity: Complexity level
            
        Returns:
            Tuple of (RGB image, class mask)
        """
        
        # Generate base terrain
        terrain = self.generate_base_terrain(water_class)
        
        # Generate water body mask
        water_mask = self.generate_water_body_mask(water_class, complexity)
        
        # Apply water to terrain
        image = self.apply_water_to_terrain(terrain, water_mask, water_class)
        
        # Add satellite artifacts
        image = self.add_satellite_artifacts(image)
        
        # Create class mask (0 = background, class_id = water body)
        class_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        class_mask[water_mask > 0] = self.water_classes[water_class]
        
        return image, class_mask
    
    def generate_dataset(self, num_images: int = 1000, 
                        save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete synthetic dataset
        
        Args:
            num_images: Total number of images to generate
            save_path: Optional path to save generated data
            
        Returns:
            Tuple of (images array, masks array)
        """
        
        if save_path is None:
            save_path = "/content/drive/MyDrive/WaterBodyResearch/data/synthetic"
        
        os.makedirs(save_path, exist_ok=True)
        
        # Class distribution (balanced with slight emphasis on swamp for Sundarbans)
        class_distribution = {
            'swamp': 0.25,      # Primary focus for Sundarbans
            'river': 0.20,      # Common in all regions
            'estuary': 0.15,    # Sundarbans specific
            'tidal_pool': 0.15, # Coastal areas
            'shallow_water': 0.15, # Common across regions
            'flood_plain': 0.10    # Brahmaputra specific
        }
        
        # Complexity distribution
        complexity_dist = ['simple', 'medium', 'complex']
        complexity_probs = [0.2, 0.6, 0.2]
        
        images = []
        masks = []
        metadata = []
        
        print(f"Generating {num_images} synthetic images...")
        
        for img_idx in tqdm(range(num_images)):
            # Sample class and complexity
            water_class = np.random.choice(
                list(class_distribution.keys()),
                p=list(class_distribution.values())
            )
            complexity = np.random.choice(complexity_dist, p=complexity_probs)
            
            # Generate image and mask
            image, mask = self.generate_single_image(water_class, complexity)
            
            # Apply augmentations (50% chance)
            if random.random() < 0.5:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            images.append(image)
            masks.append(mask)
            
            # Store metadata
            metadata.append({
                'image_id': f"synthetic_{img_idx:04d}",
                'water_class': water_class,
                'class_id': self.water_classes[water_class],
                'complexity': complexity,
                'augmented': random.random() < 0.5
            })
            
            # Save individual files every 100 images to manage memory
            if (img_idx + 1) % 100 == 0:
                batch_start = img_idx - 99
                batch_end = img_idx + 1
                
                batch_images = np.array(images[batch_start:batch_end])
                batch_masks = np.array(masks[batch_start:batch_end])
                
                np.save(f"{save_path}/images_batch_{batch_start//100:02d}.npy", batch_images)
                np.save(f"{save_path}/masks_batch_{batch_start//100:02d}.npy", batch_masks)
                
                print(f"Saved batch {batch_start//100:02d} ({batch_start}-{batch_end-1})")
        
        # Save remaining images
        if len(images) % 100 != 0:
            remaining_start = (len(images) // 100) * 100
            remaining_images = np.array(images[remaining_start:])
            remaining_masks = np.array(masks[remaining_start:])
            
            np.save(f"{save_path}/images_batch_{len(images)//100:02d}.npy", remaining_images)
            np.save(f"{save_path}/masks_batch_{len(images)//100:02d}.npy", remaining_masks)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(f"{save_path}/metadata.csv", index=False)
        
        # Save class distribution statistics
        class_counts = metadata_df['water_class'].value_counts()
        print("\nGenerated dataset statistics:")
        print(class_counts)
        
        return np.array(images), np.array(masks)
    
    def visualize_samples(self, num_samples: int = 6):
        """Visualize sample generated images for each class"""
        
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        
        for idx, water_class in enumerate(self.water_classes.keys()):
            # Generate sample
            image, mask = self.generate_single_image(water_class, 'medium')
            
            # Plot image
            axes[0, idx].imshow(image)
            axes[0, idx].set_title(f'{water_class.title()}')
            axes[0, idx].axis('off')
            
            # Plot mask
            axes[1, idx].imshow(mask, cmap='viridis')
            axes[1, idx].set_title(f'{water_class.title()} Mask')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/WaterBodyResearch/synthetic_samples.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()


def estimate_storage_requirements():
    """Estimate storage requirements for synthetic dataset"""
    
    # Image specifications
    image_size = 512 * 512 * 3  # RGB
    mask_size = 512 * 512 * 1   # Single channel mask
    num_images = 1000
    
    # Calculate sizes
    images_size_mb = (image_size * num_images) / (1024 * 1024)
    masks_size_mb = (mask_size * num_images) / (1024 * 1024)
    total_size_mb = images_size_mb + masks_size_mb
    
    print(f"Storage Requirements Estimate:")
    print(f"Images: {images_size_mb:.1f} MB")
    print(f"Masks: {masks_size_mb:.1f} MB") 
    print(f"Total: {total_size_mb:.1f} MB (~{total_size_mb/1024:.2f} GB)")
    print(f"With compression: ~{total_size_mb * 0.7:.1f} MB")
    
    return total_size_mb


# Example usage and testing
if __name__ == "__main__":
    # Estimate storage
    estimate_storage_requirements()
    
    # Create generator
    generator = WaterBodySyntheticGenerator(image_size=(512, 512))
    
    # Visualize samples
    print("Generating sample visualizations...")
    generator.visualize_samples()
    
    # Generate small test dataset
    print("Generating test dataset (100 images)...")
    test_images, test_masks = generator.generate_dataset(
        num_images=100,
        save_path="/content/drive/MyDrive/WaterBodyResearch/data/synthetic_test"
    )
    
    print(f"Generated test dataset: {test_images.shape}, {test_masks.shape}")
    print("Unique classes in test set:", np.unique(test_masks))