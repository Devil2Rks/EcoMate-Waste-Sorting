"""
Data Preprocessing Pipeline for Water Body Classification
Handles NDWI computation, normalization, and data loading

Author: B.Tech Research Team
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import cv2
from typing import Tuple, List, Dict, Optional, Union
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A


class WaterBodyDataset(Dataset):
    """
    PyTorch Dataset for water body classification with temporal sequences
    """
    
    def __init__(self, data_path: str, split: str = 'train', 
                 use_temporal: bool = True, transform: Optional[A.Compose] = None):
        """
        Initialize dataset
        
        Args:
            data_path: Path to processed data
            split: Dataset split ('train', 'val', 'test')
            use_temporal: Whether to use temporal sequences
            transform: Albumentations transform pipeline
        """
        
        self.data_path = data_path
        self.split = split
        self.use_temporal = use_temporal
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(data_path, f"{split}_metadata.csv")
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = self._create_metadata()
        
        # Water body class mapping
        self.class_mapping = {
            'swamp': 0,
            'river': 1,
            'estuary': 2, 
            'tidal_pool': 3,
            'shallow_water': 4,
            'flood_plain': 5
        }
        
        self.num_classes = len(self.class_mapping)
        
    def _create_metadata(self) -> pd.DataFrame:
        """Create metadata from available files"""
        
        metadata = []
        
        # Scan for available files
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.npy') and 'image' in file:
                    # Extract information from filename
                    base_name = file.replace('.npy', '')
                    mask_file = base_name.replace('image', 'mask') + '.npy'
                    
                    if os.path.exists(os.path.join(root, mask_file)):
                        metadata.append({
                            'image_path': os.path.join(root, file),
                            'mask_path': os.path.join(root, mask_file),
                            'region': os.path.basename(root),
                            'split': self.split
                        })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with image, mask, and metadata
        """
        
        row = self.metadata.iloc[idx]
        
        # Load image and mask
        if row['image_path'].endswith('.npy'):
            image = np.load(row['image_path'])
            mask = np.load(row['mask_path'])
        else:
            # Load GeoTIFF files
            image = self._load_geotiff(row['image_path'])
            mask = self._load_geotiff(row['mask_path'])
        
        # Handle temporal sequences
        if self.use_temporal and len(image.shape) == 4:  # (T, H, W, C)
            # Use all temporal frames
            temporal_frames = image
        else:
            # Single frame - add temporal dimension
            if len(image.shape) == 3:
                temporal_frames = image[np.newaxis, ...]  # (1, H, W, C)
            else:
                temporal_frames = image
        
        # Apply transformations
        if self.transform:
            # Apply same transform to all temporal frames
            transformed_frames = []
            for t in range(temporal_frames.shape[0]):
                frame = temporal_frames[t]
                if len(frame.shape) == 3 and frame.shape[2] > 3:
                    # Separate RGB and additional channels
                    rgb = frame[:, :, :3]
                    additional = frame[:, :, 3:]
                    
                    # Transform RGB
                    transformed = self.transform(image=rgb, mask=mask)
                    transformed_rgb = transformed['image']
                    transformed_mask = transformed['mask']
                    
                    # Concatenate back
                    transformed_frame = np.concatenate([transformed_rgb, additional], axis=2)
                else:
                    transformed = self.transform(image=frame, mask=mask)
                    transformed_frame = transformed['image']
                    transformed_mask = transformed['mask']
                
                transformed_frames.append(transformed_frame)
            
            temporal_frames = np.array(transformed_frames)
            mask = transformed_mask
        
        # Convert to torch tensors
        # Images: (T, C, H, W)
        temporal_frames = torch.from_numpy(temporal_frames).permute(0, 3, 1, 2).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return {
            'images': temporal_frames,
            'mask': mask,
            'region': row.get('region', 'unknown'),
            'image_id': f"{self.split}_{idx:04d}"
        }
    
    def _load_geotiff(self, file_path: str) -> np.ndarray:
        """Load GeoTIFF file and return as numpy array"""
        
        with rasterio.open(file_path) as src:
            # Read all bands
            image = src.read()  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            
            # Handle different data types
            if image.dtype == np.uint16:
                # Scale to 0-255 range for visualization
                image = (image / 65535.0 * 255).astype(np.uint8)
            elif image.dtype == np.float32:
                # Clip and scale
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image


class NDWIProcessor:
    """
    Processes NDWI computation and water body segmentation
    """
    
    def __init__(self):
        self.water_classes = {
            'swamp': 0, 'river': 1, 'estuary': 2,
            'tidal_pool': 3, 'shallow_water': 4, 'flood_plain': 5
        }
    
    def compute_ndwi(self, image: np.ndarray, 
                    green_idx: int = 1, nir_idx: int = 3) -> np.ndarray:
        """
        Compute NDWI from multispectral image
        
        Args:
            image: Multispectral image array (H, W, C)
            green_idx: Index of green band
            nir_idx: Index of NIR band
            
        Returns:
            NDWI array (H, W)
        """
        
        green = image[:, :, green_idx].astype(np.float32)
        nir = image[:, :, nir_idx].astype(np.float32)
        
        # Avoid division by zero
        denominator = green + nir
        denominator[denominator == 0] = 1e-8
        
        ndwi = (green - nir) / denominator
        
        # Clip to valid range
        ndwi = np.clip(ndwi, -1, 1)
        
        return ndwi
    
    def threshold_water_bodies(self, ndwi: np.ndarray, 
                             threshold: float = 0.0) -> np.ndarray:
        """
        Create binary water mask using NDWI threshold
        
        Args:
            ndwi: NDWI array
            threshold: NDWI threshold for water detection
            
        Returns:
            Binary water mask
        """
        
        water_mask = (ndwi > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        
        return water_mask
    
    def classify_water_bodies_kmeans(self, image: np.ndarray, water_mask: np.ndarray,
                                   ndwi: np.ndarray, n_clusters: int = 6) -> np.ndarray:
        """
        Classify water bodies using K-means clustering on spectral features
        
        Args:
            image: RGB+NIR image
            water_mask: Binary water mask
            ndwi: NDWI values
            n_clusters: Number of clusters (water body types)
            
        Returns:
            Classified water body mask
        """
        
        # Extract water pixels
        water_coords = np.where(water_mask > 0)
        
        if len(water_coords[0]) < 50:  # Not enough water pixels
            return np.zeros_like(water_mask)
        
        # Create feature vector for each water pixel
        features = []
        
        for y, x in zip(water_coords[0], water_coords[1]):
            # Spectral features
            rgb_features = image[y, x, :3]  # RGB values
            if image.shape[2] > 3:
                nir_feature = image[y, x, 3]  # NIR value
            else:
                nir_feature = 0
            
            ndwi_feature = ndwi[y, x]
            
            # Spatial context features (mean of 3x3 neighborhood)
            y_start, y_end = max(0, y-1), min(image.shape[0], y+2)
            x_start, x_end = max(0, x-1), min(image.shape[1], x+2)
            
            neighborhood_rgb = np.mean(image[y_start:y_end, x_start:x_end, :3], axis=(0, 1))
            neighborhood_ndwi = np.mean(ndwi[y_start:y_end, x_start:x_end])
            
            # Combine features
            pixel_features = np.concatenate([
                rgb_features, [nir_feature], [ndwi_feature],
                neighborhood_rgb, [neighborhood_ndwi]
            ])
            
            features.append(pixel_features)
        
        features = np.array(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), 
                       random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create classified mask
        classified_mask = np.zeros_like(water_mask)
        
        for idx, (y, x) in enumerate(zip(water_coords[0], water_coords[1])):
            classified_mask[y, x] = cluster_labels[idx] + 1  # +1 to avoid background class
        
        return classified_mask
    
    def manual_label_refinement(self, image: np.ndarray, classified_mask: np.ndarray,
                              ndwi: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Semi-automated label refinement using NDWI-based rules
        
        Args:
            image: RGB+NIR image
            classified_mask: K-means classified mask
            ndwi: NDWI values
            
        Returns:
            Refined mask and confidence scores
        """
        
        refined_mask = classified_mask.copy()
        confidence_scores = {}
        
        # Define NDWI-based rules for different water body types
        rules = {
            'swamp': {'ndwi_range': (0.1, 0.4), 'vegetation_nearby': True},
            'river': {'ndwi_range': (0.3, 0.8), 'linear_shape': True},
            'estuary': {'ndwi_range': (0.2, 0.6), 'sediment_presence': True},
            'tidal_pool': {'ndwi_range': (0.1, 0.5), 'small_size': True},
            'shallow_water': {'ndwi_range': (0.0, 0.3), 'large_area': True},
            'flood_plain': {'ndwi_range': (0.2, 0.7), 'irregular_shape': True}
        }
        
        # Analyze each cluster
        unique_clusters = np.unique(classified_mask)
        unique_clusters = unique_clusters[unique_clusters > 0]  # Exclude background
        
        for cluster_id in unique_clusters:
            cluster_mask = classified_mask == cluster_id
            cluster_coords = np.where(cluster_mask)
            
            if len(cluster_coords[0]) == 0:
                continue
            
            # Extract cluster features
            cluster_ndwi = ndwi[cluster_mask]
            mean_ndwi = np.mean(cluster_ndwi)
            
            # Analyze shape properties
            contours, _ = cv2.findContours(cluster_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape metrics
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = self._calculate_aspect_ratio(largest_contour)
            
            # Classify based on rules
            best_class = None
            best_confidence = 0
            
            for water_type, rule in rules.items():
                confidence = self._calculate_rule_confidence(
                    mean_ndwi, area, compactness, aspect_ratio, rule, image, cluster_mask
                )
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_class = water_type
            
            # Update mask with best classification
            if best_class and best_confidence > 0.3:
                refined_mask[cluster_mask] = self.water_classes[best_class]
                confidence_scores[f"cluster_{cluster_id}"] = {
                    'predicted_class': best_class,
                    'confidence': best_confidence,
                    'mean_ndwi': mean_ndwi,
                    'area': area
                }
        
        return refined_mask, confidence_scores
    
    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of contour bounding rectangle"""
        
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if min(width, height) > 0:
            return max(width, height) / min(width, height)
        else:
            return 1.0
    
    def _calculate_rule_confidence(self, mean_ndwi: float, area: float, 
                                 compactness: float, aspect_ratio: float,
                                 rule: Dict, image: np.ndarray, 
                                 cluster_mask: np.ndarray) -> float:
        """Calculate confidence score for rule-based classification"""
        
        confidence = 0.0
        
        # NDWI range check
        ndwi_min, ndwi_max = rule['ndwi_range']
        if ndwi_min <= mean_ndwi <= ndwi_max:
            confidence += 0.4
        else:
            # Penalty for being outside range
            distance = min(abs(mean_ndwi - ndwi_min), abs(mean_ndwi - ndwi_max))
            confidence += max(0, 0.4 - distance * 2)
        
        # Shape-based rules
        if rule.get('linear_shape', False):
            # Rivers should have high aspect ratio
            if aspect_ratio > 3:
                confidence += 0.3
        
        if rule.get('small_size', False):
            # Tidal pools should be small
            if area < 1000:  # pixels
                confidence += 0.2
        
        if rule.get('large_area', False):
            # Shallow water should cover large areas
            if area > 5000:  # pixels
                confidence += 0.2
        
        # Vegetation proximity check
        if rule.get('vegetation_nearby', False):
            vegetation_score = self._check_vegetation_proximity(image, cluster_mask)
            confidence += 0.1 * vegetation_score
        
        return min(1.0, confidence)
    
    def _check_vegetation_proximity(self, image: np.ndarray, 
                                  cluster_mask: np.ndarray) -> float:
        """Check vegetation in proximity to water body"""
        
        # Compute NDVI approximation using RGB
        if image.shape[2] >= 4:  # Has NIR
            red = image[:, :, 2].astype(np.float32)
            nir = image[:, :, 3].astype(np.float32)
            
            denominator = red + nir
            denominator[denominator == 0] = 1e-8
            ndvi = (nir - red) / denominator
        else:
            # Approximate using green-red difference
            green = image[:, :, 1].astype(np.float32)
            red = image[:, :, 2].astype(np.float32)
            
            denominator = green + red
            denominator[denominator == 0] = 1e-8
            ndvi = (green - red) / denominator
        
        # Dilate cluster mask to check surrounding area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        expanded_mask = cv2.dilate(cluster_mask.astype(np.uint8), kernel)
        
        # Calculate vegetation score in surrounding area
        surrounding_area = expanded_mask - cluster_mask.astype(np.uint8)
        
        if np.sum(surrounding_area) > 0:
            vegetation_score = np.mean(ndvi[surrounding_area > 0])
            return max(0, min(1, vegetation_score + 0.5))  # Normalize to [0, 1]
        else:
            return 0.0
    
    def __len__(self) -> int:
        return len(self.metadata)


class DataPreprocessor:
    """
    Handles data preprocessing pipeline including NDWI computation and normalization
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        
        # Normalization statistics (computed from Sentinel-2 data)
        self.sentinel2_stats = {
            'mean': [0.485, 0.456, 0.406, 0.5],  # RGB + NIR
            'std': [0.229, 0.224, 0.225, 0.25]   # RGB + NIR
        }
        
        # NDWI normalization (range: -1 to 1)
        self.ndwi_stats = {'mean': 0.0, 'std': 0.5}
    
    def preprocess_sentinel2_patch(self, patch_data: Dict) -> Dict[str, np.ndarray]:
        """
        Preprocess Sentinel-2 patch with temporal frames
        
        Args:
            patch_data: Dictionary containing patch information
            
        Returns:
            Preprocessed data dictionary
        """
        
        processed_frames = []
        
        for frame_info in patch_data['temporal_sequence']:
            # Load GeoTIFF
            with rasterio.open(frame_info['file_path']) as src:
                # Read RGB + NIR bands
                bands = src.read([1, 2, 3, 4])  # B, G, R, NIR
                bands = np.transpose(bands, (1, 2, 0))  # (H, W, C)
                
                # Reorder to RGB + NIR
                rgb_nir = bands[:, :, [2, 1, 0, 3]]  # R, G, B, NIR
                
                # Resize to target size
                rgb_nir = cv2.resize(rgb_nir, self.target_size, interpolation=cv2.INTER_CUBIC)
                
                # Compute NDWI
                ndwi = self.compute_ndwi(rgb_nir, green_idx=1, nir_idx=3)
                
                # Stack RGB + NIR + NDWI
                full_frame = np.concatenate([
                    rgb_nir, ndwi[:, :, np.newaxis]
                ], axis=2)
                
                # Normalize
                normalized_frame = self.normalize_frame(full_frame)
                
                processed_frames.append(normalized_frame)
        
        return {
            'temporal_frames': np.array(processed_frames),  # (T, H, W, C)
            'patch_id': patch_data['patch_id'],
            'region': patch_data['region']
        }
    
    def compute_ndwi(self, image: np.ndarray, green_idx: int = 1, 
                    nir_idx: int = 3) -> np.ndarray:
        """Compute NDWI from RGB+NIR image"""
        
        green = image[:, :, green_idx].astype(np.float32)
        nir = image[:, :, nir_idx].astype(np.float32)
        
        # Avoid division by zero
        denominator = green + nir
        denominator[denominator == 0] = 1e-8
        
        ndwi = (green - nir) / denominator
        return np.clip(ndwi, -1, 1)
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame using Sentinel-2 statistics
        
        Args:
            frame: Frame with shape (H, W, 5) - RGB+NIR+NDWI
            
        Returns:
            Normalized frame
        """
        
        normalized = frame.copy().astype(np.float32)
        
        # Normalize RGB+NIR channels (0-4)
        for c in range(4):
            if c < len(self.sentinel2_stats['mean']):
                normalized[:, :, c] = (normalized[:, :, c] / 255.0 - self.sentinel2_stats['mean'][c]) / self.sentinel2_stats['std'][c]
        
        # Normalize NDWI channel (already in [-1, 1])
        if frame.shape[2] > 4:
            normalized[:, :, 4] = (normalized[:, :, 4] - self.ndwi_stats['mean']) / self.ndwi_stats['std']
        
        return normalized
    
    def create_augmentation_pipeline(self, mode: str = 'train') -> A.Compose:
        """
        Create augmentation pipeline for training/validation
        
        Args:
            mode: 'train' or 'val'
            
        Returns:
            Albumentations composition
        """
        
        if mode == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.2
                ),
                A.GaussianBlur(blur_limit=(1, 3), p=0.15),
                A.GaussNoise(var_limit=(5, 25), p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.1),
            ])
        else:
            # Minimal augmentation for validation
            return A.Compose([
                A.RandomRotate90(p=0.25),
                A.Flip(p=0.25),
            ])


def create_data_loaders(data_path: str, batch_size: int = 4, 
                       num_workers: int = 2) -> Dict[str, DataLoader]:
    """
    Create PyTorch data loaders for training, validation, and testing
    
    Args:
        data_path: Path to processed data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Dictionary of data loaders
    """
    
    preprocessor = DataPreprocessor()
    
    # Create augmentation pipelines
    train_transform = preprocessor.create_augmentation_pipeline('train')
    val_transform = preprocessor.create_augmentation_pipeline('val')
    
    # Create datasets
    train_dataset = WaterBodyDataset(
        data_path, split='train', transform=train_transform
    )
    
    val_dataset = WaterBodyDataset(
        data_path, split='val', transform=val_transform
    )
    
    test_dataset = WaterBodyDataset(
        data_path, split='test', transform=None
    )
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    }
    
    print(f"Created data loaders:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    
    return data_loaders


def visualize_preprocessing_pipeline(data_path: str):
    """Visualize the preprocessing pipeline results"""
    
    # Load sample data
    dataset = WaterBodyDataset(data_path, split='train')
    
    if len(dataset) == 0:
        print("No data found for visualization")
        return
    
    # Get sample
    sample = dataset[0]
    images = sample['images']  # (T, C, H, W)
    mask = sample['mask']      # (H, W)
    
    # Convert to numpy for visualization
    if len(images.shape) == 4:  # Temporal sequence
        # Show first frame
        image = images[0].permute(1, 2, 0).numpy()  # (H, W, C)
    else:
        image = images.permute(1, 2, 0).numpy()
    
    # Denormalize for visualization
    image = np.clip(image * 0.5 + 0.5, 0, 1)  # Simple denormalization
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # RGB image
    if image.shape[2] >= 3:
        axes[0].imshow(image[:, :, :3])
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
    
    # NDWI channel (if available)
    if image.shape[2] > 4:
        ndwi = image[:, :, 4]
        axes[1].imshow(ndwi, cmap='RdYlBu')
        axes[1].set_title('NDWI')
        axes[1].axis('off')
    
    # Ground truth mask
    axes[2].imshow(mask.numpy(), cmap='tab10')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Overlay
    overlay = image[:, :, :3].copy()
    mask_np = mask.numpy()
    for class_id in range(1, 6):
        class_mask = mask_np == class_id
        if class_mask.any():
            overlay[class_mask] = overlay[class_mask] * 0.7 + np.array([1, 0, 0]) * 0.3
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/WaterBodyResearch/preprocessing_sample.png', 
               dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Test preprocessing pipeline
    preprocessor = DataPreprocessor()
    
    print("Data preprocessing pipeline initialized")
    print("Target size:", preprocessor.target_size)
    print("Normalization stats:", preprocessor.sentinel2_stats)
    
    # Test NDWI computation
    test_image = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
    ndwi = preprocessor.compute_ndwi(test_image)
    
    print(f"NDWI computation test - Range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")
    
    # Test data loader creation (will work when data is available)
    try:
        data_loaders = create_data_loaders("/content/drive/MyDrive/WaterBodyResearch/data/processed")
        print("Data loaders created successfully")
    except Exception as e:
        print(f"Data loaders creation failed: {e}")
        print("This is expected if no processed data is available yet")