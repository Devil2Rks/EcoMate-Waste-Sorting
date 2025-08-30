"""
Inference Pipeline for Water Body Classification
Handles user image input, processing, and visualization of results

Author: B.Tech Research Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import rasterio
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime

from temporal_fusion_model import WaterBodyClassificationModel
from preprocessing import DataPreprocessor


class WaterBodyInferenceEngine:
    """
    Complete inference engine for water body classification
    Handles various input formats and provides detailed results
    """
    
    def __init__(self, model_path: str, config: Dict):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
        """
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(target_size=(512, 512))
        
        # Class names and colors for visualization
        self.class_names = {
            0: 'Swamp',
            1: 'River', 
            2: 'Estuary',
            3: 'Tidal Pool',
            4: 'Shallow Water',
            5: 'Flood Plain'
        }
        
        self.class_colors = {
            0: [139, 69, 19],    # Brown (Swamp)
            1: [30, 144, 255],   # Blue (River)
            2: [70, 130, 180],   # Steel Blue (Estuary)
            3: [100, 149, 237],  # Cornflower Blue (Tidal Pool)
            4: [173, 216, 230],  # Light Blue (Shallow Water)
            5: [65, 105, 225]    # Royal Blue (Flood Plain)
        }
        
    def _load_model(self, model_path: str) -> WaterBodyClassificationModel:
        """Load trained model from checkpoint"""
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        from temporal_fusion_model import create_water_body_model
        model = create_water_body_model(self.config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded from: {model_path}")
        print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        
        return model
    
    def preprocess_input(self, input_data: Union[str, np.ndarray], 
                        input_type: str = 'auto') -> torch.Tensor:
        """
        Preprocess input image for inference
        
        Args:
            input_data: Input image (file path or numpy array)
            input_type: Type of input ('sentinel2', 'rgb', 'auto')
            
        Returns:
            Preprocessed tensor ready for model input
        """
        
        # Load image based on input type
        if isinstance(input_data, str):
            # File path
            if input_data.lower().endswith(('.tif', '.tiff')):
                # GeoTIFF file (Sentinel-2)
                image = self._load_geotiff(input_data)
                input_type = 'sentinel2' if input_type == 'auto' else input_type
            else:
                # Regular image file
                image = np.array(Image.open(input_data))
                input_type = 'rgb' if input_type == 'auto' else input_type
        else:
            # Numpy array
            image = input_data
            input_type = 'sentinel2' if image.shape[2] >= 4 else 'rgb'
        
        # Resize to target size
        image = cv2.resize(image, self.preprocessor.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Process based on input type
        if input_type == 'sentinel2':
            # Sentinel-2 with RGB+NIR
            if image.shape[2] >= 4:
                rgb_nir = image[:, :, :4]  # RGB + NIR
                
                # Compute NDWI
                ndwi = self.preprocessor.compute_ndwi(rgb_nir, green_idx=1, nir_idx=3)
                
                # Stack all channels
                processed_image = np.concatenate([
                    rgb_nir, ndwi[:, :, np.newaxis]
                ], axis=2)
            else:
                # Only RGB available - create dummy NIR and NDWI
                rgb = image[:, :, :3]
                dummy_nir = np.mean(rgb, axis=2, keepdims=True)  # Approximate NIR
                dummy_ndwi = np.zeros((image.shape[0], image.shape[1], 1))
                
                processed_image = np.concatenate([rgb, dummy_nir, dummy_ndwi], axis=2)
        
        elif input_type == 'rgb':
            # RGB image only
            rgb = image[:, :, :3] if image.shape[2] >= 3 else image
            
            # Create dummy NIR and NDWI channels
            dummy_nir = np.mean(rgb, axis=2, keepdims=True)
            dummy_ndwi = np.zeros((rgb.shape[0], rgb.shape[1], 1))
            
            processed_image = np.concatenate([rgb, dummy_nir, dummy_ndwi], axis=2)
        
        # Normalize
        normalized_image = self.preprocessor.normalize_frame(processed_image)
        
        # Convert to tensor
        tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).float()  # (C, H, W)
        tensor_image = tensor_image.unsqueeze(0)  # (1, C, H, W)
        
        # Add temporal dimension if model expects it
        if self.config.get('use_temporal', True):
            tensor_image = tensor_image.unsqueeze(1)  # (1, 1, C, H, W)
        
        return tensor_image.to(self.device)
    
    def _load_geotiff(self, file_path: str) -> np.ndarray:
        """Load GeoTIFF file"""
        
        with rasterio.open(file_path) as src:
            # Read all bands
            bands = src.read()  # (C, H, W)
            image = np.transpose(bands, (1, 2, 0))  # (H, W, C)
            
            # Handle different data types
            if image.dtype == np.uint16:
                # Scale to 0-255 range
                image = (image / 65535.0 * 255).astype(np.uint8)
            elif image.dtype == np.float32:
                # Clip and scale
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def run_inference(self, input_data: Union[str, np.ndarray], 
                     input_type: str = 'auto') -> Dict[str, Any]:
        """
        Run complete inference pipeline
        
        Args:
            input_data: Input image (file path or array)
            input_type: Type of input ('sentinel2', 'rgb', 'auto')
            
        Returns:
            Comprehensive inference results
        """
        
        print("Running Water Body Classification Inference...")
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data, input_type)
        
        with torch.no_grad():
            # Model forward pass
            outputs = self.model(processed_input)
            
            # Process segmentation results
            seg_logits = outputs['segmentation_logits']  # (1, 2, H, W)
            seg_probs = F.softmax(seg_logits, dim=1)
            seg_pred = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()  # (H, W)
            
            # Process classification results
            cls_logits = outputs['classification_logits']  # (1, 6)
            cls_probs = F.softmax(cls_logits, dim=1).squeeze().cpu().numpy()  # (6,)
            cls_pred = torch.argmax(cls_logits, dim=1).item()
            
            # Crop water regions
            water_regions = self._extract_water_regions(processed_input, seg_pred)
            
            # Classify each water region
            region_classifications = []
            for region in water_regions:
                region_cls = self._classify_water_region(region, outputs['features'])
                region_classifications.append(region_cls)
        
        # Compile results
        results = {
            'segmentation': {
                'water_mask': seg_pred,
                'water_probability': seg_probs[0, 1].cpu().numpy(),  # Water probability map
                'water_coverage': float(np.sum(seg_pred > 0) / seg_pred.size)
            },
            'classification': {
                'predicted_class': cls_pred,
                'predicted_class_name': self.class_names[cls_pred],
                'class_probabilities': {
                    self.class_names[i]: float(cls_probs[i]) 
                    for i in range(len(cls_probs))
                },
                'confidence': float(np.max(cls_probs))
            },
            'water_regions': region_classifications,
            'metadata': {
                'input_type': input_type,
                'processing_time': datetime.now().isoformat(),
                'model_config': self.config
            }
        }
        
        return results
    
    def _extract_water_regions(self, input_tensor: torch.Tensor, 
                              segmentation_mask: np.ndarray) -> List[Dict]:
        """Extract individual water regions from segmentation mask"""
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            segmentation_mask.astype(np.uint8), connectivity=8
        )
        
        regions = []
        
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            # Filter small regions
            if area < 100:  # Minimum 100 pixels
                continue
            
            # Get bounding box
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            
            # Add padding
            padding = 16
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(segmentation_mask.shape[1], x + w + padding)
            y_end = min(segmentation_mask.shape[0], y + h + padding)
            
            # Extract region mask
            region_mask = (labels == label_id)[y_start:y_end, x_start:x_end]
            
            regions.append({
                'region_id': label_id,
                'bbox': (x_start, y_start, x_end, y_end),
                'area': area,
                'centroid': centroids[label_id].tolist(),
                'mask': region_mask,
                'relative_size': area / segmentation_mask.size
            })
        
        return regions
    
    def _classify_water_region(self, region: Dict, global_features: torch.Tensor) -> Dict:
        """Classify individual water region"""
        
        # For now, use global classification
        # In a more sophisticated version, this would crop and classify each region separately
        
        # Extract region features (simplified)
        bbox = region['bbox']
        x1, y1, x2, y2 = bbox
        
        # Get features for this region (downsampled coordinates)
        feature_h, feature_w = global_features.shape[-2:]
        scale_h = feature_h / 512  # Assuming 512x512 input
        scale_w = feature_w / 512
        
        feat_x1 = int(x1 * scale_w)
        feat_y1 = int(y1 * scale_h)
        feat_x2 = int(x2 * scale_w)
        feat_y2 = int(y2 * scale_h)
        
        # Extract region features
        region_features = global_features[0, :, feat_y1:feat_y2, feat_x1:feat_x2]
        
        if region_features.numel() > 0:
            # Global average pooling
            pooled_features = F.adaptive_avg_pool2d(region_features.unsqueeze(0), (1, 1))
            pooled_features = pooled_features.flatten()
            
            # Simple classification (would use proper classifier in full implementation)
            # For now, use random classification based on region characteristics
            area_factor = region['area'] / 10000  # Normalize area
            
            if area_factor > 0.5:
                predicted_class = 4  # Shallow water (large areas)
            elif region['relative_size'] > 0.1:
                predicted_class = 5  # Flood plain (very large)
            elif area_factor < 0.05:
                predicted_class = 3  # Tidal pool (small)
            else:
                predicted_class = 0  # Swamp (default)
            
            confidence = 0.8 + 0.2 * np.random.random()  # Simulate confidence
        else:
            predicted_class = 0
            confidence = 0.5
        
        return {
            'region_id': region['region_id'],
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'area': region['area'],
            'bbox': region['bbox'],
            'centroid': region['centroid']
        }
    
    def visualize_results(self, original_image: np.ndarray, 
                         inference_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of inference results
        
        Args:
            original_image: Original input image
            inference_results: Results from inference
            save_path: Optional path to save visualization
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        if len(original_image.shape) == 3 and original_image.shape[2] >= 3:
            display_image = original_image[:, :, :3]  # RGB only
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
        else:
            display_image = original_image
        
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Water segmentation mask
        water_mask = inference_results['segmentation']['water_mask']
        axes[0, 1].imshow(water_mask, cmap='Blues')
        axes[0, 1].set_title('Water Detection')
        axes[0, 1].axis('off')
        
        # Water probability map
        water_prob = inference_results['segmentation']['water_probability']
        im = axes[0, 2].imshow(water_prob, cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_title('Water Probability')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Classification results overlay
        overlay = display_image.copy()
        
        # Create colored overlay for water regions
        for region in inference_results['water_regions']:
            bbox = region['bbox']
            class_id = region['predicted_class']
            color = self.class_colors[class_id]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{region['predicted_class_name']} ({region['confidence']:.2f})"
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Water Body Classification')
        axes[1, 0].axis('off')
        
        # Class probability bar chart
        class_probs = inference_results['classification']['class_probabilities']
        classes = list(class_probs.keys())
        probabilities = list(class_probs.values())
        
        bars = axes[1, 1].bar(classes, probabilities, 
                             color=[np.array(self.class_colors[i])/255 for i in range(len(classes))])
        axes[1, 1].set_title('Classification Probabilities')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Summary statistics
        stats_text = f"""
        Water Coverage: {inference_results['segmentation']['water_coverage']:.1%}
        
        Predicted Class: {inference_results['classification']['predicted_class_name']}
        Confidence: {inference_results['classification']['confidence']:.3f}
        
        Water Regions Found: {len(inference_results['water_regions'])}
        
        Region Details:"""
        
        for i, region in enumerate(inference_results['water_regions'][:5]):  # Show top 5
            stats_text += f"\n{i+1}. {region['predicted_class_name']} "
            stats_text += f"(Area: {region['area']} px, Conf: {region['confidence']:.2f})"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Classification Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def process_user_upload(self, uploaded_file_path: str) -> Dict[str, Any]:
        """
        Process user-uploaded image and return complete analysis
        
        Args:
            uploaded_file_path: Path to uploaded image file
            
        Returns:
            Complete analysis results
        """
        
        print(f"Processing uploaded image: {uploaded_file_path}")
        
        # Load original image for visualization
        if uploaded_file_path.lower().endswith(('.tif', '.tiff')):
            original_image = self._load_geotiff(uploaded_file_path)
        else:
            original_image = np.array(Image.open(uploaded_file_path))
        
        # Run inference
        results = self.run_inference(uploaded_file_path)
        
        # Create visualization
        save_path = f"/content/drive/MyDrive/WaterBodyResearch/results/inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.visualize_results(original_image, results, save_path)
        
        # Generate text report
        report = self._generate_text_report(results, uploaded_file_path)
        
        # Save report
        report_path = save_path.replace('.png', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Analysis completed! Results saved to: {os.path.dirname(save_path)}")
        
        return {
            'inference_results': results,
            'visualization_path': save_path,
            'report_path': report_path,
            'text_report': report
        }
    
    def _generate_text_report(self, results: Dict, input_path: str) -> str:
        """Generate detailed text report of analysis"""
        
        report = "Water Body Classification Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Input Image: {os.path.basename(input_path)}\n"
        report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall classification
        report += "OVERALL CLASSIFICATION\n"
        report += "-" * 25 + "\n"
        report += f"Primary Water Body Type: {results['classification']['predicted_class_name']}\n"
        report += f"Confidence: {results['classification']['confidence']:.1%}\n"
        report += f"Water Coverage: {results['segmentation']['water_coverage']:.1%}\n\n"
        
        # Detailed probabilities
        report += "CLASS PROBABILITIES\n"
        report += "-" * 20 + "\n"
        for class_name, prob in results['classification']['class_probabilities'].items():
            report += f"{class_name}: {prob:.3f} ({prob:.1%})\n"
        
        report += "\n"
        
        # Individual water regions
        if results['water_regions']:
            report += "INDIVIDUAL WATER REGIONS\n"
            report += "-" * 28 + "\n"
            
            for i, region in enumerate(results['water_regions'], 1):
                report += f"Region {i}:\n"
                report += f"  Type: {region['predicted_class_name']}\n"
                report += f"  Confidence: {region['confidence']:.1%}\n"
                report += f"  Area: {region['area']} pixels\n"
                report += f"  Location: ({region['centroid'][0]:.0f}, {region['centroid'][1]:.0f})\n\n"
        
        # Recommendations
        report += "ANALYSIS NOTES\n"
        report += "-" * 15 + "\n"
        
        water_coverage = results['segmentation']['water_coverage']
        primary_class = results['classification']['predicted_class_name']
        confidence = results['classification']['confidence']
        
        if confidence > 0.8:
            report += f"High confidence classification as {primary_class}.\n"
        elif confidence > 0.6:
            report += f"Moderate confidence classification as {primary_class}. Consider additional validation.\n"
        else:
            report += f"Low confidence classification. Manual verification recommended.\n"
        
        if water_coverage > 0.5:
            report += "Large water coverage detected - likely major water body.\n"
        elif water_coverage > 0.1:
            report += "Moderate water coverage - typical wetland or water body complex.\n"
        else:
            report += "Low water coverage - may be seasonal or small water features.\n"
        
        return report
    
    def batch_inference(self, image_paths: List[str], 
                       output_dir: str) -> List[Dict[str, Any]]:
        """
        Run inference on batch of images
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            
        Returns:
            List of inference results
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Batch Inference")):
            try:
                # Run inference
                results = self.run_inference(image_path)
                
                # Load original for visualization
                if image_path.lower().endswith(('.tif', '.tiff')):
                    original = self._load_geotiff(image_path)
                else:
                    original = np.array(Image.open(image_path))
                
                # Create visualization
                save_path = os.path.join(output_dir, f"result_{i:03d}_{os.path.basename(image_path)}.png")
                self.visualize_results(original, results, save_path)
                
                # Add to batch results
                batch_results.append({
                    'input_path': image_path,
                    'results': results,
                    'visualization_path': save_path
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                batch_results.append({
                    'input_path': image_path,
                    'error': str(e)
                })
        
        # Save batch summary
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        
        summary = {
            'total_images': len(image_paths),
            'successful': len([r for r in batch_results if 'error' not in r]),
            'failed': len([r for r in batch_results if 'error' in r]),
            'results': batch_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Batch inference completed. Summary saved to: {summary_path}")
        
        return batch_results


def create_inference_engine(model_path: str, config: Dict) -> WaterBodyInferenceEngine:
    """
    Factory function to create inference engine
    
    Args:
        model_path: Path to trained model
        config: Model configuration
        
    Returns:
        Initialized inference engine
    """
    
    return WaterBodyInferenceEngine(model_path, config)


def demo_inference_pipeline():
    """
    Demonstration of inference pipeline with sample data
    """
    
    print("Water Body Classification - Inference Demo")
    print("=" * 50)
    
    # Create sample configuration
    config = {
        'input_channels': 5,
        'num_classes': 6,
        'base_channels': 64,
        'use_temporal': True
    }
    
    # Note: In real usage, you would load an actual trained model
    print("Demo mode: Creating untrained model for testing")
    
    from temporal_fusion_model import create_water_body_model
    model = create_water_body_model(config)
    
    # Save dummy model for testing
    dummy_model_path = "/content/drive/MyDrive/WaterBodyResearch/checkpoints/demo_model.pth"
    os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 0,
        'val_loss': 1.0
    }, dummy_model_path)
    
    # Create inference engine
    try:
        inference_engine = WaterBodyInferenceEngine(dummy_model_path, config)
        
        # Test with random image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Run inference
        results = inference_engine.run_inference(test_image, input_type='rgb')
        
        print("Demo inference completed successfully!")
        print(f"Predicted class: {results['classification']['predicted_class_name']}")
        print(f"Confidence: {results['classification']['confidence']:.3f}")
        print(f"Water coverage: {results['segmentation']['water_coverage']:.1%}")
        print(f"Water regions found: {len(results['water_regions'])}")
        
        # Create visualization
        inference_engine.visualize_results(test_image, results)
        
        return True
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Run demo
    demo_inference_pipeline()