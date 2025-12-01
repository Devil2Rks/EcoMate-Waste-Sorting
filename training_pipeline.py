"""
Training Pipeline for Water Body Classification with Temporal Fusion
Implements hierarchical loss functions and training strategies

Author: B.Tech Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from temporal_fusion_model import WaterBodyClassificationModel, create_water_body_model
from preprocessing import WaterBodyDataset, create_data_loaders


class HierarchicalLoss(nn.Module):
    """
    Custom hierarchical loss function for water body classification
    """
    
    def __init__(self, num_classes: int = 6, hierarchy_weight: float = 0.2,
                 class_weights: Optional[torch.Tensor] = None):
        super(HierarchicalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.hierarchy_weight = hierarchy_weight
        
        # Class weights for imbalanced dataset
        if class_weights is None:
            # Default weights (can be computed from dataset statistics)
            class_weights = torch.ones(num_classes)
            class_weights[0] = 1.2  # Swamp (primary focus)
            class_weights[1] = 1.0  # River
            class_weights[2] = 1.1  # Estuary
            class_weights[3] = 1.3  # Tidal pool (rare)
            class_weights[4] = 1.0  # Shallow water
            class_weights[5] = 1.1  # Flood plain
        
        self.register_buffer('class_weights', class_weights)
        
        # Hierarchical confusion matrix (penalty for distant confusions)
        self.register_buffer('hierarchy_matrix', self._build_hierarchy_penalties())
        
    def _build_hierarchy_penalties(self) -> torch.Tensor:
        """Build penalty matrix for hierarchical confusions"""
        
        # Penalty matrix: higher values = higher penalty for confusion
        penalties = torch.ones(self.num_classes, self.num_classes)
        
        # Classes: swamp(0), river(1), estuary(2), tidal_pool(3), shallow_water(4), flood_plain(5)
        
        # Low penalties within similar groups
        # Coastal group: estuary, tidal_pool
        penalties[2, 3] = 0.5  # estuary-tidal_pool
        penalties[3, 2] = 0.5
        
        # Wetland-related: swamp, shallow_water
        penalties[0, 4] = 0.6  # swamp-shallow_water
        penalties[4, 0] = 0.6
        
        # Water flow related: river, flood_plain
        penalties[1, 5] = 0.7  # river-flood_plain
        penalties[5, 1] = 0.7
        
        # High penalties for very different types
        penalties[0, 1] = 1.5  # swamp-river
        penalties[1, 0] = 1.5
        penalties[2, 5] = 1.4  # estuary-flood_plain
        penalties[5, 2] = 1.4
        
        return penalties
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss
        
        Args:
            predictions: Model predictions (B, num_classes)
            targets: Ground truth targets (B,)
            
        Returns:
            Dictionary with loss components
        """
        
        # Standard weighted cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, weight=self.class_weights)
        
        # Hierarchical penalty loss
        pred_probs = F.softmax(predictions, dim=1)
        
        hierarchy_loss = 0.0
        batch_size = targets.shape[0]
        
        for i in range(batch_size):
            target_class = targets[i].item()
            
            for j in range(self.num_classes):
                if j != target_class:
                    penalty = self.hierarchy_matrix[target_class, j]
                    hierarchy_loss += pred_probs[i, j] * penalty
        
        hierarchy_loss = hierarchy_loss / batch_size
        
        # Total loss
        total_loss = ce_loss + self.hierarchy_weight * hierarchy_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'hierarchy_loss': hierarchy_loss
        }


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss
        
        Args:
            predictions: Predicted segmentation logits (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            Dice loss value
        """
        
        # Convert predictions to probabilities
        pred_probs = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        dice_scores = []
        
        for c in range(predictions.shape[1]):
            pred_c = pred_probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = torch.sum(pred_c * target_c, dim=(1, 2))
            union = torch.sum(pred_c, dim=(1, 2)) + torch.sum(target_c, dim=(1, 2))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice score across classes and batch
        dice_scores = torch.stack(dice_scores, dim=1)  # (B, C)
        avg_dice = torch.mean(dice_scores)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - avg_dice


class WaterBodyTrainer:
    """
    Training pipeline for water body classification model
    """
    
    def __init__(self, model: WaterBodyClassificationModel, 
                 data_loaders: Dict[str, DataLoader],
                 config: Dict):
        self.model = model
        self.data_loaders = data_loaders
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        self.classification_loss = HierarchicalLoss(
            num_classes=config.get('num_classes', 6),
            hierarchy_weight=config.get('hierarchy_weight', 0.2)
        ).to(self.device)
        
        self.segmentation_loss = DiceLoss().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging and visualization"""
        
        # Create log directory
        log_dir = f"/content/drive/MyDrive/WaterBodyResearch/logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Initialize Weights & Biases (optional)
        if self.config.get('use_wandb', False):
            wandb.init(
                project="water-body-classification",
                config=self.config,
                name=f"temporal_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.data_loaders['train'], desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(self.device)  # (B, T, C, H, W) or (B, C, H, W)
            masks = batch['mask'].to(self.device)     # (B, H, W)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute losses
            # 1. Segmentation loss (binary water detection)
            seg_targets = (masks > 0).long()  # Convert to binary
            seg_loss = (self.segmentation_loss(outputs['segmentation_logits'], seg_targets) + 
                       self.ce_loss(outputs['segmentation_logits'], seg_targets)) * 0.5
            
            # 2. Classification loss (water body types)
            # Only compute for water pixels
            water_pixels = masks > 0
            if water_pixels.any():
                # Get water body class labels
                water_labels = masks[water_pixels]
                
                # Get corresponding classification predictions
                # Use global average pooling of features for classification
                cls_features = F.adaptive_avg_pool2d(outputs['features'], (1, 1)).flatten(1)
                cls_predictions = self.model.hierarchical_classifier.hierarchical_layers['level2'](cls_features)
                
                # Compute classification loss for each sample
                cls_loss_components = []
                for b in range(images.shape[0]):
                    batch_water_mask = water_pixels[b]
                    if batch_water_mask.any():
                        # Get most frequent class in this sample as target
                        sample_labels = masks[b][batch_water_mask]
                        target_class = torch.mode(sample_labels).values
                        
                        cls_loss_components.append(
                            self.classification_loss(
                                cls_predictions[b:b+1], 
                                target_class.unsqueeze(0)
                            )['total_loss']
                        )
                
                if cls_loss_components:
                    cls_loss = torch.mean(torch.stack(cls_loss_components))
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)
            else:
                cls_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_batch_loss = seg_loss + cls_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update tracking
            total_loss += total_batch_loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Seg': f'{seg_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}'
            })
            
            # Log to TensorBoard
            global_step = epoch * len(self.data_loaders['train']) + batch_idx
            self.writer.add_scalar('Loss/Train/Total', total_batch_loss.item(), global_step)
            self.writer.add_scalar('Loss/Train/Segmentation', seg_loss.item(), global_step)
            self.writer.add_scalar('Loss/Train/Classification', cls_loss.item(), global_step)
        
        # Calculate epoch averages
        avg_total_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        
        return {
            'total_loss': avg_total_loss,
            'segmentation_loss': avg_seg_loss,
            'classification_loss': avg_cls_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        # Metrics tracking
        all_seg_predictions = []
        all_seg_targets = []
        all_cls_predictions = []
        all_cls_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc='Validation'):
                # Move data to device
                images = batch['images'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses (same as training)
                seg_targets = (masks > 0).long()
                seg_loss = (self.segmentation_loss(outputs['segmentation_logits'], seg_targets) + 
                           self.ce_loss(outputs['segmentation_logits'], seg_targets)) * 0.5
                
                # Classification loss
                water_pixels = masks > 0
                if water_pixels.any():
                    cls_features = F.adaptive_avg_pool2d(outputs['features'], (1, 1)).flatten(1)
                    cls_predictions = self.model.hierarchical_classifier.hierarchical_layers['level2'](cls_features)
                    
                    cls_loss_components = []
                    for b in range(images.shape[0]):
                        batch_water_mask = water_pixels[b]
                        if batch_water_mask.any():
                            sample_labels = masks[b][batch_water_mask]
                            target_class = torch.mode(sample_labels).values
                            
                            cls_loss_components.append(
                                self.classification_loss(
                                    cls_predictions[b:b+1], 
                                    target_class.unsqueeze(0)
                                )['total_loss']
                            )
                            
                            # Store for metrics
                            all_cls_predictions.append(torch.argmax(cls_predictions[b]).cpu())
                            all_cls_targets.append(target_class.cpu())
                    
                    if cls_loss_components:
                        cls_loss = torch.mean(torch.stack(cls_loss_components))
                    else:
                        cls_loss = torch.tensor(0.0, device=self.device)
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)
                
                total_batch_loss = seg_loss + cls_loss
                
                # Update tracking
                total_loss += total_batch_loss.item()
                total_seg_loss += seg_loss.item() 
                total_cls_loss += cls_loss.item()
                num_batches += 1
                
                # Store predictions for metrics
                seg_pred = torch.argmax(outputs['segmentation_logits'], dim=1)
                all_seg_predictions.extend(seg_pred.cpu().numpy().flatten())
                all_seg_targets.extend(seg_targets.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_total_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        
        # Segmentation metrics
        seg_accuracy = np.mean(np.array(all_seg_predictions) == np.array(all_seg_targets))
        
        # Classification metrics (if available)
        if all_cls_predictions and all_cls_targets:
            cls_accuracy = np.mean(np.array(all_cls_predictions) == np.array(all_cls_targets))
        else:
            cls_accuracy = 0.0
        
        # Log validation metrics
        self.writer.add_scalar('Loss/Val/Total', avg_total_loss, epoch)
        self.writer.add_scalar('Loss/Val/Segmentation', avg_seg_loss, epoch)
        self.writer.add_scalar('Loss/Val/Classification', avg_cls_loss, epoch)
        self.writer.add_scalar('Metrics/Val/Segmentation_Accuracy', seg_accuracy, epoch)
        self.writer.add_scalar('Metrics/Val/Classification_Accuracy', cls_accuracy, epoch)
        
        return {
            'total_loss': avg_total_loss,
            'segmentation_loss': avg_seg_loss,
            'classification_loss': avg_cls_loss,
            'segmentation_accuracy': seg_accuracy,
            'classification_accuracy': cls_accuracy
        }
    
    def train(self, num_epochs: int = 30):
        """
        Complete training pipeline
        
        Args:
            num_epochs: Number of training epochs
        """
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Track losses
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = f"/content/drive/MyDrive/WaterBodyResearch/checkpoints/best_model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.best_val_loss,
                    'config': self.config
                }, checkpoint_path)
                
                print(f"✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Val Seg Acc: {val_metrics['segmentation_accuracy']:.4f}")
            print(f"Val Cls Acc: {val_metrics['classification_accuracy']:.4f}")
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_seg_acc': val_metrics['segmentation_accuracy'],
                    'val_cls_acc': val_metrics['classification_accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping check
            if self.config.get('early_stopping', False):
                patience = self.config.get('early_stopping_patience', 10)
                if epoch - self._get_best_epoch() > patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✓ Best model loaded for final evaluation")
        
        # Save training curves
        self.plot_training_curves()
        
        print("Training completed!")
    
    def _get_best_epoch(self) -> int:
        """Get epoch with best validation loss"""
        if not self.val_losses:
            return 0
        return np.argmin(self.val_losses)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(epochs[-len(lr_history):], lr_history, 'g-')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/WaterBodyResearch/training_curves.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()


class PretrainingSyntheticTrainer:
    """
    Specialized trainer for pre-training on synthetic data
    """
    
    def __init__(self, model: WaterBodyClassificationModel, 
                 synthetic_data_loader: DataLoader, config: Dict):
        self.model = model
        self.data_loader = synthetic_data_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer for pre-training (higher learning rate)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('pretrain_lr', 2e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Loss functions
        self.classification_loss = HierarchicalLoss(
            num_classes=config.get('num_classes', 6)
        ).to(self.device)
        
        self.segmentation_loss = DiceLoss().to(self.device)
        
    def pretrain(self, num_epochs: int = 20):
        """
        Pre-train model on synthetic data
        
        Args:
            num_epochs: Number of pre-training epochs
        """
        
        print(f"Starting pre-training on synthetic data for {num_epochs} epochs")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.data_loader, desc=f'Pretrain Epoch {epoch+1}')
            
            for batch in progress_bar:
                # Move data to device
                images = batch['images'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                seg_targets = (masks > 0).long()
                seg_loss = self.segmentation_loss(outputs['segmentation_logits'], seg_targets)
                
                # Classification loss
                cls_features = F.adaptive_avg_pool2d(outputs['features'], (1, 1)).flatten(1)
                cls_predictions = self.model.hierarchical_classifier.hierarchical_layers['level2'](cls_features)
                
                # Use mask values as classification targets
                cls_targets = []
                for b in range(images.shape[0]):
                    water_mask = masks[b] > 0
                    if water_mask.any():
                        target_class = torch.mode(masks[b][water_mask]).values
                    else:
                        target_class = torch.tensor(0)  # Default to first class
                    cls_targets.append(target_class)
                
                cls_targets = torch.stack(cls_targets).to(self.device)
                cls_loss = self.classification_loss(cls_predictions, cls_targets)['total_loss']
                
                # Total loss
                total_batch_loss = seg_loss + cls_loss
                
                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()
                
                # Update tracking
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'Loss': f'{total_batch_loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Pretrain Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save pre-trained model
        pretrain_path = "/content/drive/MyDrive/WaterBodyResearch/checkpoints/pretrained_synthetic.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'pretrain_epochs': num_epochs
        }, pretrain_path)
        
        print("Pre-training completed and model saved!")


def create_training_config() -> Dict:
    """Create default training configuration"""
    
    return {
        # Model parameters
        'input_channels': 5,  # RGB + NIR + NDWI
        'num_classes': 6,     # 6 water body types
        'base_channels': 64,
        'use_temporal': True,
        
        # Training parameters
        'learning_rate': 1e-3,
        'pretrain_lr': 2e-3,
        'weight_decay': 1e-4,
        'batch_size': 4,      # Colab T4 GPU limitation
        'num_epochs': 30,
        'pretrain_epochs': 20,
        
        # Loss parameters
        'hierarchy_weight': 0.2,
        'segmentation_weight': 1.0,
        'classification_weight': 1.0,
        
        # Training strategies
        'early_stopping': True,
        'early_stopping_patience': 10,
        'gradient_clipping': 1.0,
        
        # Logging
        'use_wandb': False,  # Set to True if wandb account available
        'log_interval': 10,
        'save_interval': 5,
        
        # Data parameters
        'num_workers': 2,
        'pin_memory': True
    }


def run_complete_training_pipeline(data_path: str, config: Optional[Dict] = None):
    """
    Run the complete training pipeline
    
    Args:
        data_path: Path to processed data
        config: Training configuration (optional)
    """
    
    if config is None:
        config = create_training_config()
    
    print("Water Body Classification Training Pipeline")
    print("=" * 60)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_data_loaders(
        data_path, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_water_body_model(config)
    
    # Pre-training on synthetic data (if available)
    synthetic_path = f"{data_path}/synthetic"
    if os.path.exists(synthetic_path):
        print("\nStarting pre-training on synthetic data...")
        
        # Create synthetic data loader
        from synthetic_data_generator import WaterBodySyntheticGenerator
        generator = WaterBodySyntheticGenerator()
        
        # Generate synthetic data for pre-training
        synthetic_images, synthetic_masks = generator.generate_dataset(
            num_images=1000,
            save_path=synthetic_path
        )
        
        # Create synthetic dataset and loader
        # Note: This would need a custom dataset class for synthetic data
        # For now, we'll skip pre-training and go directly to main training
        print("Synthetic pre-training skipped - proceeding to main training")
    
    # Main training
    print("\nStarting main training...")
    trainer = WaterBodyTrainer(model, data_loaders, config)
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\nTraining pipeline completed!")
    return trainer.model, trainer.best_val_loss


# Example usage
if __name__ == "__main__":
    # Test training configuration
    config = create_training_config()
    print("Training configuration created:")
    print(json.dumps(config, indent=2))
    
    # Test model creation
    model = create_water_body_model(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test loss functions
    hierarchical_loss = HierarchicalLoss(num_classes=6)
    dice_loss = DiceLoss()
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test input
    test_input = torch.randn(2, 3, 5, 512, 512).to(device)  # (B, T, C, H, W)
    test_targets = torch.randint(0, 6, (2,)).to(device)
    
    with torch.no_grad():
        outputs = model(test_input)
        
        # Test classification loss
        cls_features = F.adaptive_avg_pool2d(outputs['features'], (1, 1)).flatten(1)
        cls_predictions = model.hierarchical_classifier.hierarchical_layers['level2'](cls_features)
        
        loss_output = hierarchical_loss(cls_predictions, test_targets)
        
        print(f"\nTest forward pass completed:")
        print(f"Segmentation output: {outputs['segmentation_logits'].shape}")
        print(f"Classification output: {outputs['classification_logits'].shape}")
        print(f"Test loss: {loss_output['total_loss'].item():.4f}")
    
    print("\nTraining pipeline test completed successfully!")