"""
Baseline Models for Comparison: U-Net and DeepLabV3 (from scratch)
Implementation for rigorous evaluation against our temporal fusion approach

Author: B.Tech Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class UNetBaseline(nn.Module):
    """
    U-Net implementation from scratch for water body segmentation baseline
    """
    
    def __init__(self, input_channels: int = 5, num_classes: int = 6, 
                 base_channels: int = 64):
        super(UNetBaseline, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Encoder (downsampling path)
        self.encoder1 = self._double_conv(input_channels, base_channels)
        self.encoder2 = self._double_conv(base_channels, base_channels * 2)
        self.encoder3 = self._double_conv(base_channels * 2, base_channels * 4)
        self.encoder4 = self._double_conv(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._double_conv(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.decoder4 = self._double_conv(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = self._double_conv(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = self._double_conv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = self._double_conv(base_channels * 2, base_channels)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
        # Max pooling for encoder
        self.pool = nn.MaxPool2d(2, stride=2)
        
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Double convolution block used in U-Net"""
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Dictionary with segmentation outputs
        """
        
        # Handle temporal input by taking first frame
        if len(x.shape) == 5:
            x = x[:, 0]  # Take first temporal frame
        
        # Encoder path
        enc1 = self.encoder1(x)          # (B, 64, H, W)
        pool1 = self.pool(enc1)          # (B, 64, H/2, W/2)
        
        enc2 = self.encoder2(pool1)      # (B, 128, H/2, W/2)
        pool2 = self.pool(enc2)          # (B, 128, H/4, W/4)
        
        enc3 = self.encoder3(pool2)      # (B, 256, H/4, W/4)
        pool3 = self.pool(enc3)          # (B, 256, H/8, W/8)
        
        enc4 = self.encoder4(pool3)      # (B, 512, H/8, W/8)
        pool4 = self.pool(enc4)          # (B, 512, H/16, W/16)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # (B, 1024, H/16, W/16)
        
        # Decoder path with skip connections
        up4 = self.upconv4(bottleneck)   # (B, 512, H/8, W/8)
        merge4 = torch.cat([up4, enc4], dim=1)  # (B, 1024, H/8, W/8)
        dec4 = self.decoder4(merge4)     # (B, 512, H/8, W/8)
        
        up3 = self.upconv3(dec4)         # (B, 256, H/4, W/4)
        merge3 = torch.cat([up3, enc3], dim=1)  # (B, 512, H/4, W/4)
        dec3 = self.decoder3(merge3)     # (B, 256, H/4, W/4)
        
        up2 = self.upconv2(dec3)         # (B, 128, H/2, W/2)
        merge2 = torch.cat([up2, enc2], dim=1)  # (B, 256, H/2, W/2)
        dec2 = self.decoder2(merge2)     # (B, 128, H/2, W/2)
        
        up1 = self.upconv1(dec2)         # (B, 64, H, W)
        merge1 = torch.cat([up1, enc1], dim=1)  # (B, 128, H, W)
        dec1 = self.decoder1(merge1)     # (B, 64, H, W)
        
        # Final classification
        output = self.final_conv(dec1)   # (B, num_classes, H, W)
        
        return {
            'segmentation_logits': output,
            'classification_logits': output,  # Same as segmentation for U-Net
            'features': dec1
        }


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for DeepLabV3
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 atrous_rates: List[int] = [6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.atrous_rates = atrous_rates
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                             dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels * (len(atrous_rates) + 2)  # +2 for 1x1 and global pooling
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP module"""
        
        size = x.shape[-2:]
        
        # 1x1 convolution
        conv1x1_out = self.conv1x1(x)
        
        # Atrous convolutions
        atrous_outs = []
        for atrous_conv in self.atrous_convs:
            atrous_outs.append(atrous_conv(x))
        
        # Global average pooling
        global_out = self.global_avg_pool(x)
        global_out = F.interpolate(global_out, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all outputs
        concat_out = torch.cat([conv1x1_out] + atrous_outs + [global_out], dim=1)
        
        # Final projection
        output = self.project(concat_out)
        
        return output


class DeepLabV3Baseline(nn.Module):
    """
    DeepLabV3 implementation from scratch for water body segmentation baseline
    """
    
    def __init__(self, input_channels: int = 5, num_classes: int = 6,
                 base_channels: int = 64):
        super(DeepLabV3Baseline, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Backbone encoder (simplified ResNet-like)
        self.backbone = self._build_backbone(input_channels, base_channels)
        
        # ASPP module
        self.aspp = ASPPModule(base_channels * 8, base_channels * 2)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(base_channels // 4, num_classes, 1)
        
    def _build_backbone(self, input_channels: int, base_channels: int) -> nn.Module:
        """Build simplified ResNet-like backbone"""
        
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Layer 1
            self._make_layer(base_channels, base_channels, 2, stride=1),
            
            # Layer 2
            self._make_layer(base_channels, base_channels * 2, 2, stride=2),
            
            # Layer 3 (with dilation)
            self._make_layer(base_channels * 2, base_channels * 4, 2, stride=1, dilation=2),
            
            # Layer 4 (with higher dilation)
            self._make_layer(base_channels * 4, base_channels * 8, 2, stride=1, dilation=4)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1, dilation: int = 1) -> nn.Module:
        """Create a layer with multiple residual blocks"""
        
        layers = []
        
        # First block (may have stride or channel change)
        layers.append(ResNetBlock(in_channels, out_channels, stride, dilation))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DeepLabV3
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Dictionary with segmentation outputs
        """
        
        # Handle temporal input by taking first frame
        if len(x.shape) == 5:
            x = x[:, 0]  # Take first temporal frame
        
        # Backbone feature extraction
        features = self.backbone(x)  # (B, 512, H/8, W/8)
        
        # ASPP module
        aspp_features = self.aspp(features)  # (B, 128, H/8, W/8)
        
        # Decoder
        decoded = self.decoder(aspp_features)  # (B, 16, H, W)
        
        # Final classification
        output = self.classifier(decoded)  # (B, num_classes, H, W)
        
        return {
            'segmentation_logits': output,
            'classification_logits': output,  # Same as segmentation
            'features': decoded
        }


class ResNetBlock(nn.Module):
    """ResNet-style residual block with optional dilation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, dilation: int = 1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation,
                              dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block"""
        
        residual = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class SimpleClassificationBaseline(nn.Module):
    """
    Simple CNN baseline for direct water body classification
    """
    
    def __init__(self, input_channels: int = 5, num_classes: int = 6):
        super(SimpleClassificationBaseline, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through simple classifier"""
        
        # Handle temporal input
        if len(x.shape) == 5:
            x = x[:, 0]  # Take first temporal frame
        
        features = self.features(x)
        classification_logits = self.classifier(features)
        
        # Create dummy segmentation output (all water)
        batch_size, _, height, width = x.shape
        dummy_seg = torch.ones(batch_size, 2, height, width, device=x.device)
        dummy_seg[:, 0] *= 0.1  # Low probability for background
        dummy_seg[:, 1] *= 0.9  # High probability for water
        
        return {
            'segmentation_logits': dummy_seg,
            'classification_logits': classification_logits,
            'features': features.squeeze()
        }


class BaselineTrainer:
    """
    Trainer for baseline models with simplified training loop
    """
    
    def __init__(self, model: nn.Module, data_loaders: Dict[str, DataLoader],
                 model_name: str, config: Dict):
        self.model = model
        self.data_loaders = data_loaders
        self.model_name = model_name
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch: int) -> float:
        """Train baseline model for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.data_loaders['train'], desc=f'{self.model_name} Epoch {epoch}'):
            images = batch['images'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss based on model type
            if 'segmentation_logits' in outputs:
                # Segmentation loss
                loss = self.loss_fn(outputs['segmentation_logits'], masks)
            else:
                # Classification loss (use most frequent class in mask as target)
                targets = []
                for b in range(images.shape[0]):
                    water_mask = masks[b] > 0
                    if water_mask.any():
                        target_class = torch.mode(masks[b][water_mask]).values
                    else:
                        target_class = torch.tensor(0)
                    targets.append(target_class)
                
                targets = torch.stack(targets).to(self.device)
                loss = self.loss_fn(outputs['classification_logits'], targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Validate baseline model for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc=f'{self.model_name} Val'):
                images = batch['images'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss and accuracy
                if 'segmentation_logits' in outputs:
                    loss = self.loss_fn(outputs['segmentation_logits'], masks)
                    
                    # Segmentation accuracy
                    pred_masks = torch.argmax(outputs['segmentation_logits'], dim=1)
                    correct_predictions += torch.sum(pred_masks == masks).item()
                    total_predictions += masks.numel()
                else:
                    # Classification accuracy
                    targets = []
                    for b in range(images.shape[0]):
                        water_mask = masks[b] > 0
                        if water_mask.any():
                            target_class = torch.mode(masks[b][water_mask]).values
                        else:
                            target_class = torch.tensor(0)
                        targets.append(target_class)
                    
                    targets = torch.stack(targets).to(self.device)
                    loss = self.loss_fn(outputs['classification_logits'], targets)
                    
                    pred_classes = torch.argmax(outputs['classification_logits'], dim=1)
                    correct_predictions += torch.sum(pred_classes == targets).item()
                    total_predictions += targets.shape[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = 20):
        """Train baseline model"""
        
        print(f"Training {self.model_name} for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch + 1)
            
            # Validation
            val_loss, val_accuracy = self.validate_epoch(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = f"/content/drive/MyDrive/WaterBodyResearch/checkpoints/{self.model_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_loss': best_val_loss,
                    'val_accuracy': val_accuracy,
                    'model_name': self.model_name
                }, checkpoint_path)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"{self.model_name} training completed!")
        return best_val_loss, max(self.val_accuracies)


def create_baseline_models(config: Dict) -> Dict[str, nn.Module]:
    """
    Create all baseline models for comparison
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary of baseline models
    """
    
    models = {
        'unet': UNetBaseline(
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ),
        'deeplabv3': DeepLabV3Baseline(
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ),
        'simple_cnn': SimpleClassificationBaseline(
            input_channels=config['input_channels'],
            num_classes=config['num_classes']
        )
    }
    
    return models


def train_all_baselines(data_loaders: Dict[str, DataLoader], config: Dict) -> Dict[str, Dict]:
    """
    Train all baseline models for comparison
    
    Args:
        data_loaders: Data loaders for training/validation
        config: Training configuration
        
    Returns:
        Dictionary with baseline results
    """
    
    print("Training Baseline Models for Comparison")
    print("=" * 50)
    
    # Create baseline models
    baseline_models = create_baseline_models(config)
    
    results = {}
    
    for model_name, model in baseline_models.items():
        print(f"\nTraining {model_name.upper()}...")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = BaselineTrainer(model, data_loaders, model_name, config)
        
        # Train model
        best_val_loss, best_val_accuracy = trainer.train(
            num_epochs=config.get('baseline_epochs', 15)
        )
        
        results[model_name] = {
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'val_accuracies': trainer.val_accuracies,
            'model': model
        }
        
        print(f"{model_name} - Best Val Loss: {best_val_loss:.4f}, "
              f"Best Val Acc: {best_val_accuracy:.4f}")
    
    # Save baseline comparison
    comparison_path = "/content/drive/MyDrive/WaterBodyResearch/baseline_comparison.json"
    comparison_data = {
        model_name: {
            'best_val_loss': float(results[model_name]['best_val_loss']),
            'best_val_accuracy': float(results[model_name]['best_val_accuracy'])
        }
        for model_name in results.keys()
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Plot comparison
    plot_baseline_comparison(results)
    
    return results


def plot_baseline_comparison(results: Dict[str, Dict]):
    """Plot comparison of baseline models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    axes[0, 0].set_title('Training Loss Comparison')
    for model_name, result in results.items():
        epochs = range(1, len(result['train_losses']) + 1)
        axes[0, 0].plot(epochs, result['train_losses'], label=model_name)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation loss comparison
    axes[0, 1].set_title('Validation Loss Comparison')
    for model_name, result in results.items():
        epochs = range(1, len(result['val_losses']) + 1)
        axes[0, 1].plot(epochs, result['val_losses'], label=model_name)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation accuracy comparison
    axes[1, 0].set_title('Validation Accuracy Comparison')
    for model_name, result in results.items():
        epochs = range(1, len(result['val_accuracies']) + 1)
        axes[1, 0].plot(epochs, result['val_accuracies'], label=model_name)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Final performance bar chart
    axes[1, 1].set_title('Final Performance Comparison')
    model_names = list(results.keys())
    best_accuracies = [results[name]['best_val_accuracy'] for name in model_names]
    
    bars = axes[1, 1].bar(model_names, best_accuracies)
    axes[1, 1].set_ylabel('Best Validation Accuracy')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, best_accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/WaterBodyResearch/baseline_comparison.png', 
               dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Test baseline model creation
    config = {
        'input_channels': 5,
        'num_classes': 6,
        'base_channels': 64,
        'learning_rate': 1e-3,
        'baseline_epochs': 15
    }
    
    print("Creating baseline models...")
    baseline_models = create_baseline_models(config)
    
    for name, model in baseline_models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name}: {param_count:,} parameters")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(2, 5, 512, 512).to(device)
    
    for name, model in baseline_models.items():
        model.to(device)
        with torch.no_grad():
            outputs = model(test_input)
            print(f"{name} output shapes:")
            for key, tensor in outputs.items():
                print(f"  {key}: {tensor.shape}")
    
    print("Baseline models test completed successfully!")