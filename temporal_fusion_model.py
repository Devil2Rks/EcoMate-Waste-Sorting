"""
Novel Temporal Fusion Architecture with ConvLSTM for Water Body Classification
Primary research contribution: Temporal dynamics modeling for Sundarbans seasonal variations

Author: B.Tech Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for spatial-temporal feature learning
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, 
                 kernel_size: int = 3, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional gates: input, forget, output, candidate
        self.conv_gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
    def forward(self, input_tensor: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLSTM cell
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            hidden_state: Tuple of (hidden, cell) states
            
        Returns:
            New (hidden, cell) states
        """
        
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute gates
        gates = self.conv_gates(combined)
        
        # Split into individual gates
        input_gate, forget_gate, output_gate, candidate_gate = torch.split(
            gates, self.hidden_channels, dim=1
        )
        
        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate_gate = torch.tanh(candidate_gate)
        
        # Update cell state
        c_next = forget_gate * c_cur + input_gate * candidate_gate
        
        # Update hidden state
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int, 
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states"""
        
        hidden = torch.zeros(batch_size, self.hidden_channels, height, width, 
                           device=device, dtype=torch.float)
        cell = torch.zeros(batch_size, self.hidden_channels, height, width, 
                         device=device, dtype=torch.float)
        
        return hidden, cell


class MultilevelPixelScaler(nn.Module):
    """
    Multi-level pixel scaling with cross-scale attention
    Processes features at 3 scales: 1x, 1/2x, 1/4x
    """
    
    def __init__(self, input_channels: int = 5, base_channels: int = 64):  # RGB+NIR+NDWI
        super(MultilevelPixelScaler, self).__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # Three scale levels
        self.scales = [1, 2, 4]  # 1x, 1/2x, 1/4x
        self.scale_channels = [base_channels, base_channels * 2, base_channels * 4]
        
        # Scale-specific feature extractors
        self.scale_extractors = nn.ModuleList([
            self._build_scale_extractor(input_channels, channels, scale)
            for channels, scale in zip(self.scale_channels, self.scales)
        ])
        
        # Cross-scale attention
        self.cross_attention = CrossScaleAttention(self.scale_channels)
        
        # Feature fusion
        total_channels = sum(self.scale_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
    def _build_scale_extractor(self, in_channels: int, out_channels: int, 
                              scale: int) -> nn.Module:
        """Build feature extractor for specific scale"""
        
        # Kernel size increases with scale for larger receptive field
        kernel_size = 3 + (scale - 1) * 2
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size, 
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Residual connection
            ResidualBlock(out_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multilevel pixel scaler
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Fused multi-scale features
        """
        
        scale_features = []
        original_size = x.shape[-2:]
        
        # Extract features at different scales
        for extractor, scale in zip(self.scale_extractors, self.scales):
            if scale > 1:
                # Downsample input
                scaled_input = F.interpolate(
                    x, scale_factor=1/scale, mode='bilinear', 
                    align_corners=False, antialias=True
                )
            else:
                scaled_input = x
            
            # Extract features
            features = extractor(scaled_input)
            
            # Upsample back to original size
            if scale > 1:
                features = F.interpolate(
                    features, size=original_size, mode='bilinear', 
                    align_corners=False
                )
            
            scale_features.append(features)
        
        # Apply cross-scale attention
        attended_features = self.cross_attention(scale_features)
        
        # Fuse features
        concatenated = torch.cat(attended_features, dim=1)
        fused_features = self.fusion_conv(concatenated)
        
        return fused_features


class CrossScaleAttention(nn.Module):
    """Lightweight cross-scale attention mechanism"""
    
    def __init__(self, channels_list: List[int]):
        super(CrossScaleAttention, self).__init__()
        
        self.channels_list = channels_list
        
        # Attention computation for each scale
        self.attention_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for channels in channels_list
        ])
        
    def forward(self, scale_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply cross-scale attention"""
        
        attended_features = []
        
        # Compute attention weights for each scale
        attention_maps = []
        for features, attention_conv in zip(scale_features, self.attention_convs):
            attention = attention_conv(features)
            attention_maps.append(attention)
        
        # Apply attention with cross-scale information
        for i, (features, attention) in enumerate(zip(scale_features, attention_maps)):
            # Compute cross-scale attention weights
            other_attentions = [att for j, att in enumerate(attention_maps) if j != i]
            
            if other_attentions:
                cross_attention = torch.mean(torch.stack(other_attentions), dim=0)
                combined_attention = 0.7 * attention + 0.3 * cross_attention
            else:
                combined_attention = attention
            
            attended_features.append(features * combined_attention)
        
        return attended_features


class ResidualBlock(nn.Module):
    """Residual block for feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class TemporalFusionModule(nn.Module):
    """
    ConvLSTM-based temporal fusion module
    Primary research novelty for capturing seasonal dynamics
    """
    
    def __init__(self, input_channels: int = 64, hidden_channels: int = 128,
                 num_layers: int = 2):
        super(TemporalFusionModule, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Stack of ConvLSTM layers
        self.convlstm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_channels = input_channels if i == 0 else hidden_channels
            
            self.convlstm_layers.append(
                ConvLSTMCell(layer_input_channels, hidden_channels)
            )
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_channels)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, input_channels, 1)
        )
        
    def forward(self, temporal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process temporal sequence through ConvLSTM
        
        Args:
            temporal_features: Features with shape (B, T, C, H, W)
            
        Returns:
            Dictionary with temporal fusion results
        """
        
        batch_size, seq_len, channels, height, width = temporal_features.shape
        device = temporal_features.device
        
        # Initialize hidden states for all layers
        hidden_states = []
        for layer in self.convlstm_layers:
            h, c = layer.init_hidden(batch_size, height, width, device)
            hidden_states.append((h, c))
        
        # Process temporal sequence
        layer_outputs = []
        
        for t in range(seq_len):
            layer_input = temporal_features[:, t]  # (B, C, H, W)
            
            # Pass through ConvLSTM layers
            for layer_idx, (layer, (h, c)) in enumerate(zip(self.convlstm_layers, hidden_states)):
                h, c = layer(layer_input, (h, c))
                hidden_states[layer_idx] = (h, c)
                layer_input = h  # Output becomes input for next layer
            
            layer_outputs.append(h)
        
        # Stack temporal outputs
        temporal_outputs = torch.stack(layer_outputs, dim=1)  # (B, T, C, H, W)
        
        # Apply temporal attention
        attended_output = self.temporal_attention(temporal_outputs)
        
        # Project to output space
        final_output = self.output_proj(attended_output)
        
        return {
            'temporal_features': final_output,
            'temporal_sequence': temporal_outputs,
            'attention_weights': self.temporal_attention.last_attention_weights
        }


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for ConvLSTM outputs"""
    
    def __init__(self, channels: int):
        super(TemporalAttention, self).__init__()
        
        self.channels = channels
        
        # Attention computation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1)
        )
        
        self.last_attention_weights = None
        
    def forward(self, temporal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention to sequence
        
        Args:
            temporal_sequence: Tensor with shape (B, T, C, H, W)
            
        Returns:
            Attended temporal features (B, C, H, W)
        """
        
        batch_size, seq_len, channels, height, width = temporal_sequence.shape
        
        # Compute attention weights for each time step
        attention_weights = []
        
        for t in range(seq_len):
            frame = temporal_sequence[:, t]  # (B, C, H, W)
            attention = self.attention_conv(frame)  # (B, 1, H, W)
            attention_weights.append(attention)
        
        # Stack and normalize attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # (B, T, 1, H, W)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Store for visualization
        self.last_attention_weights = attention_weights.detach()
        
        # Apply attention to temporal sequence
        attended_features = torch.sum(
            temporal_sequence * attention_weights, dim=1
        )  # (B, C, H, W)
        
        return attended_features


class WaterBodySegmentationHead(nn.Module):
    """
    Segmentation head for water body detection and cropping
    """
    
    def __init__(self, input_channels: int = 64, num_classes: int = 2):  # Water/Non-water
        super(WaterBodySegmentationHead, self).__init__()
        
        # Decoder layers
        self.decoder = nn.ModuleList([
            # Upsampling layer 1
            nn.Sequential(
                nn.ConvTranspose2d(input_channels, input_channels // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(input_channels // 2),
                nn.ReLU(inplace=True)
            ),
            # Upsampling layer 2  
            nn.Sequential(
                nn.ConvTranspose2d(input_channels // 2, input_channels // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(input_channels // 4),
                nn.ReLU(inplace=True)
            ),
            # Upsampling layer 3
            nn.Sequential(
                nn.ConvTranspose2d(input_channels // 4, input_channels // 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(input_channels // 8),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final classification layer
        self.classifier = nn.Conv2d(input_channels // 8, num_classes, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segmentation head
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        
        x = features
        
        # Apply decoder layers
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # Final classification
        segmentation_logits = self.classifier(x)
        
        return segmentation_logits


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical classifier for water body sub-types with contextual attention
    """
    
    def __init__(self, input_channels: int = 64, num_classes: int = 6):
        super(HierarchicalClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Context attention for surrounding pixels
        self.context_attention = SpatialContextAttention(input_channels)
        
        # Feature extraction for classification
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(input_channels, input_channels * 2, 3, padding=1),
            nn.BatchNorm2d(input_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(input_channels * 2, input_channels * 4, 3, padding=1),
            nn.BatchNorm2d(input_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Hierarchical classification layers
        self.hierarchical_layers = nn.ModuleDict({
            # Level 1: Water type (wetland vs surface vs flowing vs coastal vs temporary vs special)
            'level1': nn.Linear(input_channels * 4, 6),
            
            # Level 2: Specific sub-types
            'level2': nn.Linear(input_channels * 4 + 6, num_classes)
        })
        
        # Hierarchical relationships matrix
        self.register_buffer('hierarchy_matrix', self._build_hierarchy_matrix())
        
    def _build_hierarchy_matrix(self) -> torch.Tensor:
        """Build hierarchical relationships between classes"""
        
        # Define class hierarchy
        hierarchy = torch.zeros(6, 6)  # 6 main categories
        
        # Classes: swamp(0), river(1), estuary(2), tidal_pool(3), shallow_water(4), flood_plain(5)
        # Group relationships (higher values = more similar)
        
        # Wetlands group: swamp
        hierarchy[0, 0] = 1.0  # swamp with itself
        
        # Flowing water group: river  
        hierarchy[1, 1] = 1.0  # river with itself
        
        # Coastal group: estuary, tidal_pool
        hierarchy[2, 2] = 1.0  # estuary
        hierarchy[2, 3] = 0.6  # estuary-tidal_pool similarity
        hierarchy[3, 2] = 0.6  # tidal_pool-estuary similarity
        hierarchy[3, 3] = 1.0  # tidal_pool
        
        # Surface water group: shallow_water
        hierarchy[4, 4] = 1.0  # shallow_water
        
        # Temporary water group: flood_plain
        hierarchy[5, 5] = 1.0  # flood_plain
        
        # Cross-group similarities (lower values)
        hierarchy[0, 4] = 0.3  # swamp-shallow_water (both can be shallow)
        hierarchy[4, 0] = 0.3
        
        hierarchy[1, 5] = 0.4  # river-flood_plain (rivers create flood plains)
        hierarchy[5, 1] = 0.4
        
        return hierarchy
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hierarchical classification forward pass
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            Dictionary with classification results at different levels
        """
        
        # Apply context attention
        context_features = self.context_attention(features)
        
        # Extract global features
        global_features = self.feature_extractor(context_features)
        
        # Level 1 classification (main categories)
        level1_logits = self.hierarchical_layers['level1'](global_features)
        level1_probs = F.softmax(level1_logits, dim=1)
        
        # Level 2 classification (specific sub-types)
        level2_input = torch.cat([global_features, level1_probs], dim=1)
        level2_logits = self.hierarchical_layers['level2'](level2_input)
        
        return {
            'level1_logits': level1_logits,
            'level2_logits': level2_logits,
            'level1_probs': level1_probs,
            'level2_probs': F.softmax(level2_logits, dim=1),
            'global_features': global_features,
            'context_features': context_features
        }
    
    def compute_hierarchical_loss(self, level2_logits: torch.Tensor, 
                                 targets: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical loss with class relationship penalties"""
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(level2_logits, targets)
        
        # Hierarchical penalty
        pred_probs = F.softmax(level2_logits, dim=1)
        
        hierarchy_penalty = 0.0
        for i in range(targets.shape[0]):
            target_class = targets[i].item()
            
            for j in range(self.num_classes):
                if j != target_class:
                    # Penalty inversely proportional to hierarchy similarity
                    similarity = self.hierarchy_matrix[target_class, j]
                    penalty = pred_probs[i, j] * (1 - similarity)
                    hierarchy_penalty += penalty
        
        hierarchy_penalty = hierarchy_penalty / targets.shape[0]
        
        return ce_loss + 0.2 * hierarchy_penalty


class SpatialContextAttention(nn.Module):
    """Single-layer self-attention for spatial context"""
    
    def __init__(self, channels: int):
        super(SpatialContextAttention, self).__init__()
        
        self.channels = channels
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.output_conv = nn.Conv2d(channels, channels, 1)
        
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial context attention
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Context-aware features (B, C, H, W)
        """
        
        batch_size, channels, height, width = x.shape
        
        # Compute Q, K, V
        query = self.query_conv(x).view(batch_size, -1, height * width)  # (B, C//8, H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)      # (B, C//8, H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, H*W)
        
        # Compute attention scores
        attention_scores = torch.bmm(query.transpose(1, 2), key)  # (B, H*W, H*W)
        attention_weights = F.softmax(attention_scores / np.sqrt(channels // 8), dim=2)
        
        # Apply attention to values
        attended_values = torch.bmm(value, attention_weights.transpose(1, 2))  # (B, C, H*W)
        
        # Reshape back to spatial dimensions
        attended_values = attended_values.view(batch_size, channels, height, width)
        
        # Apply learnable scale and residual connection
        output = self.output_conv(attended_values)
        output = x + self.scale * output
        
        return output


class WaterBodyClassificationModel(nn.Module):
    """
    Complete water body classification model with temporal fusion
    Primary research contribution combining all novel components
    """
    
    def __init__(self, input_channels: int = 5, num_classes: int = 6,
                 base_channels: int = 64, use_temporal: bool = True):
        super(WaterBodyClassificationModel, self).__init__()
        
        self.input_channels = input_channels  # RGB + NIR + NDWI
        self.num_classes = num_classes
        self.use_temporal = use_temporal
        
        # Multi-level pixel scaler
        self.pixel_scaler = MultilevelPixelScaler(input_channels, base_channels)
        
        # Temporal fusion (if using temporal data)
        if use_temporal:
            self.temporal_fusion = TemporalFusionModule(base_channels, base_channels * 2)
        
        # Segmentation head for water detection
        self.segmentation_head = WaterBodySegmentationHead(base_channels, num_classes=2)
        
        # Hierarchical classifier for water body types
        self.hierarchical_classifier = HierarchicalClassifier(base_channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass
        
        Args:
            x: Input tensor (B, T, C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary with all model outputs
        """
        
        if len(x.shape) == 5 and self.use_temporal:
            # Temporal input: (B, T, C, H, W)
            batch_size, seq_len, channels, height, width = x.shape
            
            # Process each frame through pixel scaler
            temporal_features = []
            for t in range(seq_len):
                frame_features = self.pixel_scaler(x[:, t])  # (B, C, H, W)
                temporal_features.append(frame_features)
            
            # Stack temporal features
            temporal_features = torch.stack(temporal_features, dim=1)  # (B, T, C, H, W)
            
            # Apply temporal fusion
            fusion_output = self.temporal_fusion(temporal_features)
            final_features = fusion_output['temporal_features']
            
        else:
            # Single frame input: (B, C, H, W)
            if len(x.shape) == 5:
                x = x[:, 0]  # Take first frame
            
            final_features = self.pixel_scaler(x)
            fusion_output = {'temporal_features': final_features}
        
        # Water body segmentation
        segmentation_logits = self.segmentation_head(final_features)
        
        # Water body classification
        classification_output = self.hierarchical_classifier(final_features)
        
        return {
            'segmentation_logits': segmentation_logits,
            'classification_logits': classification_output['level2_logits'],
            'level1_logits': classification_output['level1_logits'],
            'features': final_features,
            'temporal_info': fusion_output,
            'classification_probs': classification_output['level2_probs']
        }
    
    def crop_water_regions(self, image: torch.Tensor, 
                          segmentation_mask: torch.Tensor,
                          min_area: int = 100) -> List[Dict]:
        """
        Crop water regions from segmented image
        
        Args:
            image: Original image (B, C, H, W)
            segmentation_mask: Binary segmentation mask (B, H, W)
            min_area: Minimum area for valid water regions
            
        Returns:
            List of cropped water regions with bounding boxes
        """
        
        batch_size = image.shape[0]
        cropped_regions = []
        
        for b in range(batch_size):
            mask_np = segmentation_mask[b].cpu().numpy().astype(np.uint8)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
            
            for label_id in range(1, num_labels):  # Skip background (0)
                area = stats[label_id, cv2.CC_STAT_AREA]
                
                if area >= min_area:
                    # Get bounding box
                    x = stats[label_id, cv2.CC_STAT_LEFT]
                    y = stats[label_id, cv2.CC_STAT_TOP]
                    w = stats[label_id, cv2.CC_STAT_WIDTH]
                    h = stats[label_id, cv2.CC_STAT_HEIGHT]
                    
                    # Add padding
                    padding = 16
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(image.shape[3], x + w + padding)
                    y_end = min(image.shape[2], y + h + padding)
                    
                    # Crop image and mask
                    cropped_image = image[b, :, y_start:y_end, x_start:x_end]
                    cropped_mask = (labels == label_id)[y_start:y_end, x_start:x_end]
                    
                    cropped_regions.append({
                        'batch_idx': b,
                        'region_id': label_id,
                        'image': cropped_image,
                        'mask': torch.from_numpy(cropped_mask.astype(np.uint8)),
                        'bbox': (x_start, y_start, x_end, y_end),
                        'area': area,
                        'centroid': centroids[label_id]
                    })
        
        return cropped_regions


# Model factory function
def create_water_body_model(config: Dict) -> WaterBodyClassificationModel:
    """
    Factory function to create water body classification model
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    
    model = WaterBodyClassificationModel(
        input_channels=config.get('input_channels', 5),
        num_classes=config.get('num_classes', 6),
        base_channels=config.get('base_channels', 64),
        use_temporal=config.get('use_temporal', True)
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        'input_channels': 5,  # RGB + NIR + NDWI
        'num_classes': 6,     # 6 water body types
        'base_channels': 64,
        'use_temporal': True
    }
    
    # Create model
    model = create_water_body_model(config).to(device)
    
    # Test temporal input
    batch_size, seq_len, channels, height, width = 2, 3, 5, 512, 512
    test_input = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(test_input)
    
    print("Forward pass completed:")
    print(f"Segmentation logits: {outputs['segmentation_logits'].shape}")
    print(f"Classification logits: {outputs['classification_logits'].shape}")
    print(f"Features: {outputs['features'].shape}")
    
    # Test water region cropping
    dummy_seg_mask = torch.randint(0, 2, (batch_size, height, width))
    cropped_regions = model.crop_water_regions(test_input[:, 0], dummy_seg_mask)
    
    print(f"Cropped {len(cropped_regions)} water regions")
    
    print("Model architecture test completed successfully!")