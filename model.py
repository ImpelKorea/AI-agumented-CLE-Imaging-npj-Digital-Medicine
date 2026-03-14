"""
Model definition file
"""

import torch
import torch.nn as nn
import gradientCAM.swin_agcam as timm
import torch.nn.functional as F



class SwinEncoder(nn.Module):

    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=6):
        super().__init__()
        
        # Load complete Swin model
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Analyze SwinTransformer structure
        self.patch_embed = self.backbone.patch_embed
        self.pos_drop = getattr(self.backbone, 'pos_drop', nn.Identity())
        self.layers = self.backbone.layers  # 4 SwinTransformerStages
        self.norm = self.backbone.norm
        
        # Check Feature dimension
        self.num_features = getattr(self.backbone, 'num_features', 1024)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.num_features, num_classes)
        )
        
        print(f"🔍 SwinTransformer structure:")
        print(f"   - Patch Embed: {type(self.patch_embed).__name__}")
        print(f"   - Number of Stages: {len(self.layers)}")
        print(f"   - Feature dimension: {self.num_features}")
        
        for i, layer in enumerate(self.layers):
            print(f"   - Stage {i}: {type(layer).__name__}")
    
    
    def forward(self, x):
        """Standard forward"""
        # Patch Embedding
        x = self.patch_embed(x)  # [B, H, W, C]
        x = self.pos_drop(x)
        
        # Pass through each Stage sequentially
        for layer in self.layers:
            x = layer(x)  # [B, H, W, C]
        
        # Final processing
        x = self.norm(x)  # [B, H, W, C]
        
        # Global Average Pooling
        x = x.mean(dim=[1, 2])  # [B, C]
        
        output = self.classifier(x)
        return output
    
    def get_features(self, x):
        """Extract original features (for Contrastive Loss)"""
        # Patch Embedding
        x = self.patch_embed(x)  # [B, H, W, C]
        x = self.pos_drop(x)
        
        # Pass through each Stage sequentially
        for layer in self.layers:
            x = layer(x)  # [B, H, W, C]
        
        # Final processing
        x = self.norm(x)  # [B, H, W, C]
        
        # Global Average Pooling
        x = x.mean(dim=[1, 2])  # [B, C]
        
        return x



from collections import Counter
class BagLevelSwinModelWithProxyFeature(nn.Module):
    """
    Simplified Model for single-image inference and visualization.
    """
    def __init__(self,
                model_name='swin_base_patch4_window7_224',
                num_classes=6, 
                **kwargs
                ):
        super().__init__()
        self.feature_extractor = SwinEncoder(model_name, num_classes)
        self.instance_classifier = self.feature_extractor.classifier
        self.feature_dim = self.feature_extractor.num_features
        self.num_classes = num_classes

    def forward(self, imgs: torch.Tensor):
        # 1) Extract image features
        instance_feats = self.feature_extractor.get_features(imgs)            # [M,D]
        instance_feats.requires_grad_(True)                          # Prepare to receive gradients
        img_logits = self.instance_classifier(instance_feats)           # [M,C]
        
        return img_logits