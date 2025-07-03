import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QuerySegmentationLoss(nn.Module):
    """
    Loss function for query-based segmentation models like SPSam.
    Combines classification loss for query predictions with mask loss.
    
    Args:
        num_classes: Number of semantic classes
        dice_weight: Weight for the dice loss component
        ce_weight: Weight for the cross entropy loss component
        mask_weight: Weight for the mask loss component
        ignore_index: Label value to ignore in loss calculation
    """
    def __init__(
        self, 
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        mask_weight: float = 5.0,
        ignore_index: Optional[int] = 255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.mask_weight = mask_weight
        self.ignore_index = ignore_index
        
        # Loss for class predictions
        self.class_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, pred_logits, pred_masks, target):
        """
        Calculate loss for query-based segmentation.
        
        Args:
            pred_logits: Tensor of shape [B, Q, C+1] with class predictions for each query
            pred_masks: Tensor of shape [B, Q, H, W] with mask predictions for each query
            target: Tensor of shape [B, H, W] with ground truth semantic labels
            
        Returns:
            Combined loss value
        """
        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]
        
        # Convert semantic segmentation target to one-hot encoding
        # Shape: [B, C, H, W]
        target_one_hot = F.one_hot(
            target.clamp(0, self.num_classes-1), 
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Create binary masks for each class from the target
        # Shape: [B, C, H, W]
        target_masks = target_one_hot
        
        # Initialize total loss
        total_loss = 0.0
        
        # Create ground truth for query predictions
        # We'll assign each query to the best matching ground truth mask
        gt_class = torch.zeros(
            (batch_size, num_queries), 
            dtype=torch.long, 
            device=pred_logits.device
        )
        
        # Calculate IoU between each predicted mask and each target mask
        for b in range(batch_size):
            for q in range(num_queries):
                # Get predicted mask for this query
                pred_mask = torch.sigmoid(pred_masks[b, q])  # [H, W]
                
                best_iou = 0.0
                best_class = self.num_classes  # Default to "no object" class
                
                # Find the best matching class for this query
                for c in range(self.num_classes):
                    target_mask = target_masks[b, c]  # [H, W]
                    
                    # Skip if this class doesn't exist in the target
                    if target_mask.sum() < 1:
                        continue
                    
                    # Calculate IoU
                    intersection = (pred_mask * target_mask).sum()
                    union = pred_mask.sum() + target_mask.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_class = c
                
                # Assign the best matching class to this query
                gt_class[b, q] = best_class
        
        # Classification loss
        cls_loss = self.class_loss(
            pred_logits.view(-1, pred_logits.shape[-1]), 
            gt_class.view(-1)
        )
        
        # Mask loss - only for queries that match a ground truth class
        mask_loss = 0.0
        valid_queries = 0
        
        for b in range(batch_size):
            for q in range(num_queries):
                c = gt_class[b, q].item()
                
                # Skip "no object" queries
                if c == self.num_classes:
                    continue
                    
                # Get predicted mask and target mask for this class
                pred_mask = torch.sigmoid(pred_masks[b, q])
                target_mask = target_masks[b, c]
                
                # Dice loss
                intersection = (pred_mask * target_mask).sum()
                dice_loss = 1 - (2 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
                
                # Binary cross entropy loss
                bce_loss = F.binary_cross_entropy(
                    pred_mask, 
                    target_mask,
                    reduction='mean'
                )
                
                # Combine losses
                mask_loss += self.dice_weight * dice_loss + self.ce_weight * bce_loss
                valid_queries += 1
        
        # Average mask loss over valid queries
        if valid_queries > 0:
            mask_loss = mask_loss / valid_queries
            total_loss = cls_loss + self.mask_weight * mask_loss
        else:
            total_loss = cls_loss
            
        return total_loss