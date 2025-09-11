import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from scipy.optimize import linear_sum_assignment

from models.base_model import BaseSegmentationModel
from models.sam.modeling import Sam
from models.sam.predictor import SamPredictor

from models.super_pixel.superpixel import SPExtractorSAM, SuperpixelExtractor as BaseSuperpixelExtractor
from skimage.measure import regionprops


class HungarianMatcher(nn.Module):
    """Hungarian匹配器，用于匹配预测和真实标签"""
    
    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行匹配
        
        Args:
            outputs: dict包含'pred_logits' (B, N, C+1)和'pred_masks' (B, N, H, W)
            targets: list of dict，每个包含'labels' (num_targets,)和'masks' (num_targets, H, W)
        
        Returns:
            list of tuples (pred_indices, target_indices)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # 展平以便计算成本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*N, C+1]
        out_mask = outputs["pred_masks"].flatten(0, 1)  # [B*N, H, W]
        
        # 收集所有目标
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_mask = torch.cat([v["masks"] for v in targets])
        
        # 计算分类成本
        cost_class = -out_prob[:, tgt_ids]  # 分类成本：负对数概率
        
        # 计算mask成本
        out_mask = out_mask.flatten(1)
        tgt_mask = tgt_mask.flatten(1)
        
        # Focal loss成本
        cost_mask = self.batch_sigmoid_focal_loss(out_mask, tgt_mask)  # focal loss
        
        # Dice loss成本
        cost_dice = self.batch_dice_loss(out_mask, tgt_mask)  # dice loss
        
        # 最终成本矩阵
        C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
        C = C.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    def batch_sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """批量计算sigmoid focal loss"""
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean(1)
    
    def batch_dice_loss(self, inputs, targets):
        """批量计算dice loss"""
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

class SPSamCriterion(nn.Module):
    """SPSam的损失函数"""
    
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        """分类损失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction='mean')
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """掩码损失"""
        assert "pred_masks" in outputs
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        
        masks = [t["masks"] for t in targets]
        target_masks = torch.cat(masks, dim=0).to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        # 展平空间维度
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        
        losses = {
            "loss_mask": self.sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": self.dice_loss(src_masks, target_masks, num_masks),
        }
        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def sigmoid_focal_loss(self, inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
        """Sigmoid focal loss"""
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def dice_loss(self, inputs, targets, num_masks):
        """Dice loss"""
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()
    
    def forward(self, outputs, targets):
        """计算损失"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # 执行匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算目标数量
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 计算所有损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

class SPSam(BaseSegmentationModel):
    """
    SPSam模型：结合超像素提示和SAM的分割模型
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        num_classes: int = 6,
        n_segments: int = 300,
        compactness: float = 10.0,
        multimask_output: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.multimask_output = multimask_output
        self.device = device
        
        # 初始化SAM模型
        from models.sam.build_sam import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        
        # 初始化SAM预测器
        self.predictor = SamPredictor(sam)
        
        # 初始化超像素提取器
        self.superpixel_extractor = SPExtractorSAM(
            n_segments=n_segments,
            compactness=compactness
        )
        
        # 分类头 - 用于将SAM特征映射到类别
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 for background
        
        # 初始化匈牙利匹配器和损失函数
        matcher = HungarianMatcher(cost_class=1.0, cost_mask=1.0, cost_dice=1.0)
        weight_dict = {"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0}
        losses = ["labels", "masks"]
        self.criterion = SPSamCriterion(num_classes, matcher, weight_dict, losses)
    
    def forward(self, images: torch.Tensor, targets=None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像 (B, C, H, W)
            targets: 训练目标 (可选)
        
        Returns:
            预测结果字典
        """
        if self.training:
            assert targets is not None, "Targets must be provided in training mode"
            return self._forward_train(images, targets)
        else:
            return self._forward_inference(images)
    
    def _forward_inference(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理前向传播"""
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 使用超像素提取器获取提示点
        centers_batch, labels_batch = self.superpixel_extractor.extract_centers(images)
        
        all_masks = []
        all_features = []
        all_iou_scores = []
        
        # 2. 对每张图像使用SAM预测
        for i in range(batch_size):
            image = images[i]  # (C, H, W)
            centers = centers_batch[i] if centers_batch[i] is not None else None
            labels = labels_batch[i] if labels_batch[i] is not None else None
            
            if centers is not None and len(centers) > 0:
                # 直接使用torch tensor设置图像到SAM预测器
                original_size = image.shape[-2:]  # (H, W)
                transformed_image = image.unsqueeze(0)  # 添加batch维度: (1, C, H, W)
                
                # 使用set_torch_image方法，避免numpy转换
                self.predictor.set_torch_image(transformed_image, original_size)
                
                # 将numpy提示转换为torch tensor
                centers_torch = torch.as_tensor(centers, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
                
                # 添加batch维度
                centers_torch = centers_torch.unsqueeze(0)  # (1, N, 2)
                labels_torch = labels_torch.unsqueeze(0)    # (1, N)
                
                # 使用predict_torch方法，完全避免numpy转换
                masks, iou_preds, low_res_masks = self.predictor.predict_torch(
                    point_coords=centers_torch,
                    point_labels=labels_torch,
                    multimask_output=self.multimask_output,
                )
                
                # 移除batch维度
                masks = masks[0]  # (C, H, W)
                iou_preds = iou_preds[0]  # (C,)
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0)
                    iou_preds = iou_preds.unsqueeze(0)
                
                # 对掩膜进行NMS处理
                masks, iou_preds = self.mask_nms(masks, iou_preds, iou_threshold=0.8)
                
                # 获取图像嵌入特征用于分类
                image_embedding = self.predictor.get_image_embedding()  # (1, C, H', W')
                image_embedding = image_embedding.squeeze(0)  # (C, H', W')
                
                # 为每个mask计算特征
                mask_features = []
                for mask in masks:
                    # 将mask resize到特征图尺寸
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(), 
                        size=image_embedding.shape[-2:], 
                        mode='bilinear'
                    ).squeeze()
                    
                    # 使用mask加权平均特征
                    weighted_feat = (image_embedding * mask_resized.unsqueeze(0)).sum(dim=(1,2)) / (mask_resized.sum() + 1e-6)
                    mask_features.append(weighted_feat)
                
                if mask_features:
                    image_embedding = torch.stack(mask_features)  # (N, C)
                else:
                    image_embedding = torch.zeros(0, image_embedding.shape[0], device=device)
                
                all_masks.append(masks)
                all_features.append(image_embedding)
                all_iou_scores.append(iou_preds)
            else:
                raise ValueError("No point superpixels extracted for image")
        
        # 3. 处理特征并生成分类预测
        max_masks = max(masks.shape[0] for masks in all_masks)
        
        # 填充到相同长度
        padded_masks = []
        padded_features = []
        
        for masks, features in zip(all_masks, all_features):
            if masks.shape[0] < max_masks:
                pad_size = max_masks - masks.shape[0]
                h, w = masks.shape[-2:]
                mask_pad = torch.zeros((pad_size, h, w), device=device)
                masks = torch.cat([masks, mask_pad], dim=0)
                
                feat_pad = torch.zeros((pad_size, features.shape[-1]), device=device)
                features = torch.cat([features, feat_pad], dim=0)
            
            padded_masks.append(masks)
            padded_features.append(features)
        
        # 堆叠为批次
        pred_masks = torch.stack(padded_masks, dim=0)  # (B, N, H, W)
        batch_features = torch.stack(padded_features, dim=0)  # (B, N, 256)
        
        # 生成分类预测
        pred_logits = self.class_embed(batch_features)  # (B, N, num_classes+1)
        
        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }

    def mask_nms(self, masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对掩膜进行NMS处理，合并重叠度高的掩膜
        
        Args:
            masks: 掩膜张量 (N, H, W)
            scores: IoU分数 (N,)
            iou_threshold: IoU阈值，超过此值的掩膜将被合并
        
        Returns:
            filtered_masks: 过滤后的掩膜 (M, H, W)
            filtered_scores: 过滤后的分数 (M,)
        """
        if masks.shape[0] <= 1:
            return masks, scores
        
        # 计算掩膜面积
        areas = masks.sum(dim=(1, 2)).float()  # (N,)
        
        # 按分数和面积排序（分数优先，面积次之）
        combined_scores = scores + 0.1 * (areas / areas.max())  # 给面积一个小权重
        sorted_indices = torch.argsort(combined_scores, descending=True)
        
        keep = []
        processed = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)
        
        for i in sorted_indices:
            if processed[i]:
                continue
            
            current_mask = masks[i]
            current_area = areas[i]
            keep.append(i.item())
            processed[i] = True
            
            # 找到与当前掩膜重叠的其他掩膜
            for j in sorted_indices:
                if processed[j] or i == j:
                    continue
                
                other_mask = masks[j]
                
                # 计算IoU
                intersection = (current_mask & other_mask).sum().float()
                union = (current_mask | other_mask).sum().float()
                iou = intersection / (union + 1e-6)
                
                if iou > iou_threshold:
                    # 合并掩膜（取并集）
                    merged_mask = current_mask | other_mask
                    masks[i] = merged_mask
                    current_mask = merged_mask
                    
                    # 更新分数（取较高分数）
                    scores[i] = max(scores[i], scores[j])
                    
                    # 标记为已处理
                    processed[j] = True
        
        # 返回保留的掩膜
        keep_indices = torch.tensor(keep, device=masks.device)
        filtered_masks = masks[keep_indices]
        filtered_scores = scores[keep_indices]
        
        return filtered_masks, filtered_scores
    
    def _forward_train(self, images: torch.Tensor, targets) -> Dict[str, torch.Tensor]:
        """训练前向传播"""
        # 获取推理结果
        outputs = self._forward_inference(images)
        
        # 准备训练目标
        prepared_targets = self.prepare_targets(targets, images)
        
        # 计算损失
        losses = self.criterion(outputs, prepared_targets)
        
        return {**outputs, **losses}
    
    def prepare_targets(self, targets, images):
        """将语义分割目标转换为实例分割格式"""
        prepared_targets = []
        
        for target, image in zip(targets, images):
            h, w = target.shape[-2:]
            
            # 获取唯一的类别ID（排除背景）
            unique_classes = torch.unique(target)
            unique_classes = unique_classes[unique_classes != 0]  # 排除背景
            
            if len(unique_classes) == 0:
                # 如果没有前景类别，创建空目标
                prepared_targets.append({
                    "labels": torch.empty(0, dtype=torch.long, device=target.device),
                    "masks": torch.empty(0, h, w, dtype=torch.bool, device=target.device)
                })
                continue
            
            labels = []
            masks = []
            
            for class_id in unique_classes:
                # 为每个类别创建二值掩码
                class_mask = (target == class_id)
                if class_mask.sum() > 0:  # 确保掩码不为空
                    labels.append(class_id - 1)  # 转换为0-based索引
                    masks.append(class_mask)
            
            if len(labels) > 0:
                prepared_targets.append({
                    "labels": torch.tensor(labels, dtype=torch.long, device=target.device),
                    "masks": torch.stack(masks, dim=0)
                })
            else:
                prepared_targets.append({
                    "labels": torch.empty(0, dtype=torch.long, device=target.device),
                    "masks": torch.empty(0, h, w, dtype=torch.bool, device=target.device)
                })
        
        return prepared_targets

