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

DEBUG = False

# class HungarianMatcher(nn.Module):
#     """Hungarian匹配器，用于匹配预测和真实标签"""
    
#     def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_mask = cost_mask
#         self.cost_dice = cost_dice
    
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         """
#         执行匹配
        
#         Args:
#             outputs: dict包含'pred_logits' (B, N, C+1)和'pred_masks' (B, N, H, W)
#             targets: list of dict，每个包含'labels' (num_targets,)和'masks' (num_targets, H, W)
        
#         Returns:
#             list of tuples (pred_indices, target_indices)
#         """
#         batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
#         # 对每个batch分别处理
#         indices = []
        
#         for batch_idx in range(batch_size):
#             # 获取当前batch的预测
#             out_prob = outputs["pred_logits"][batch_idx].softmax(-1)  # [N, C+1]
#             out_mask = outputs["pred_masks"][batch_idx]  # [N, H, W]
            
#             # 获取当前batch的目标
#             tgt_ids = targets[batch_idx]["labels"]  # [num_targets]
#             tgt_mask = targets[batch_idx]["masks"]  # [num_targets, H, W]
            
#             if len(tgt_ids) == 0:
#                 # 如果没有目标，返回空匹配
#                 indices.append((torch.as_tensor([], dtype=torch.int64), 
#                               torch.as_tensor([], dtype=torch.int64)))
#                 continue
            
#             # 展平mask用于计算成本
#             out_mask_flat = out_mask.flatten(1).float()  # [N, H*W] - 转换为float
#             tgt_mask_flat = tgt_mask.flatten(1).float()  # [num_targets, H*W] - 确保也是float
            
#             # 计算分类成本 [N, num_targets]
#             cost_class = -out_prob[:, tgt_ids].transpose(0, 1)  # [num_targets, N] -> [N, num_targets]
            
#             # 计算mask成本 [N, num_targets]
#             cost_mask = torch.cdist(out_mask_flat, tgt_mask_flat, p=1)
            
#             # 计算dice成本 [N, num_targets]
#             cost_dice = torch.zeros_like(cost_mask)
#             for i in range(num_queries):
#                 for j in range(len(tgt_ids)):
#                     cost_dice[i, j] = self.batch_dice_loss(
#                         out_mask_flat[i:i+1], tgt_mask_flat[j:j+1]
#                     ).item()
            
#             # 最终成本矩阵 [N, num_targets]
#             C = (self.cost_mask * cost_mask + 
#                  self.cost_class * cost_class + 
#                  self.cost_dice * cost_dice)
            
#             # Hungarian算法匹配
#             pred_indices, tgt_indices = linear_sum_assignment(C.cpu().numpy())
            
#             indices.append((torch.as_tensor(pred_indices, dtype=torch.int64), 
#                            torch.as_tensor(tgt_indices, dtype=torch.int64)))
        
#         return indices
    
#     def batch_sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
#         """批量计算sigmoid focal loss"""
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         loss = ce_loss * ((1 - p_t) ** gamma)
        
#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss
        
#         return loss.mean(1)
    
#     def batch_dice_loss(self, inputs, targets):
#         """批量计算dice loss"""
#         inputs = inputs.sigmoid()
#         numerator = 2 * (inputs * targets).sum(-1)  # 在最后一个维度求和
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = 1 - (numerator + 1) / (denominator + 1)
#         return loss

class SPSamCriterion(nn.Module):
    """SPSam的损失函数
    
    Args:
        num_classes: 类别数量
        matcher: 匹配器 (可选，仅用于实例分割)
        weight_dict: 损失权重字典
        losses: 损失列表
        use_hungarian: 是否使用Hungarian匹配 (True for instance, False for semantic)
        
    Returns:
        损失字典
    
    """
    
    def __init__(self, num_classes, matcher=None, weight_dict=None, losses=None, use_hungarian=False):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {}
        self.losses = losses or ["labels", "masks"]
        self.use_hungarian = use_hungarian
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        """分类损失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 确保类别索引在有效范围内
        if len(target_classes_o) > 0:
            max_class = target_classes_o.max().item()
            if max_class >= self.num_classes:
                target_classes_o = torch.clamp(target_classes_o, 0, self.num_classes - 1)
        
        # 背景类索引
        background_class_idx = self.num_classes  # 使用num_classes作为背景类
        target_classes = torch.full(src_logits.shape[:2], background_class_idx,
                                  dtype=torch.int64, device=src_logits.device)
        
        # 安全检查索引
        if len(idx[0]) > 0:
            max_batch_idx = idx[0].max().item()
            max_src_idx = idx[1].max().item()
            batch_size, num_queries = target_classes.shape
            
            if max_batch_idx >= batch_size or max_src_idx >= num_queries:
                return {"loss_ce": torch.tensor(0.0, device=src_logits.device, requires_grad=True)}
        
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
        """Sigmoid focal loss on probabilities.
        This function is robust to boolean targets/inputs coming from masks.
        We treat `inputs` as probabilities in [0,1] (e.g., predicted masks),
        not raw logits.
        """
        # Ensure floating point tensors
        inputs = inputs.float()
        targets = targets.float()

        # If inputs might be logits, optionally map with sigmoid.
        # Here we assume masks/probabilities; just clamp to [0,1]
        prob = inputs.clamp(0.0, 1.0)

        # Standard BCE on probabilities (NOT with logits)
        ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")

        # p_t for focal modulation
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss

        return loss.mean()
    
    def dice_loss(self, inputs, targets, num_masks):
        """Dice loss on probabilities (robust to bool targets)."""
        inputs = inputs.float().clamp(0.0, 1.0)
        targets = targets.float()
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()
    
    def forward(self, outputs, targets):
        """计算损失"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        if self.use_hungarian and self.matcher is not None:
            # 实例分割模式：使用Hungarian匹配
            indices = self.matcher(outputs_without_aux, targets)
            num_masks = sum(len(t["labels"]) for t in targets)
            num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
            
            # 计算所有损失
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        else:
            # 语义分割模式：直接计算损失，不使用匹配
            losses = self.compute_semantic_loss(outputs, targets)
        
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)
    
    def compute_semantic_loss(self, outputs, targets):
        """直接的语义分割损失计算"""
        pred_logits = outputs["pred_logits"]  # [B, N, C+1]
        pred_masks = outputs["pred_masks"]    # [B, N, H, W]
        
        # 将查询预测转换为语义分割logits格式
        semantic_logits = self.convert_queries_to_semantic_logits(pred_logits, pred_masks)  # [B, C+1, H, W]
        
        # 确保targets是正确的数据类型
        targets = targets.long()  # 转换为int64
        
        # 计算标准的语义分割损失
        ce_loss = F.cross_entropy(
            semantic_logits, 
            targets, 
            ignore_index=255
        )
        
        return {"loss_ce": ce_loss}
    
    def convert_queries_to_semantic_logits(self, pred_logits, pred_masks):
        """将查询预测转换为语义分割logits"""
        batch_size, num_queries, num_classes_plus1 = pred_logits.shape
        _, _, height, width = pred_masks.shape
        device = pred_masks.device
        
        # 初始化语义分割logits (B, C+1, H, W)
        semantic_logits = torch.zeros(batch_size, num_classes_plus1, height, width, device=device)
        
        # 获取类别概率和mask概率
        class_probs = F.softmax(pred_logits, dim=-1)  # [B, N, C+1]
        mask_probs = pred_masks.sigmoid()  # [B, N, H, W]
        
        # 对每个查询，将其贡献累加到对应类别的logits中
        for query_idx in range(num_queries):
            query_class_probs = class_probs[:, query_idx, :]  # [B, C+1]
            query_mask_probs = mask_probs[:, query_idx, :, :]  # [B, H, W]
            
            # 将类别概率和mask概率结合
            for class_idx in range(num_classes_plus1):
                class_prob = query_class_probs[:, class_idx]  # [B]
                combined_prob = class_prob.unsqueeze(-1).unsqueeze(-1) * query_mask_probs  # [B, H, W]
                semantic_logits[:, class_idx, :, :] += combined_prob
        
        return semantic_logits
    
    def compute_iou_matrix(self, pred_masks, gt_masks):
        """计算预测masks和ground truth masks之间的IoU矩阵"""
        # Binarize predicted and GT masks robustly
        if pred_masks.dtype == torch.bool:
            pred_bin = pred_masks
        else:
            # If already probabilities in [0,1], threshold at 0.5; if logits, user should map before.
            pred_bin = (pred_masks > 0.5)

        if gt_masks.dtype == torch.bool:
            gt_bin = gt_masks
        else:
            gt_bin = (gt_masks > 0.5)

        N, M = pred_bin.shape[0], gt_bin.shape[0]
        iou_matrix = torch.zeros(N, M, device=pred_bin.device)

        for i in range(N):
            for j in range(M):
                intersection = (pred_bin[i] & gt_bin[j]).sum().float()
                union = (pred_bin[i] | gt_bin[j]).sum().float()
                iou_matrix[i, j] = intersection / (union + 1e-6)

        return iou_matrix
    
    def convert_semantic_to_instance(self, semantic_mask):
        """将语义分割mask转换为实例格式"""
        unique_classes = torch.unique(semantic_mask)
        unique_classes = unique_classes[unique_classes != 0]  # 排除背景
        
        labels = []
        masks = []
        
        for class_id in unique_classes:
            if class_id > 0 and class_id <= self.num_classes:
                labels.append(class_id - 1)  # 转为0-based
                masks.append(semantic_mask == class_id)
        
        if len(labels) > 0:
            return torch.tensor(labels, device=semantic_mask.device), torch.stack(masks)
        else:
            return torch.empty(0, dtype=torch.long, device=semantic_mask.device), torch.empty(0, *semantic_mask.shape, dtype=torch.bool, device=semantic_mask.device)

class SPSam(BaseSegmentationModel):
    """
    SPSam模型：结合超像素提示和SAM的分割模型
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        num_classes: int = 8,
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
        
        # 初始化损失函数
        self.criterion = SPSamCriterion(num_classes, 
                                        weight_dict = {"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0}, 
                                        losses = ["labels", "masks"])
    
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
        result = self.superpixel_extractor.extract_centers(images)
        centers_batch = result['centers']
        labels_batch = result['labels']
        
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
                # print("No point superpixels extracted for image {image[]}, skipping...")
                # continue
                raise ValueError("No point superpixels extracted for image", i)
        
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
        
        # 对于语义分割，直接使用原始targets，不需要prepare_targets
        if self.criterion.use_hungarian:
            # 实例分割模式：需要转换格式
            prepared_targets = self.prepare_targets(targets, images)
            losses = self.criterion(outputs, prepared_targets)
        else:
            # 语义分割模式：直接计算损失
            losses = self.criterion.compute_semantic_loss(outputs, targets)
        
        return {**outputs, **losses}
    
    def prepare_targets(self, targets, images):
        """将语义分割目标转换为实例分割格式"""
        prepared_targets = []
        
        for target, image in zip(targets, images):
            h, w = target.shape[-2:]
            
            # 获取唯一的类别ID（排除背景）
            unique_classes = torch.unique(target)
            unique_classes = unique_classes[unique_classes != 0]  # 排除背景
            
            if DEBUG:
                print(f"Debug - unique_classes: {unique_classes}")
                print(f"Debug - num_classes: {self.num_classes}")
                print(f"Debug - class_embed output size: {self.class_embed.out_features}")
            
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
                # 确保class_id在有效范围内 (1-based -> 0-based)
                zero_based_id = class_id - 1  # 转换为0-based索引
                if zero_based_id >= self.num_classes or zero_based_id < 0:
                    print(f"Warning: Skipping class_id {class_id} (0-based: {zero_based_id}) outside range [0, {self.num_classes-1}]")
                    continue
                
                # 为每个类别创建二值掩码
                class_mask = (target == class_id)
                if class_mask.sum() > 0:  # 确保掩码不为空
                    labels.append(zero_based_id)  # 使用0-based索引
                    masks.append(class_mask)
            
            if len(labels) > 0:
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=target.device)
                
                prepared_targets.append({
                    "labels": labels_tensor,
                    "masks": torch.stack(masks, dim=0)
                })
            else:
                prepared_targets.append({
                    "labels": torch.empty(0, dtype=torch.long, device=target.device),
                    "masks": torch.empty(0, h, w, dtype=torch.bool, device=target.device)
                })
        
        return prepared_targets

