import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tools.metric import Evaluator
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn.functional as F

class BaseSegmentationModel(nn.Module):
    """分割模型基类"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass

    def _convert_to_semantic_segmentation(self, pred_logits, pred_masks):
        """
        将实例分割结果转换为语义分割格式
        
        Args:
            pred_logits: (B, N, num_classes+1) 分类预测
            pred_masks: (B, N, H, W) 掩膜预测
        
        Returns:
            semantic_seg: (B, H, W) 语义分割结果
        """
        batch_size, num_queries, height, width = pred_masks.shape
        device = pred_masks.device
        
        # 初始化语义分割结果（背景为0）
        semantic_seg = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            # 获取每个query的最大概率类别（排除背景类）
            class_probs = pred_logits[b].softmax(dim=-1)  # (N, num_classes+1)
            class_scores, class_ids = class_probs[:, :-1].max(dim=-1)  # 排除背景类
            
            # 获取掩膜概率
            mask_probs = pred_masks[b].sigmoid()  # (N, H, W)
            
            # 结合分类和掩膜置信度
            combined_scores = class_scores.unsqueeze(-1).unsqueeze(-1) * mask_probs  # (N, H, W)
            
            # 找到每个像素的最佳预测
            best_scores, best_indices = combined_scores.max(dim=0)  # (H, W)
            
            # 只保留置信度高于阈值的预测
            confidence_threshold = 0.5
            valid_mask = best_scores > confidence_threshold
            
            # 分配类别ID（+1因为0是背景）
            for n in range(num_queries):
                query_mask = (best_indices == n) & valid_mask
                if query_mask.sum() > 0:
                    semantic_seg[b][query_mask] = class_ids[n] + 1
        
        return semantic_seg

    def train_epoch(self, config, train_loader, optimizer, device, epoch):
        self.train()
        iter_loss_list = []
        
        # 只在需要时初始化metrics
        if epoch % 1 == 1 or epoch == 1:  # 第1个epoch和每5个epoch评估一次
            metrics = Evaluator(num_class=config.num_classes)
            calculate_metrics = True
        else:
            calculate_metrics = False
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            # if i==5:
            #     break
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            img_id = batch['img_id']
            
            # 处理mask维度
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            
            optimizer.zero_grad()
            try:
                outputs = self(img, mask)
            except Exception as e:
                msg,idx=e.args
                print(f"{msg} {img_id} in batch {i} sample {idx}")
                continue
            # 计算总损失
            total_loss = 0
            loss_weights = getattr(self.criterion, 'weight_dict', {
                'loss_ce': 1.0, 'loss_mask': 1.0, 'loss_dice': 1.0
            })
            
            for loss_name, loss_value in outputs.items():
                if loss_name.startswith('loss_'):
                    weight = loss_weights.get(loss_name, 1.0)
                    total_loss += weight * loss_value
            
            # 调试信息
            if epoch == 1 and i == 0:
                print(f"# Image shape: {img.shape}, range: [{img.min().item():.3f}, {img.max().item():.3f}]")
                print(f"# Mask shape: {mask.shape}, unique values: {torch.unique(mask).cpu().numpy()}")
                if 'pred_logits' in outputs:
                    print(f"# Pred logits shape: {outputs['pred_logits'].shape}")
                if 'pred_masks' in outputs:
                    print(f"# Pred masks shape: {outputs['pred_masks'].shape}")
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            batch_loss = total_loss.item()
            iter_loss_list.append(batch_loss)
            
            # 更新进度条
            pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
            
            # 记录到wandb
            wandb.log({
                'iteration': epoch * len(train_loader) + i,
                'batch_loss': batch_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 只在指定epoch计算指标
            if calculate_metrics and 'pred_logits' in outputs and 'pred_masks' in outputs:
                pred_semantic = self._convert_to_semantic_segmentation(
                    outputs['pred_logits'], outputs['pred_masks']
                )
                
                # 计算指标
                for j in range(mask.size(0)):
                    metrics.add_batch(mask[j].cpu().numpy(), pred_semantic[j].cpu().numpy())
        
        # 计算平均指标
        avg_loss = np.mean(iter_loss_list) if len(iter_loss_list) > 0 else 0
        
        if calculate_metrics:
            mIoU = np.nanmean(metrics.Intersection_over_Union())
            OA = np.nanmean(metrics.OA())
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")
            wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'train_mIoU': mIoU, 'train_OA': OA})
        else:
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
            wandb.log({'epoch': epoch, 'train_loss': avg_loss})
        
        return avg_loss
    
    def validate_epoch(self, config, val_loader, device=None, max_samples=100):
        """验证函数"""
        self.eval()
        total_loss = 0
        iter_loss_list = []
        metrics = Evaluator(num_class=config.num_classes)
        
        # 简化的采样逻辑
        total_batches = len(val_loader)
        if max_samples and max_samples < total_batches:
            target_batches = max_samples
            print(f"Using first {max_samples}/{total_batches} batches for validation")
        else:
            target_batches = total_batches
            print(f"Using all {total_batches} batches for validation")

        pbar = tqdm(val_loader, desc="Validation", total=target_batches)
        processed_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= target_batches:
                    break
                
                img = batch['img'].to(device)
                mask = batch['gt_semantic_seg'].to(device)
                
                # 处理mask维度
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)
                
                # 前向传播
                outputs = self(img, mask)
                
                # 计算损失
                if isinstance(outputs, dict) and any(k.startswith('loss_') for k in outputs.keys()):
                    # 训练模式：模型直接返回损失
                    batch_loss = 0
                    loss_weights = getattr(self, 'criterion', None)
                    if loss_weights and hasattr(loss_weights, 'weight_dict'):
                        loss_weights = loss_weights.weight_dict
                    else:
                        loss_weights = {'loss_ce': 1.0, 'loss_mask': 1.0, 'loss_dice': 1.0}
                    
                    for loss_name, loss_value in outputs.items():
                        if loss_name.startswith('loss_'):
                            weight = loss_weights.get(loss_name, 1.0)
                            batch_loss += weight * loss_value
                    
                    batch_loss = batch_loss.item()
                elif isinstance(outputs, dict) and 'pred_logits' in outputs and 'pred_masks' in outputs:
                    # 验证模式：需要手动计算损失
                    if hasattr(self, 'criterion'):
                        losses = self.criterion.compute_semantic_loss(outputs, mask)
                        batch_loss = sum(loss.item() for loss in losses.values())
                    else:
                        batch_loss = 0
                else:
                    # 传统分割模型
                    batch_loss = 0
                
                iter_loss_list.append(batch_loss)
                total_loss += batch_loss
                processed_count += 1
                
                # 更新进度条
                pbar.set_postfix({
                    "val_batch_loss": f"{batch_loss:.4f}", 
                    "avg_val_loss": f"{total_loss/processed_count:.4f}",
                    "processed": f"{processed_count}/{target_batches}"
                })
                
                # 计算指标
                if 'pred_logits' in outputs and 'pred_masks' in outputs:
                    # SPSam等实例分割模型
                    pred_semantic = self._convert_to_semantic_segmentation(
                        outputs['pred_logits'], outputs['pred_masks']
                    )
                else:
                    # 传统语义分割模型
                    pred_semantic = outputs.argmax(dim=1)
                
                # 添加到评估器
                for j in range(mask.size(0)):
                    metrics.add_batch(mask[j].cpu().numpy(), pred_semantic[j].cpu().numpy())

        # 计算平均指标
        avg_loss = np.mean(iter_loss_list) if len(iter_loss_list) > 0 else 0
        mIoU = np.nanmean(metrics.Intersection_over_Union())
        OA = np.nanmean(metrics.OA())

        print(f"Validation ({processed_count} batches) - Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")

        return avg_loss, mIoU, OA
    
