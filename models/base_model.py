import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tools.metric import Evaluator
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn.functional as F

class BaseSegmentationModel(nn.Module, ABC):
    """分割模型基类"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass
    
    def train_one_epoch(self, config, train_loader, optimizer, loss_fn, device, epoch):
        """训练一个epoch的基础实现"""
        self.train()
        iter_loss_list = []
        metrics = Evaluator(num_class=config.num_classes)
        first_batch = True
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            
            optimizer.zero_grad()
            
            # 调用模型特定的前向传播和损失计算
            loss, pred_semantic = self._compute_loss_and_pred(img, mask, loss_fn, first_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            # 记录和更新
            batch_loss = loss.item()
            iter_loss_list.append(batch_loss)
            pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
            
            wandb.log({
                'iteration': epoch * len(train_loader) + i,
                'batch_loss': batch_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 计算指标
            for j in range(mask.size(0)):
                metrics.add_batch(mask[j].cpu().numpy(), pred_semantic[j].cpu().numpy())
            
            first_batch = False
        
        # 计算平均指标
        avg_loss = np.mean(iter_loss_list) if len(train_loader) > 0 else 0
        mIoU = np.nanmean(metrics.Intersection_over_Union())
        OA = np.nanmean(metrics.OA())
        
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")
        wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'train_mIoU': mIoU, 'train_OA': OA})
        
        return avg_loss
    
    def validate(self, config, val_loader, loss_fn, device):
        """验证函数"""
        self.eval()
        total_loss = 0
        metrics = Evaluator(num_class=config.num_classes)
        
        pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                img = batch['img'].to(device)
                mask = batch['gt_semantic_seg'].to(device)
                
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)

                # 调用模型特定的验证逻辑
                batch_loss, pred_semantic = self._validate_batch(img, mask, loss_fn)
                
                if not torch.isfinite(torch.tensor(batch_loss)):
                    print(f"Warning: Non-finite validation loss at batch {i}, value: {batch_loss}")
                    continue
                    
                total_loss += batch_loss
                pbar.set_postfix({"val_batch_loss": f"{batch_loss:.4f}", "avg_val_loss": f"{total_loss/(i+1):.4f}"})

                # 计算指标
                for j in range(mask.size(0)):
                    metrics.add_batch(mask[j].cpu().numpy(), pred_semantic[j].cpu().numpy())

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        mIoU = np.nanmean(metrics.Intersection_over_Union())
        OA = np.nanmean(metrics.OA())
        
        print(f"Validation - Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")
        
        return avg_loss, mIoU, OA
    
    def _validate_batch(self, img, mask, loss_fn):
        """默认的验证批次处理，子类可以重写"""
        outputs = self(img)
        
        # 标准tensor输出
        if isinstance(outputs, torch.Tensor):
            pred_logits = outputs
            pred_semantic = outputs.argmax(dim=1)
        else:
            # 如果是其他格式，子类应该重写此方法
            raise NotImplementedError("Subclass should implement _validate_batch for non-tensor outputs")
        
        # 计算损失
        loss = loss_fn(pred_logits, mask)
        return loss.item(), pred_semantic
    
    @abstractmethod
    def _compute_loss_and_pred(self, img, mask, loss_fn, first_batch):
        """计算损失和预测结果，子类需要实现"""
        pass
