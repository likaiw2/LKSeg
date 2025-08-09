import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
import random
import wandb
from tools.cfg import py2cfg
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_MODE"] = "online"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", 
        # required=True,
        # default='config/loveda/sfanet.py',
        # default='config/earthvqa-sfanet.py',
        # default='config/loveda-spsam.py',
        # default='config/loveda-mfanet.py',
        default='config/loveda-sp_mask2former.py',
        )
    return parser.parse_args()

def train_one_epoch(config, model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    iter_loss_list = []
    metrics = Evaluator(num_class=config.num_classes)
    first_batch = True
    
    # 创建进度条
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    
    for i, batch in enumerate(pbar):
        img = batch['image'].to(device).float()  # 添加.float()
        mask = batch['semantic_mask'].to(device)
        
        # 打印第一个批次的形状和值范围
        if first_batch:
            print(f"# Image shape: {img.shape}, range: [{img.min().item()}, {img.max().item()}]")
            print(f"# Mask shape: {mask.shape}, range: [{mask.min().item()}, {mask.max().item()}], unique values: {torch.unique(mask).cpu().numpy()}")
            # print(f"# Output shape: {outputs.shape}, range: [{outputs.min().item()}, {outputs.max().item()}]")
            first_batch = False

        optimizer.zero_grad()
        
        # 构造SP_Mask2Former期望的批量输入格式
        batched_inputs = []
        for b in range(img.shape[0]):
            sample_input = {
                "image": img[b].float(),
                "height": img.shape[2],
                "width": img.shape[3],
            }
            
            # 添加超像素掩码
            if 'superpixel_mask' in batch and batch['superpixel_mask'] is not None:
                sample_input["superpixel_mask"] = batch['superpixel_mask'][b]
            
            # 如果训练时需要ground truth，构造instances格式
            if model.training:
                gt_mask = mask[b]  # [H, W]
                unique_classes = torch.unique(gt_mask)
                unique_classes = unique_classes[unique_classes != config.ignore_index if hasattr(config, 'ignore_index') else unique_classes != 255]
                
                # 为每个类别创建二进制掩码
                gt_masks_list = []
                gt_classes_list = []
                
                for cls in unique_classes:
                    if hasattr(config, 'ignore_index') and cls == config.ignore_index:
                        continue
                    if cls == 255:  # 默认忽略值
                        continue
                    binary_mask = (gt_mask == cls).float()
                    if binary_mask.sum() > 0:  # 只保留非空掩码
                        gt_masks_list.append(binary_mask)
                        gt_classes_list.append(cls)
                
                if len(gt_masks_list) > 0:
                    # 创建mock instances对象
                    instance = type('MockInstance', (), {})()
                    instance.gt_masks = torch.stack(gt_masks_list)
                    instance.gt_classes = torch.tensor(gt_classes_list, device=device)
                else:
                    # 空实例
                    instance = type('MockInstance', (), {})()
                    instance.gt_masks = torch.zeros((0, mask.shape[1], mask.shape[2]), device=device)
                    instance.gt_classes = torch.zeros((0,), dtype=torch.long, device=device)
                
                sample_input["instances"] = instance
            
            batched_inputs.append(sample_input)

        # 模型前向传播
        outputs = model(batched_inputs)
        
        # 处理损失
        if model.training:
            # 训练时outputs是损失字典
            loss = sum(outputs.values())
        else:
            # 推理时需要转换输出格式并计算损失
            batch_preds = []
            for sample_output in outputs:
                if 'sem_seg' in sample_output:
                    batch_preds.append(sample_output['sem_seg'])
                elif 'pred_logits' in sample_output and 'pred_masks' in sample_output:
                    pred_logits = sample_output['pred_logits'].unsqueeze(0)
                    pred_masks = sample_output['pred_masks'].unsqueeze(0)
                    sem_seg = get_semantic_seg_from_query_outputs(pred_logits, pred_masks)
                    batch_preds.append(sem_seg.squeeze(0))
                else:
                    batch_preds.append(torch.zeros_like(mask[0]))
            
            outputs = torch.stack(batch_preds)
            loss = loss_fn(outputs, mask)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        batch_loss = loss.item()
        iter_loss_list.append(batch_loss)
        
        # 更新进度条显示当前批次的损失
        pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
        
        # 记录每个iteration的损失到wandb
        wandb.log({
            'iteration': epoch * len(loader) + i,
            'batch_loss': batch_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # 使用detach()分离梯度，然后再转换为NumPy数组
        pred_mask = outputs.detach()
        
        # 确保预测掩码的形状与真实标签相同
        # 假设outputs的形状是[batch_size, num_classes, height, width]
        pred_classes = pred_mask.argmax(dim=1)  # 形状变为[batch_size, height, width]
        
        for j in range(mask.size(0)):
            # 确保形状匹配
            gt_np = mask[j].cpu().numpy()
            pred_np = pred_classes[j].cpu().numpy()
            
            # 打印形状以进行调试
            if i == 0 and j == 0:
                print(f"# GT shape: {gt_np.shape}, Pred shape: {pred_np.shape}")
                print(f"# GT unique values: {np.unique(gt_np)}")
            
            metrics.add_batch(gt_np, pred_np)

    # 计算平均损失和指标
    avg_loss = np.mean(iter_loss_list) if len(loader) > 0 else 0
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    
    # 打印epoch摘要
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")
    
    # 记录每个epoch的指标到wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': avg_loss, 
        'train_mIoU': mIoU, 
        'train_OA': OA
    })
    
    return avg_loss

def get_semantic_seg_from_query_outputs(pred_logits, pred_masks):
    """
    Convert query-based predictions to semantic segmentation format.
    
    Args:
        pred_logits: Tensor of shape [B, Q, C+1] with class predictions for each query
        pred_masks: Tensor of shape [B, Q, H, W] with mask predictions for each query
        
    Returns:
        Tensor of shape [B, H, W] with class predictions for each pixel
    """
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]
    num_classes = pred_logits.shape[2] - 1  # Subtract 1 for "no object" class
    
    # Remove the "no object" class (usually the last one)
    class_probs = torch.softmax(pred_logits, dim=2)[:, :, :-1]  # [B, Q, C]
    
    # Get the highest probability class for each query
    query_class = class_probs.argmax(dim=2)  # [B, Q]
    
    # Get the confidence score for each query's predicted class
    query_score = torch.gather(
        class_probs, 
        2, 
        query_class.unsqueeze(2)
    ).squeeze(2)  # [B, Q]
    
    # Apply sigmoid to mask predictions
    mask_probs = torch.sigmoid(pred_masks)  # [B, Q, H, W]
    
    # Initialize output segmentation map
    height, width = pred_masks.shape[2:]
    segmentation = torch.zeros((batch_size, height, width), device=pred_masks.device, dtype=torch.long)
    
    # For each batch item
    for b in range(batch_size):
        # Initialize a map to track the highest confidence at each pixel
        pixel_confidence = torch.zeros((height, width), device=pred_masks.device)
        
        # For each query
        for q in range(num_queries):
            # Skip if this is a low confidence prediction
            if query_score[b, q] < 0.5:  # Confidence threshold
                continue
                
            # Get class and mask for this query
            cls = query_class[b, q]
            mask = mask_probs[b, q] * query_score[b, q]  # Weight by class confidence
            
            # Update segmentation where this mask is more confident
            update_mask = mask > pixel_confidence
            segmentation[b][update_mask] = cls
            pixel_confidence[update_mask] = mask[update_mask]
    
    return segmentation

def validate(config, model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics = Evaluator(num_class=config.num_classes)
    
    # 创建验证进度条
    pbar = tqdm(loader, desc="Validation")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            img = batch['image'].to(device)
            mask = batch['semantic_mask'].to(device)
            
            # 处理掩码维度（如果需要）
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

            outputs = model(img)
            
            # 计算损失
            loss = loss_fn(outputs, mask)
            batch_loss = loss.item()
            
            # 检查损失是否为NaN
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite validation loss at batch {i}, value: {batch_loss}")
                quit()
                
            total_loss += batch_loss
            
            # 更新进度条
            pbar.set_postfix({"val_batch_loss": f"{batch_loss:.4f}", "avg_val_loss": f"{total_loss/(i+1):.4f}"})

            # 转换为语义分割格式以计算指标
            pred_mask = outputs.argmax(dim=1)
            
            for j in range(mask.size(0)):
                metrics.add_batch(mask[j].cpu().numpy(), pred_mask[j].cpu().numpy())

    # 计算平均损失和指标
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    
    # 打印验证摘要
    print(f"Validation - Loss: {avg_loss:.4f}, mIoU: {mIoU:.4f}, OA: {OA:.4f}")
    
    return avg_loss, mIoU, OA

def main():
    args = get_args()
    
    best_mIoU = 0
    best_ckpt_path = ""

    config = py2cfg(args.config_path)
    
    seed_everything(42)

    # 初始化 wandb
    wandb.init(project=config.log_name, 
               config=vars(config),    
               )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.net
    if config.pretrained_ckpt_path:
        model.load_state_dict(torch.load(config.pretrained_ckpt_path, map_location=device))

    model = model.to(device)

    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler
    train_loader = config.train_loader
    val_loader = config.val_loader

    # 创建用于跟踪指标的列表
    train_losses = []
    val_losses = []
    val_mious = []
    val_oas = []
    
    # 创建保存指标的目录
    metrics_dir = os.path.join(config.weights_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # 训练/验证循环
    for epoch in range(1, config.max_epoch + 1):
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{config.max_epoch} - Learning rate: {current_lr:.6f}")
        
        train_loss = train_one_epoch(config, model, train_loader, optimizer, config.loss, device, epoch)
        train_losses.append(train_loss)
        
        # 定期验证
        if epoch % config.check_val_every_n_epoch == 0:
            val_loss, mIoU, OA = validate(config, model, val_loader, config.loss, device)
            val_losses.append(val_loss)
            val_mious.append(mIoU)
            val_oas.append(OA)
            
            # 记录验证指标
            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss, 
                'val_mIoU': mIoU, 
                'val_OA': OA
            })
            
            # 保存最新权重
            os.makedirs(config.weights_path, exist_ok=True)
            latest_ckpt_path = os.path.join(config.weights_path, f"{config.model_name}_latest.pth")
            torch.save(model.state_dict(), latest_ckpt_path)

            # 保存最佳权重
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                best_ckpt_path = os.path.join(config.weights_path, f"{config.model_name}_best.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best model saved at {best_ckpt_path} with mIoU={best_mIoU:.4f}")
            

        if lr_scheduler is not None:
            lr_scheduler.step()


        # 保存指标到文件
        metrics_data = {
            'epoch': list(range(1, epoch + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_mIoU': val_mious,
            'val_OA': val_oas
        }
        
        # 使用CSV格式保存指标
        
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = os.path.join(metrics_dir, "training_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
