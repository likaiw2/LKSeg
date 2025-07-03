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

os.environ["WANDB_MODE"] = "offline"


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
        default='config/loveda-spsam.py',
        )
    return parser.parse_args()

class Supervision_Train(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    metrics = Evaluator(num_class=model.config.num_classes)
    first_batch = True
    for batch in tqdm(loader, desc=f"Train Epoch {epoch}"):
        img = batch['img'].to(device)
        mask = batch['gt_semantic_seg'].to(device)

        # Handle mask dimensions - remove extra channel dimension if present
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)  # Remove channel dimension: [B, 1, H, W] -> [B, H, W]

        optimizer.zero_grad()
        outputs = model(img)

        # SPSam outputs a dictionary with pred_logits and pred_masks
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]    # [B, Q, H, W]

        # Debug information for first batch
        if first_batch:
            print(f"Image shape: {img.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Pred logits shape: {pred_logits.shape}")
            print(f"Pred masks shape: {pred_masks.shape}")
            first_batch = False

        # Calculate loss - using a specialized loss for query-based segmentation
        loss = loss_fn(pred_logits, pred_masks, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Convert query predictions to semantic segmentation format for metrics
        # This requires selecting the best mask for each pixel
        pred_mask = get_semantic_seg_from_query_outputs(pred_logits, pred_masks)
        
        for i in range(mask.size(0)):
            metrics.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

    avg_loss = total_loss / len(loader)
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    wandb.log({'train_loss': avg_loss, 'train_mIoU': mIoU, 'train_OA': OA})
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

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics = Evaluator(num_class=model.config.num_classes)
    with torch.no_grad():
        for batch in loader:
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            
            # Handle mask dimensions if needed
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

            outputs = model(img)
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs["pred_masks"]
            
            # Calculate loss
            loss = loss_fn(pred_logits, pred_masks, mask)
            total_loss += loss.item()

            # Convert to semantic segmentation format for metrics
            pred_mask = get_semantic_seg_from_query_outputs(pred_logits, pred_masks)
            
            for i in range(mask.size(0)):
                metrics.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

    avg_loss = total_loss / len(loader)
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    wandb.log({'val_loss': avg_loss, 'val_mIoU': mIoU, 'val_OA': OA})
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
    model = Supervision_Train(config).to(device)
    if config.pretrained_ckpt_path:
        model.load_state_dict(torch.load(config.pretrained_ckpt_path, map_location=device))

    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler
    train_loader = config.train_loader
    val_loader = config.val_loader

    # 初始化验证指标
    val_loss = 0
    mIoU = 0
    OA = 0
    
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
        train_loss = train_one_epoch(model, train_loader, optimizer, config.loss, device, epoch)
        train_losses.append(train_loss)
        
        if epoch % config.check_val_every_n_epoch == 0:
            val_loss, mIoU, OA = validate(model, val_loader, config.loss, device)
            val_losses.append(val_loss)
            val_mious.append(mIoU)
            val_oas.append(OA)
        else:
            # 如果这个epoch没有验证，使用上一个值填充
            if val_losses:
                val_losses.append(val_losses[-1])
                val_mious.append(val_mious[-1])
                val_oas.append(val_oas[-1])
            else:
                val_losses.append(0)
                val_mious.append(0)
                val_oas.append(0)

        if lr_scheduler is not None:
            lr_scheduler.step()
            # 记录当前学习率
            current_lr = lr_scheduler.get_last_lr()[0]
            wandb.log({'learning_rate': current_lr}, step=epoch)

        # 保存最新权重
        os.makedirs(config.weights_path, exist_ok=True)
        latest_ckpt_path = os.path.join(config.weights_path, f"{config.weights_name}_latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        # 保存最佳权重
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_ckpt_path = os.path.join(config.weights_path, f"{config.weights_name}_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved at {best_ckpt_path} with mIoU={best_mIoU:.4f}")
        
        # 保存指标到文件
        metrics_data = {
            'epoch': list(range(1, epoch + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_mIoU': val_mious,
            'val_OA': val_oas
        }
        metrics_path = os.path.join(metrics_dir, "training_metrics.npz")
        np.savez(metrics_path, **metrics_data)
        
        # 绘制loss曲线
        if epoch > 1:  # 至少有两个点才能绘制曲线
            plt.figure(figsize=(12, 8))
            
            # 绘制训练和验证loss
            plt.subplot(2, 1, 1)
            plt.plot(range(1, epoch + 1), train_losses, 'b-', label='Train Loss')
            plt.plot(range(1, epoch + 1), val_losses, 'r-', label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.grid(True)
            
            # 绘制验证mIoU和OA
            plt.subplot(2, 1, 2)
            plt.plot(range(1, epoch + 1), val_mious, 'g-', label='Val mIoU')
            plt.plot(range(1, epoch + 1), val_oas, 'y-', label='Val OA')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.title('Validation Metrics')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            plot_path = os.path.join(metrics_dir, f"metrics_plot_epoch_{epoch}.png")
            plt.savefig(plot_path)
            
            # 上传图像到wandb
            wandb.log({"metrics_plot": wandb.Image(plot_path)}, step=epoch)
            
            plt.close()

        print(f"Epoch {epoch} done. train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mIoU={mIoU:.4f}, val_OA={OA:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
