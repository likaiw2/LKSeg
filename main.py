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
import torch.nn.functional as F

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = "online"
# os.environ["WANDB_MODE"] = "offline"


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
        default='config/loveda-spsam.py',)
    arg("-m", "--mode", type=str, help="train or test", 
        default='train')
    arg("-w", "--weights_path", type=Path, help="Path to the model weights for testing.",
        default=None)
    return parser.parse_args()

def train_mode(config, device):
    """训练模式"""
    best_mIoU = 0
    best_ckpt_path = ""
    
    # 初始化 wandb
    wandb.init(project=config.log_name, 
               config=vars(config),    
               )

    model = config.net.to(device)
    if config.pretrained_ckpt_path:
        model.load_state_dict(torch.load(config.pretrained_ckpt_path, map_location=device))

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
        
        # 调用模型的训练方法
        train_loss = model.train_epoch(config, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 定期验证
        if epoch % config.check_val_every_n_epoch == 0:
            val_loss, mIoU, OA = model.validate_epoch(config, val_loader, device, max_samples=50)
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
        
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = os.path.join(metrics_dir, "training_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)

    wandb.finish()
    print(f"Training completed. Best model: {best_ckpt_path} with mIoU={best_mIoU:.4f}")

def test_mode(config, device, weights_path):
    """测试模式"""
    model = config.net.to(device)
    
    # 加载模型权重
    if weights_path and weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")
    elif config.pretrained_ckpt_path:
        model.load_state_dict(torch.load(config.pretrained_ckpt_path, map_location=device))
        print(f"Loaded model weights from {config.pretrained_ckpt_path}")
    else:
        print("Warning: No model weights specified for testing!")
        return
    
    # 使用测试集或验证集
    test_loader = config.test_loader if hasattr(config, 'test_loader') else config.val_loader
    
    # 使用配置文件中的输出目录
    output_dir = Path(config.weights_path) / "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行测试
    print("Starting testing...")
    model.eval()
    metrics = Evaluator(num_class=config.num_classes)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for i, batch in enumerate(pbar):
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            
            # 获取预测结果
            outputs = model(img)
            
            # 处理不同类型的输出
            if hasattr(model, '_validate_batch'):
                _, pred_semantic = model._validate_batch(img, mask)
            else:
                if isinstance(outputs, torch.Tensor):
                    pred_semantic = outputs.argmax(dim=1)
                else:
                    raise NotImplementedError("Model should implement _validate_batch for non-tensor outputs")
            
            # 计算指标
            for j in range(mask.size(0)):
                metrics.add_batch(mask[j].cpu().numpy(), pred_semantic[j].cpu().numpy())
            
            # 可选：保存预测结果
            if i < 10:  # 只保存前10个样本的可视化结果
                save_predictions(img, mask, pred_semantic, output_dir, i)
    
    # 计算最终指标
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    mF1 = np.nanmean(metrics.F1())
    
    print(f"\nTest Results:")
    print(f"mIoU: {mIoU:.4f}")
    print(f"OA: {OA:.4f}")
    print(f"mF1: {mF1:.4f}")
    
    # 保存测试结果
    results = {
        'mIoU': mIoU,
        'OA': OA,
        'mF1': mF1,
        'per_class_IoU': metrics.Intersection_over_Union().tolist(),
        'per_class_F1': metrics.F1().tolist()
    }
    
    import json
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test completed. Results saved to {output_dir}")
    return mIoU, OA, mF1

def save_predictions(img, mask, pred, output_dir, batch_idx):
    """保存预测结果的可视化"""
    import matplotlib.pyplot as plt
    
    batch_size = img.shape[0]
    for i in range(min(batch_size, 2)):  # 每个batch最多保存2张图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        img_np = img[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 真实标签
        axes[1].imshow(mask[i].cpu().numpy(), cmap='tab10')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # 预测结果
        axes[2].imshow(pred[i].cpu().numpy(), cmap='tab10')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_batch{batch_idx}_sample{i}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    args = get_args()
    
    config = py2cfg(args.config_path)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train':
        print("Starting training mode...")
        train_mode(config, device)
    elif args.mode == 'test':
        print("Starting testing mode...")
        test_mode(config, device, args.weights_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
