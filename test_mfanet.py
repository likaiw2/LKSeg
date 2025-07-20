import os
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from tools.cfg import py2cfg
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", 
        default='config/loveda-mfanet.py')
    arg("-w", "--weights_path", type=Path, help="Path to the model weights.",
        default='/home/liw324/code/Segment/LKSeg/out/model_weights/loveda/mfanet_20250706_124606/mfanet_best.pth')
    arg("-o", "--output_dir", type=Path, help="Directory to save results",
        default='results')
    return parser.parse_args()

def test(config, model, loader, device, output_dir):
    model.eval()
    metrics = Evaluator(num_class=config.num_classes)
    
    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    class_names = config.CLASSES if hasattr(config, 'CLASSES') else [f'Class {i}' for i in range(config.num_classes)]
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Testing")):
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            img_id = batch['img_id']
            img_type = batch['img_type']
            
            # 处理掩码维度（如果需要）
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            
            # 前向传播
            outputs = model(img)
            
            # 获取预测掩码
            pred_mask = outputs.argmax(dim=1)
            
            # 可视化每张图片
            for j in range(img.size(0)):
                # 获取单张图片数据
                single_img = np.array(Image.open(f"/home/liw324/code/Segment/LKSeg/data/LoveDA/Val/{img_type[j]}/images_png/{img_id[j]}.png").convert('RGB'))
                single_mask = mask[j].cpu().numpy()
                single_pred = pred_mask[j].cpu().numpy()
                
                # 转换图片为可视化格式 (C,H,W) -> (H,W,C)
                # img_vis = single_img.permute(1, 2, 0).numpy()
                # 如果图片是归一化的，需要反归一化
                # if img_vis.max() <= 3.0:
                #     img_vis = (img_vis * 255).astype(np.uint8)
                
                # 创建可视化
                fig, axes = plt.subplots(1, 3, figsize=(15, 10))
                
                axes[0].imshow(single_img)
                axes[0].set_title(f'Image {img_id[j]}')
                axes[0].axis('off')
                
                axes[1].imshow(single_mask, cmap='tab10')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(single_pred, cmap='tab10')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'predictions', f'{img_id[j]}_comparison.png'))
                plt.close()
                
                print(f"Saved visualization for {img_id[j]}")
            
            # 计算指标
            for j in range(mask.size(0)):
                gt_np = mask[j].cpu().numpy()
                pred_np = pred_mask[j].cpu().numpy()
                metrics.add_batch(gt_np, pred_np)
            if i >9:
                break
    # 计算指标
    IoU = metrics.Intersection_over_Union()
    mIoU = np.nanmean(IoU)
    OA = np.nanmean(metrics.OA())
    # F1 = metrics.F1Score()
    # mF1 = np.nanmean(F1)
    
    # 打印结果
    print(f"Test Results:")
    print(f"mIoU: {mIoU:.4f}")
    print(f"OA: {OA:.4f}")
    # print(f"mF1: {mF1:.4f}")
    
    # 打印每个类别的IoU
    print("\nPer-class IoU:")
    for i, class_iou in enumerate(IoU):
        print(f"{class_names[i]}: {class_iou:.4f}")
    
    # 保存指标到CSV
    results = {
        'Metric': ['mIoU', 'OA', 'mF1'] + [f'IoU_{class_names[i]}' for i in range(len(IoU))],
        'Value': [mIoU, OA] + [iou for iou in IoU]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    return mIoU, OA


def main():
    args = get_args()
    
    config = py2cfg(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.net.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    print(f"Loaded model weights from {args.weights_path}")
    
    # 使用验证集作为测试集（如果没有专门的测试集）
    test_loader = config.test_loader if hasattr(config, 'test_loader') else config.val_loader
    
    # 运行测试
    output_dir = args.output_dir
    mIoU, OA = test(config, model, test_loader, device, output_dir)
    
    print(f"\nTest completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
