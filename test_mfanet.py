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
    
    # 创建进度条
    pbar = tqdm(loader, desc="Testing")
    
    class_names = config.CLASSES if hasattr(config, 'CLASSES') else [f'Class {i}' for i in range(config.num_classes)]
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)
            
            # 处理掩码维度（如果需要）
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            
            # 前向传播
            outputs = model(img)
            
            # 获取预测掩码
            pred_mask = outputs.argmax(dim=1)
            
            # 计算指标
            for j in range(mask.size(0)):
                gt_np = mask[j].cpu().numpy()
                pred_np = pred_mask[j].cpu().numpy()
                metrics.add_batch(gt_np, pred_np)
                
                # 可选：保存预测结果可视化
                if i < 10:  # 只保存前10个批次的结果
                    # 创建可视化图像
                    vis_img = visualize_prediction(
                        img[j].cpu().numpy().transpose(1, 2, 0),
                        gt_np,
                        pred_np,
                        class_names
                    )
                    
                    # 保存可视化结果
                    save_path = os.path.join(output_dir, 'predictions', f'sample_{i}_{j}.png')
                    plt.imsave(save_path, vis_img)
    
    # 计算指标
    IoU = metrics.Intersection_over_Union()
    mIoU = np.nanmean(IoU)
    OA = np.nanmean(metrics.OA())
    F1 = metrics.F1Score()
    mF1 = np.nanmean(F1)
    
    # 打印结果
    print(f"Test Results:")
    print(f"mIoU: {mIoU:.4f}")
    print(f"OA: {OA:.4f}")
    print(f"mF1: {mF1:.4f}")
    
    # 打印每个类别的IoU
    print("\nPer-class IoU:")
    for i, class_iou in enumerate(IoU):
        print(f"{class_names[i]}: {class_iou:.4f}")
    
    # 保存指标到CSV
    results = {
        'Metric': ['mIoU', 'OA', 'mF1'] + [f'IoU_{class_names[i]}' for i in range(len(IoU))],
        'Value': [mIoU, OA, mF1] + [iou for iou in IoU]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    return mIoU, OA, mF1

def visualize_prediction(image, ground_truth, prediction, class_names):
    """
    创建一个可视化图像，显示原始图像、真实标签和预测结果
    """
    # 创建颜色映射
    n_classes = len(class_names)
    colors = plt.cm.get_cmap('tab20', n_classes)
    
    # 归一化图像
    if image.max() > 1:
        image = image / 255.0
    
    # 创建分割掩码的彩色表示
    gt_colored = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3))
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    
    for i in range(n_classes):
        gt_colored[ground_truth == i] = colors(i)[:3]
        pred_colored[prediction == i] = colors(i)[:3]
    
    # 创建组合图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 创建图例
    patches = [plt.Rectangle((0, 0), 1, 1, color=colors(i)[:3]) for i in range(n_classes)]
    plt.legend(patches, class_names, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=min(5, n_classes), mode='expand')
    
    # 保存图像并返回
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_img

def main():
    args = get_args()
    
    config = py2cfg(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.net.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    print(f"Loaded model weights from {args.weights_path}")
    
    # 使用验证集作为测试集（如果没有专门的测试集）
    test_loader = config.test_loader if hasattr(config, 'test_loader') else config.val_loader
    
    # 运行测试
    output_dir = args.output_dir
    mIoU, OA, mF1 = test(config, model, test_loader, device, output_dir)
    
    print(f"\nTest completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()