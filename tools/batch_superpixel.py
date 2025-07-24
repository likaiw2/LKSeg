import os
import sys
import cv2
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.super_pixel.superpixel import SuperpixelExtractor

def process_single_image(args):
    """处理单张图片的工作函数"""
    img_file, input_path, output_folder, algorithm, custom_params, save_visualization = args
    
    try:
        # 计算相对路径，保持目录结构
        relative_path = img_file.relative_to(input_path)
        output_dir = Path(output_folder) / relative_path.parent
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            return f"无法读取图片: {img_file}"
            
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor格式 [1, C, H, W]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # 初始化superpixel提取器（每个进程独立初始化）
        extractor = SuperpixelExtractor(algorithm)
        
        # 生成superpixel
        _, _, _, assigned_masks = extractor(img_tensor, custom_params)
        
        # 获取superpixel标签矩阵 [H, W]
        superpixel_labels = assigned_masks[0].numpy().astype(np.int32)
        
        # 保存结果到对应位置
        output_file = output_dir / f"{img_file.stem}_superpixel.npy"
        np.save(output_file, superpixel_labels)
        
        # 可选：保存可视化结果
        if save_visualization:
            from skimage.segmentation import mark_boundaries
            vis_img = img / 255.0
            boundaries = mark_boundaries(vis_img, superpixel_labels)
            vis_output = output_dir / f"{img_file.stem}_vis.png"
            cv2.imwrite(str(vis_output), (boundaries * 255).astype(np.uint8)[:,:,::-1])
        
        return f"成功处理: {img_file.name}"
        
    except Exception as e:
        return f"处理 {img_file} 时出错: {e}"

def process_images_batch_parallel(input_folder, output_folder, algorithm="slic", 
                                custom_params=None, num_workers=None, save_visualization=True):
    """
    并行处理输入文件夹下的所有图片生成superpixel结果
    
    Args:
        input_folder (str): 输入图片文件夹路径
        output_folder (str): 输出结果文件夹路径
        algorithm (str): superpixel算法
        custom_params (dict): 自定义参数
        num_workers (int): 并行进程数，None表示使用CPU核心数
        save_visualization (bool): 是否保存可视化结果
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 递归获取所有PNG图片文件
    input_path = Path(input_folder)
    image_files = list(input_path.rglob("*.png"))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到PNG图片文件")
        return
    
    # 设置进程数
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(image_files))
    
    print(f"找到 {len(image_files)} 张PNG图片，使用 {algorithm} 算法，{num_workers} 个进程并行处理...")
    
    # 准备参数列表
    args_list = [(img_file, input_path, output_folder, algorithm, custom_params, save_visualization) 
                 for img_file in image_files]
    
    # 使用进程池并行处理
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, args_list),
            total=len(image_files),
            desc="处理图片"
        ))
    
    # 打印结果统计
    success_count = sum(1 for r in results if "成功处理" in r)
    error_count = len(results) - success_count
    
    print(f"处理完成！成功: {success_count}, 失败: {error_count}")
    print(f"结果保存在 {output_folder}")
    
    # 打印错误信息
    for result in results:
        if "成功处理" not in result:
            print(result)

def process_images_batch_gpu_batched(input_folder, output_folder, algorithm="slic", 
                                   custom_params=None, batch_size=8, device="cuda:0"):
    """
    GPU批处理版本（如果SuperpixelExtractor支持批处理）
    """
    os.makedirs(output_folder, exist_ok=True)
    
    input_path = Path(input_folder)
    image_files = list(input_path.rglob("*.png"))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到PNG图片文件")
        return
    
    # 设置设备
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(device)
    else:
        device = "cpu"
        print("CUDA不可用，使用CPU处理")
    
    extractor = SuperpixelExtractor(algorithm)
    
    print(f"找到 {len(image_files)} 张PNG图片，使用 {algorithm} 算法，批大小 {batch_size}，设备 {device}")
    
    # 批处理
    for i in tqdm(range(0, len(image_files), batch_size), desc="批处理"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_paths = []
        
        # 加载批次图片
        for img_file in batch_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                batch_images.append(img_tensor)
                batch_paths.append(img_file)
        
        if not batch_images:
            continue
            
        # 批处理
        try:
            batch_tensor = torch.stack(batch_images).to(device)
            _, _, _, assigned_masks = extractor(batch_tensor, custom_params)
            
            # 保存结果
            for j, img_file in enumerate(batch_paths):
                relative_path = img_file.relative_to(input_path)
                output_dir = Path(output_folder) / relative_path.parent
                os.makedirs(output_dir, exist_ok=True)
                
                superpixel_labels = assigned_masks[j].cpu().numpy().astype(np.int32)
                output_file = output_dir / f"{img_file.stem}_superpixel.npy"
                np.save(output_file, superpixel_labels)
                
        except Exception as e:
            print(f"批处理出错: {e}")
            # 回退到单张处理
            for img_file in batch_paths:
                try:
                    process_single_image((img_file, input_path, output_folder, 
                                        algorithm, custom_params, False))
                except Exception as e2:
                    print(f"单张处理 {img_file} 出错: {e2}")

def main():
    # 配置参数
    input_folder = "/home/likai/code/LKSeg/data/LoveDA"
    output_folder = "/home/likai/code/LKSeg/temp/superpixel/LoveDA"
    
    algorithm = "slic"
    slic_parameters_dict = {
        "n_segments": 100,
        "compactness": 20,
        "sigma": 1,
        "start_label": 0,
        "min_size_factor": 0.5,
        "max_num_iter": 10,
        "enforce_connectivity": True,
    }
    
    # 使用CPU多进程并行处理
    process_images_batch_parallel(
        input_folder=input_folder,
        output_folder=output_folder, 
        algorithm=algorithm,
        custom_params=slic_parameters_dict,
        num_workers=8,  # 可以调整进程数，None表示自动使用CPU核心数
        save_visualization=True  # 是否保存可视化结果
    )

if __name__ == "__main__":
    main()
