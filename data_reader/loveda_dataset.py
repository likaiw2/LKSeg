# -*- coding: utf-8 -*-

import sys
import os
# Add parent directory to path for importing models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_reader.transform import Compose,RandomCrop,PadImage,RandomHorizontalFlip,RandomVerticalFlip,Resize,RandomScale,ColorJitter,SmartCropV1,SmartCropV2
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import DataLoader
import cv2
from skimage.segmentation import mark_boundaries

class LoveDATrainDataset(Dataset):
    def __init__(self, 
                 data_root='data/LoveDA/Train', 
                 img_dir='images_png', 
                 mask_dir='masks_png', 
                 img_suffix='.png', 
                 mask_suffix='.png',
                 superpixel=False,
                 superpixel_dict=None,
                 superpixel_type=None,
                 transform=None, 
                 test_mode=False,
                 original_size=[1024,1024],
                 output_size=[512,512]):
        
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.superpixel = superpixel
        self.transform = transform
        self.test_mode = test_mode
        self.original_size = original_size
        self.img_size = output_size
        self.img_ids = self._collect_img_id_region()
        
        if self.superpixel:
            if superpixel_dict is not None:
                self.superpixel_type = superpixel_type
                self.superpixel_dict = superpixel_dict
            else:
                self.superpixel_type = "slic"
                self.superpixel_dict = {
                    "n_segments": 100,
                    "compactness": 20,
                    "sigma": 1,
                    "start_label": 0,
                    "min_size_factor": 0.5,
                    "max_num_iter": 10,
                    "enforce_connectivity": True,
                }
        
        self.COLOR_MAP = dict(
            nothing=(0, 0, 0),              # 0 black
            Background=(255, 255, 255),     # 1 white
            Building=(255, 0, 0),           # 2 red
            Road=(255, 255, 0),             # 3 yellow
            Water=(0, 0, 255),              # 4 blue
            Barren=(159, 129, 183),         # 5 purple
            Forest=(0, 255, 0),             # 6 green
            Agricultural=(255, 195, 128),   # 7 orange
        )
        self.CLASSES = list(self.COLOR_MAP.keys())
        self.PALETTE = list(self.COLOR_MAP.values())



    def __getitem__(self, index):

        img, mask = self._load_image_and_mask(index)

        # apply data augmentation
        if self.test_mode:
            img, mask = img.resize(self.original_size), mask.resize(self.original_size)
        else:
            if self.transform:
                img, mask = self.transform(img, mask)
            else:
                # use normal transform
                self.transform = Compose([
                    RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                    SmartCropV1(crop_size=self.img_size[0], max_ratio=0.75, ignore_index=0, nopad=False),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    PadImage(self.img_size, ignore_index=0),
                ])
                img, mask = self.transform(img, mask)

        # generate superpixel mask
        if not self.superpixel:
            sp_input = None
        else:
            from models.super_pixel.superpixel import SuperpixelExtractor
            
            # 将PIL图像转换为tensor格式 [1, C, H, W]
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
            
            # 初始化superpixel提取器
            extractor = SuperpixelExtractor(self.superpixel_type)
            _, _, _, assigned_masks = extractor(img_tensor, self.superpixel_dict)
            
            # 获取superpixel标签矩阵 [H, W]
            sp_input = assigned_masks[0].numpy().astype(np.int32)

        # convert into numpy and standardize
        img = np.array(img).astype(np.float32) / 255.0                                      # normalize to [0,1]  
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])     # imagenet standardization  
        mask = np.array(mask)
        
        # convert into tensor
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        img_id, img_type = self.img_ids[index]
        
        return {
            'image': img,
            'semantic_mask': mask,
            'img_id': img_id,
            'img_type': img_type,
            'superpixel_mask': sp_input
        }

    def __len__(self):
        return len(self.img_ids)

    def _collect_img_id_region(self):
        '''
            collect image ids and region type.
            
            return:
                img_ids: list of tuple (img_id, region)
        
        '''
        img_ids = []
        for region in ['Urban', 'Rural']:
            img_path = osp.join(self.data_root, region, self.img_dir)
            mask_path = osp.join(self.data_root, region, self.mask_dir)
            filenames = os.listdir(img_path)
            
            img_ids += [(f.split('.')[0], region) for f in filenames]
        return img_ids

    def _load_image_and_mask(self, index):
        '''
            load image and mask.
            
            input:
                index: int
            
            return: 
                img: PIL.Image
                mask: PIL.Image.mode = 'L' (grayscale)
        '''
        img_id, img_type = self.img_ids[index]
        img_path = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        return img, mask

    def save_image(self, img, filename, denormalize=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        保存图像到指定文件
        
        Args:
            img: PIL.Image, numpy.ndarray, or torch.Tensor
            filename: str, 保存路径
            denormalize: bool or None, 是否需要反标准化。None时自动判断
            mean: list, ImageNet均值 (用于反标准化)
            std: list, ImageNet标准差 (用于反标准化)
        """
        
        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 转换为numpy数组
        if isinstance(img, torch.Tensor):
            # Tensor -> numpy
            img_np = img.detach().cpu().numpy()
            
            # 处理维度 [C, H, W] -> [H, W, C]
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)
                
        elif isinstance(img, Image.Image):
            # PIL -> numpy
            img_np = np.array(img)
            
        elif isinstance(img, np.ndarray):
            img_np = img.copy()
            
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # 自动判断是否需要反标准化
        if denormalize is None:
            # 如果数值范围在[-3, 3]左右，可能是标准化后的数据
            if img_np.ndim >= 2 and img_np.shape[-1] == 3:
                img_min, img_max = img_np.min(), img_np.max()
                # 标准化后的数据通常在[-2.5, 2.5]范围内
                denormalize = (img_min < -1.0 or img_max < 1.5) and (img_min > -5.0 and img_max < 5.0)
            else:
                denormalize = False
        
        # 执行反标准化
        if denormalize and img_np.ndim >= 2 and img_np.shape[-1] == 3:
            mean = np.array(mean).reshape(1, 1, 3)
            std = np.array(std).reshape(1, 1, 3)
            img_np = img_np * std + mean
        
        # 确保值在[0,1]范围内
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # 处理灰度图
        if img_np.ndim == 2:
            cv2.imwrite(filename, img_np)
        elif img_np.shape[-1] == 1:
            cv2.imwrite(filename, img_np.squeeze(-1))
        elif img_np.shape[-1] == 3:
            # RGB -> BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")
        
        print(f"Image saved: {filename} (denormalize: {denormalize})")

    def save_mask(self, mask, filename, use_color_map=True):
        """
        保存掩码到指定文件
        
        Args:
            mask: PIL.Image, numpy.ndarray, or torch.Tensor
            filename: str, 保存路径
            use_color_map: bool, 是否使用COLOR_MAP进行彩色保存，False则保存原数值标签
        """
        
        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 转换为numpy数组
        if isinstance(mask, torch.Tensor):
            # Tensor -> numpy
            mask_np = mask.detach().cpu().numpy()
            
        elif isinstance(mask, Image.Image):
            # PIL -> numpy
            mask_np = np.array(mask)
            
        elif isinstance(mask, np.ndarray):
            mask_np = mask.copy()
            
        else:
            raise ValueError(f"Unsupported mask type: {type(mask)}")
        
        # 确保是2D数组
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)
        elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np.squeeze(-1)
        
        if use_color_map:
            # 使用COLOR_MAP进行彩色保存
            h, w = mask_np.shape
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 根据类别索引映射颜色
            for class_idx, color in enumerate(self.PALETTE):
                mask_rgb[mask_np == class_idx] = color
            
            # RGB -> BGR for OpenCV
            mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, mask_bgr)
            print(f"Color mask saved: {filename}")
            
        else:
            # 保存原数值标签（灰度图）
            mask_np = mask_np.astype(np.uint8)
            cv2.imwrite(filename, mask_np)
            print(f"Label mask saved: {filename}")


if __name__ == '__main__':
    
    # Test superpixel functionality
    print("Testing superpixel functionality...")
    
    # Create dataset with superpixel enabled
    train_dataset_sp = LoveDATrainDataset(
        data_root='/home/likai/code/LKSeg/data/LoveDA/Train',
        superpixel=True,
        superpixel_type="slic",
        superpixel_dict={
            "n_segments": 100,
            "compactness": 20,
            "sigma": 1,
            "start_label": 0,
            "min_size_factor": 0.5,
            "max_num_iter": 10,
            "enforce_connectivity": True,
        }
    )
    
    # Test single sample with superpixel
    sample_sp = train_dataset_sp[0]
    print(f"\nSuperpixel sample keys: {sample_sp.keys()}")
    
    if 'superpixel_mask' in sample_sp:
        sp_mask = sample_sp['superpixel_mask']
        print(f"Superpixel mask shape: {sp_mask.shape}")
        print(f"Superpixel mask dtype: {sp_mask.dtype}")
        print(f"Number of superpixels: {len(np.unique(sp_mask))}")
        print(f"Superpixel labels range: [{sp_mask.min()}, {sp_mask.max()}]")
        
        # Save visualization
        img_array = np.array(sample_sp['img'].permute(1, 2, 0) * 255, dtype=np.uint8)
        
        # Create superpixel boundary visualization
        boundaries = mark_boundaries(img_array / 255.0, sp_mask)
        
        # Save results
        os.makedirs('temp/superpixel_test', exist_ok=True)
        train_dataset_sp.save_image(sample_sp['img'], 'temp/superpixel_test/original.png')
        train_dataset_sp.save_mask(sp_mask, 'temp/superpixel_test/superpixel_labels.png', use_color_map=False)
        
        # Save boundary visualization
        import cv2
        boundaries_bgr = cv2.cvtColor((boundaries * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite('temp/superpixel_test/superpixel_boundaries.png', boundaries_bgr)
        
        print("Superpixel test passed! Results saved to temp/superpixel_test/")
    else:
        print("Superpixel mask not found in sample")
    
    # Test without superpixel for comparison
    train_dataset_normal = LoveDATrainDataset(
        data_root='/home/likai/code/LKSeg/data/LoveDA/Train',
        superpixel=False
    )
    
    sample_normal = train_dataset_normal[0]
    print(f"\nNormal sample keys: {sample_normal.keys()}")
    print(f"Has sp_input: {'sp_input' in sample_normal}")
    
    # Test dataset basic properties
    print(f"Dataset length: {len(train_dataset_normal)}")
    print(f"Number of classes: {len(train_dataset_normal.CLASSES)}")
    print(f"Classes: {train_dataset_normal.CLASSES}")
    
    # Test single sample
    sample = train_dataset_normal[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['img'].shape}")
    print(f"Image dtype: {sample['img'].dtype}")
    print(f"Image range: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
    
    print(f"Mask shape: {sample['gt_semantic_seg'].shape}")
    print(f"Mask dtype: {sample['gt_semantic_seg'].dtype}")
    print(f"Mask unique values: {torch.unique(sample['gt_semantic_seg']).numpy()}")
    print(f"Image ID: {sample['img_id']}")
    print(f"Image type: {sample['img_type']}")
    
    # # Test multiple samples to ensure consistency
    # print(f"\nTesting 5 random samples:")
    # for i in range(5):
    #     idx = random.randint(0, len(train_dataset)-1)
    #     sample = train_dataset[idx]
    #     print(f"Sample {idx}: img={sample['img'].shape}, mask={sample['gt_semantic_seg'].shape}, "
    #           f"mask_range=[{sample['gt_semantic_seg'].min()}, {sample['gt_semantic_seg'].max()}], "
    #           f"id={sample['img_id']}, type={sample['img_type']}")
    
    # # Test DataLoader compatibility
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=0  # Set to 0 for testing
    # )
    
    # batch = next(iter(train_loader))
    # print(f"\nBatch test:")
    # print(f"Batch img shape: {batch['img'].shape}")
    # print(f"Batch mask shape: {batch['gt_semantic_seg'].shape}")
    # print(f"Batch img_id: {batch['img_id']}")
    # print(f"Batch img_type: {batch['img_type']}")
    
    
