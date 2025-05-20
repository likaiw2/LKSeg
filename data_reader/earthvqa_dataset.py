from transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as albu
from PIL import Image
import random


COLOR_MAP = dict(
    # nothing=(0, 0, 0),              # 0 black
    Background=(255, 255, 255),     # 1 white
    Building=(255, 0, 0),           # 2 red
    Road=(255, 255, 0),             # 3 yellow
    Water=(0, 0, 255),              # 4 blue
    Barren=(159, 129, 183),         # 5 purple
    Forest=(0, 255, 0),             # 6 green
    Agricultural=(255, 195, 128),   # 7 orange
    Playground=(165,0,165),         # 8 pink
    Pond=(0,185,246),               # 9 cyan
)

CLASSES = list(COLOR_MAP.keys())
PALETTE = list(COLOR_MAP.values())

ORIGIN_IMG_SIZE = (1024, 1024)

def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Sharpen(),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

class EarthVQADataset(Dataset):
    def __init__(self, data_root='data/EarthVQA/Train', 
                 img_dir='images_png', 
                 mask_dir='masks_png',
                 img_suffix='.png', 
                 mask_suffix='.png', 
                 transform=train_aug, 
                 img_size=ORIGIN_IMG_SIZE):
        
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids()

    def get_img_ids(self):
        img_filename_list = os.listdir(osp.join(self.data_root, self.img_dir))
        mask_filename_list = os.listdir(osp.join(self.data_root, self.mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [id.split('.')[0] for id in img_filename_list]
        return img_ids

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        mask_np = np.array(mask)
        mask_np[mask_np == 0] = len(CLASSES)  # 将背景标记为255
        mask_np -= 1
        mask = Image.fromarray(mask_np)
        
        if self.transform:
            img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id, 'img_type': 'unknown'}
        return results

    def __len__(self):
        return len(self.img_ids)
    
    
    
    
if __name__ == '__main__':
    # Example usage
    train_dataset = EarthVQADataset(data_root='/home/liw324/code/Segment/LKSeg/data/EarthVQA/Train')
    
    print("########")
    sample = train_dataset[0]
    print(sample['img'].size())
    print(sample['gt_semantic_seg'].size())
    print(sample['gt_semantic_seg'].unique())
    print(sample['img_id'])
    print(sample['img_type'])
    print("########")
    

    image_path = f"/home/liw324/code/Segment/LKSeg/data/EarthVQA/Train/masks_png/{sample['img_id']}.png"
    np_image = np.array(Image.open(image_path))
    print("ori_mask",np.unique(np_image))
    
    # import torchvision.transforms.functional as F
    # from torchvision.utils import save_image
    # from PIL import Image

    # # 获取样本
    # # sample = train_dataset[0]
    # img_tensor = sample['img']             # (3, H, W), float32
    # mask_tensor = sample['gt_semantic_seg']  # (H, W), long

    # # 保存图像（输入图像），转为 0-255 范围
    # save_image(img_tensor, f"sample{sample['img_id']}_img.png")

    # # 保存 mask（标签图）
    # # 将语义标签转换为可视图像
    # mask_np = mask_tensor.numpy().astype(np.uint8)
    # color_map = np.array(PALETTE, dtype=np.uint8)  # shape: (7, 3)
    # color_mask = color_map[mask_np]                # (H, W, 3)

    # # 保存为PNG图像
    # Image.fromarray(color_mask).save(f"sample{sample['img_id']}_mask.png")
    
    
