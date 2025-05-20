from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
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
)

CLASSES = list(COLOR_MAP.keys())
PALETTE = list(COLOR_MAP.values())

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Sharpen(),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # multi-scale training and crop
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']

    return img, mask


def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)


def test_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_test_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class LoveDATrainDataset(Dataset):
    def __init__(self, 
                 data_root='data/LoveDA/Train', 
                 img_dir='images_png', 
                 mosaic_ratio=0.25,
                 mask_dir='masks_png', 
                 img_suffix='.png', 
                 mask_suffix='.png',
                 transform=train_aug, 
                 img_size=ORIGIN_IMG_SIZE):
        
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self._collect_img_ids()

    def __getitem__(self, index):
        # 随机决定是否使用 mosaic 增强
        if random.random() < self.mosaic_ratio:
            img, mask = self._build_mosaic_image_and_mask(index)
        else:
            img, mask = self._load_image_and_mask(index)

        # 应用数据增强
        if self.transform:
            img, mask = self.transform(img, mask)

        # 转为 PyTorch tensor 格式
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        img_id, img_type = self.img_ids[index]
        
        return {
            'img': img,
            'gt_semantic_seg': mask,
            'img_id': img_id,
            'img_type': img_type
        }

    def __len__(self):
        return len(self.img_ids)

    def _collect_img_ids(self):
        # 统一获取 Urban 和 Rural 图像ID
        img_ids = []
        for region in ['Urban', 'Rural']:
            img_path = osp.join(self.data_root, region, self.img_dir)
            mask_path = osp.join(self.data_root, region, self.mask_dir)
            filenames = os.listdir(img_path)
            assert len(filenames) == len(os.listdir(mask_path)), f"len(filenames)"
            img_ids += [(f.split('.')[0], region) for f in filenames]
        return img_ids

    def _load_image_and_mask(self, index):
        # 加载一张图像和其掩膜
        img_id, img_type = self.img_ids[index]
        img_path = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        mask_np = np.array(mask)
        mask_np[mask_np == 0] = len(CLASSES)  # 将背景标记为255
        mask_np -= 1
        mask = Image.fromarray(mask_np)
        
        return img, mask

    def _build_mosaic_image_and_mask(self, index):
        # 构造 Mosaic 增强图像
        indexes = [index] + random.choices(range(len(self)), k=3)
        imgs_masks = [tuple(map(np.array, self._load_image_and_mask(i))) for i in indexes]

        w, h = self.img_size
        offset_x = random.randint(w // 4, 3 * w // 4)
        offset_y = random.randint(h // 4, 3 * h // 4)

        sizes = [
            (offset_x, offset_y),
            (w - offset_x, offset_y),
            (offset_x, h - offset_y),
            (w - offset_x, h - offset_y)
        ]

        crops = [
            albu.RandomCrop(width=sw, height=sh)(image=img, mask=mask)
            for (img, mask), (sw, sh) in zip(imgs_masks, sizes)
        ]

        # 拼接图像和掩膜
        top = np.concatenate((crops[0]['image'], crops[1]['image']), axis=1)
        bottom = np.concatenate((crops[2]['image'], crops[3]['image']), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((crops[0]['mask'], crops[1]['mask']), axis=1)
        bottom_mask = np.concatenate((crops[2]['mask'], crops[3]['mask']), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        return Image.fromarray(img), Image.fromarray(mask)
    

if __name__ == '__main__':
    # Example usage
    train_dataset = LoveDATrainDataset(data_root='/home/liw324/code/Segment/LKSeg/data/LoveDA/Train',
                                       mosaic_ratio=0)
    
    print("########")
    sample = train_dataset[0]
    print(sample['img'].size())
    print(sample['gt_semantic_seg'].size())
    print(sample['gt_semantic_seg'].unique())
    print(sample['img_id'])
    print(sample['img_type'])
    print("########")
    

    image_path = f"/home/liw324/code/Segment/LKSeg/data/LoveDA/Train/{sample['img_type']}/masks_png/{sample['img_id']}.png"
    np_image = np.array(Image.open(image_path))
    print("ori_mask",np.unique(np_image))
    
    import torchvision.transforms.functional as F
    from torchvision.utils import save_image
    from PIL import Image

    # 获取样本
    # sample = train_dataset[0]
    img_tensor = sample['img']             # (3, H, W), float32
    mask_tensor = sample['gt_semantic_seg']  # (H, W), long

    # 保存图像（输入图像），转为 0-255 范围
    save_image(img_tensor, f"sample{sample['img_id']}_img.png")

    # 保存 mask（标签图）
    # 将语义标签转换为可视图像
    mask_np = mask_tensor.numpy().astype(np.uint8)
    color_map = np.array(PALETTE, dtype=np.uint8)  # shape: (7, 3)
    color_mask = color_map[mask_np]                # (H, W, 3)

    # 保存为PNG图像
    Image.fromarray(color_mask).save(f"sample{sample['img_id']}_mask.png")
    
    
