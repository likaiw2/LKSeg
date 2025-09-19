import cv2
import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

import numpy as np
from typing import Tuple, List

SUPERPIXEL_PARAMETERS = {
    "felzenszwalb": {
        "scale": 600,  # Higher scale means less and larger segments
        "sigma": 0.8,  # is the diameter of a Gaussian kernel, used for smoothing the image prior to segmentation.
        "min_size": 400,  # Minimum component size. Enforced using postprocessing.
    },
    
    "slic": {
        "n_segments": 50,  # 100  # The (approximate) number of labels in the segmented output image.
        "compactness": 20,
        # Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. We recommend exploring possible values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
        "sigma": 1,  # 0,  # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
        "start_label": 0,
        "min_size_factor": 0.5,  # Proportion of the minimum segment size to be removed with respect to the supposed segment size `depth*width*height/n_segments`
        "max_num_iter": 10,  # Maximum number of iterations of k-means
        "enforce_connectivity": True,  # Whether the generated segments are connected or not
    },
    
    "quickshift": {
        "ratio": 1.0,  # 1.0,  # Balances color-space proximity and image-space proximity. Higher values give more weight to color-space.
        "kernel_size": 3,  # 5,  # Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
        "max_dist": 10,  # Cut-off point for data distances. Higher means fewer clusters.
        "sigma": 1,  # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
    },
    
    "watershed": {
        "markers": 200,  # The number of markers, i.e. the number of segments in the output segmentation.
        "compactness": 1e-5,  # Use compact watershed with given compactness parameter. Higher values result in more regularly-shaped watershed basins.
    },
    
    "seeds": {
        "image_width": 1024,
        "image_height": 1024,
        "image_channels": 3,
        "num_superpixels": 200,  # Desired number of superpixels. Note that the actual number may be smaller due to restrictions (depending on the image size and num_levels). Use getNumberOfSuperpixels() to get the actual number.
        "num_levels": 4,  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
        "prior": 1,  # enable 3x3 shape smoothing term if >0. A larger value leads to smoother shapes. prior must be in the range [0, 5].
        "histogram_bins": 5,  # Number of histogram bins.
        "double_step": False,  # If true, iterate each block level twice for higher accuracy.
        "num_iterations": 10,  # Number of iterations. Higher number improves the result.
    }
}

class SuperpixelExtractor:
    """Proposes class-agnostic masks for the given images using superpixels.

    Args:
        parameters_dict (dict): The parameters for the superpixel algorithm. Contains also which algorithm to use.
    """

    def __init__(self, algorithm):
        # print("Parameters dict at beginning of init", algorithm)

        if isinstance(algorithm, str):

            self.algorithm = algorithm
            if algorithm == "felzenszwalb":
                parameters_dict = SUPERPIXEL_PARAMETERS["felzenszwalb"]
            elif algorithm == "slic":
                parameters_dict = SUPERPIXEL_PARAMETERS["slic"]
            elif algorithm == "quickshift":
                parameters_dict = SUPERPIXEL_PARAMETERS["quickshift"]
            elif algorithm == "watershed":
                parameters_dict = SUPERPIXEL_PARAMETERS["watershed"]
            elif algorithm == "seeds":
                parameters_dict = SUPERPIXEL_PARAMETERS["seeds"]
                self.num_iterations = parameters_dict.pop("num_iterations")
            else:
                raise NotImplementedError(f"Superpixel algorithm {algorithm} not implemented")
        elif isinstance(algorithm, dict):

            self.algorithm = algorithm.pop("algorithm")
            if self.algorithm == "seeds":
                self.num_iterations = algorithm.pop("num_iterations")
            parameters_dict = algorithm
        else:
            raise TypeError(f"Algorithm must be either a string or a dictionary, but is {type(algorithm)}")

        self.parameters_dict = parameters_dict

    def __call__(self, images,parameters_dict=None):
        """
        Args:
            images (torch.Tensor): [B, C, H, W]
        Output:
            pred_masks_batch: torch.Tensor [NUM_MASKS, IMAGE_HEIGHT, IMAGE_WIDTH]
                Contains the list of binary masks for each image in the batch. NUM_MASKS is the number of total proposed
                masks for all images.
            n_pred_masks: List [BATCH_SIZE]
                Number of proposed masks for each image in the batch.
            covered_pixels_batch: torch.Tensor [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH]
                Indicates for each pixel whether it is covered by a mask.
            assigned_masks_batch: torch.Tensor [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH]
                Indicates for each pixel which mask it is assigned to.
        """
        if parameters_dict is not None:
            self.parameters_dict = parameters_dict

        pred_masks_batch = []
        n_pred_masks = []
        assigned_masks_batch = []
        covered_pixels_batch = torch.ones(images.shape[0], images.shape[2], images.shape[3]).type(torch.bool)

        for img in images:
            if self.algorithm == "seeds":
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img.astype(np.uint8))
            else:
                img = img.permute(1, 2, 0).cpu().numpy() / 255

            if self.algorithm == "felzenszwalb":
                superpixel_mask = felzenszwalb(img, **self.parameters_dict)

            elif self.algorithm == "slic":
                superpixel_mask = slic(img, **self.parameters_dict)

            elif self.algorithm == "quickshift":
                superpixel_mask = quickshift(img, **self.parameters_dict)

            elif self.algorithm == "watershed":
                gradient = sobel(rgb2gray(img))
                superpixel_mask = watershed(gradient, **self.parameters_dict)

            elif self.algorithm == "seeds":
                superpix_seeds = cv2.ximgproc.createSuperpixelSEEDS(**self.parameters_dict)
                superpix_seeds.iterate(img, self.num_iterations)
                superpixel_mask = superpix_seeds.getLabels()
                num_superpixels = superpix_seeds.getNumberOfSuperpixels()
            else:
                raise NotImplementedError(f"Superpixel algorithm {self.algorithm} not implemented.")

            # create a binary mask for each superpixel
            if self.algorithm == "seeds":
                superpixel_mask_binary = np.array([superpixel_mask == i for i in np.arange(num_superpixels)])
            else:
                superpixel_mask_binary = np.array([superpixel_mask == i for i in np.unique(superpixel_mask)])
            num_superpixel = superpixel_mask_binary.shape[0]

            pred_masks_batch.append(superpixel_mask_binary)
            n_pred_masks.append(num_superpixel)
            assigned_masks_batch.append(superpixel_mask[None, :, :])

        pred_masks_batch = torch.Tensor(np.concatenate(pred_masks_batch, axis=0)).type(torch.bool)
        # n_pred_masks = torch.Tensor(n_pred_masks).type(torch.long)
        assigned_masks_batch = torch.Tensor(np.concatenate(assigned_masks_batch, axis=0)).type(torch.long)

        if self.algorithm == "watershed":
            assigned_masks_batch = assigned_masks_batch - 1

        return pred_masks_batch, n_pred_masks, covered_pixels_batch, assigned_masks_batch

class SEEDSSuperpixelExtractor:

    def __init__(self, num_superpixels, compactness_superpixels):
        self.num_superpixels = num_superpixels
        self.compactness_superpixels = compactness_superpixels

    def calculate_pe_opencv_seeds(self, img):
        img = np.ascontiguousarray(img.astype(np.uint8))
        image_height, image_width = img.shape[:2]

        num_levels = 4  # SEEDS Number of Levels
        prior = int(self.compactness_superpixels)  # SEEDS Smoothing Prior | range: [0, 5] | default: 1
        num_histogram_bins = 5  # SEEDS histogram bins
        double_step = False  # SEEDS two steps
        num_iterations = 10  # Iterations

        superpix_seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width,
            image_height,
            3,
            self.num_superpixels,
            num_levels,
            prior,
            histogram_bins=num_histogram_bins,
            double_step=double_step,
        )
        superpix_seeds.iterate(img, num_iterations)

        segments = superpix_seeds.getLabels()

        if segments.min() == 0:
            segments = segments + 1

        regions_seeds = regionprops(segments + 1)
        centroids = np.array([np.rint(c.centroid).astype(np.int32) for c in regions_seeds])

        return segments, centroids

class SPExtractorSAM(SuperpixelExtractor):
    """继承超像素提取器，专门为SAM提供prompt格式的输出"""
    
    def __init__(self, algorithm: str = "slic", **kwargs):
        super().__init__(algorithm)
        # 可以通过kwargs覆盖默认参数
        if kwargs:
            self.parameters_dict.update(kwargs)
    
    def extract_centers(self, images: torch.Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        提取超像素中心点作为SAM的点提示
        
        Args:
            images: 输入图像张量 (B, C, H, W)
            
        Returns:
            centers_batch: 每张图像的中心点坐标列表 [(N1, 2), (N2, 2), ...]，格式为(x, y)
            labels_batch: 每张图像的点标签列表 [(N1,), (N2,), ...]，全部为前景点(1)
        """
        # 调用父类方法获取超像素结果
        _, n_pred_masks, _, assigned_masks_batch = self(images)
        
        centers_batch = []
        labels_batch = []
        
        for i in range(images.shape[0]):
            assigned_mask = assigned_masks_batch[i].numpy()  # (H, W)
            unique_labels = np.unique(assigned_mask)
            unique_labels = unique_labels[unique_labels > 0]  # 排除背景
            
            centers = []
            for label in unique_labels:
                mask_coords = np.where(assigned_mask == label)
                if len(mask_coords[0]) > 0:
                    # 计算中心点 (注意：转换为x,y格式)
                    center_y = mask_coords[0].mean()
                    center_x = mask_coords[1].mean()
                    centers.append([center_x, center_y])
            
            centers = np.array(centers, dtype=np.float32)
            labels = np.ones(len(centers), dtype=np.int32)  # 全部标记为前景点
            
            centers_batch.append(centers)
            labels_batch.append(labels)
        
        return {
            'centers': centers_batch,
            'labels': labels_batch
        }
    
    def extract_boxes(self, images: torch.Tensor) -> List[np.ndarray]:
        """
        提取超像素边界框作为SAM的框提示
        
        Args:
            images: 输入图像张量 (B, C, H, W)
            
        Returns:
            boxes_batch: 每张图像的边界框列表 [(N1, 4), (N2, 4), ...]，格式为(x1, y1, x2, y2)
        """
        # 调用父类方法获取超像素结果
        _, n_pred_masks, _, assigned_masks_batch = self(images)
        
        boxes_batch = []
        
        for i in range(images.shape[0]):
            assigned_mask = assigned_masks_batch[i].numpy()  # (H, W)
            
            # 使用regionprops计算每个区域的边界框
            props = regionprops(assigned_mask.astype(int))
            
            boxes = []
            for prop in props:
                # regionprops返回(min_row, min_col, max_row, max_col)
                min_row, min_col, max_row, max_col = prop.bbox
                # 转换为SAM期望的(x1, y1, x2, y2)格式
                box = [min_col, min_row, max_col, max_row]
                boxes.append(box)
            
            boxes = np.array(boxes, dtype=np.float32)
            boxes_batch.append(boxes)
        
        return {"boxes": boxes_batch}
    
    def extract_masks(self, images: torch.Tensor) -> Tuple[List[np.ndarray], List[int]]:
        """
        提取超像素掩码作为SAM的掩码提示
        
        Args:
            images: 输入图像张量 (B, C, H, W)
            
        Returns:
            masks_batch: 每张图像的二值掩码列表 [(N1, H, W), (N2, H, W), ...]
            n_masks_batch: 每张图像的掩码数量列表 [N1, N2, ...]
        """
        # 调用父类方法获取超像素结果
        pred_masks_batch, n_pred_masks, _, assigned_masks_batch = self(images)
        
        masks_batch = []
        n_masks_batch = []
        
        # 重新组织掩码数据，按图像分组
        mask_start_idx = 0
        for i in range(images.shape[0]):
            n_masks = n_pred_masks[i]
            
            # 提取当前图像的所有掩码
            image_masks = pred_masks_batch[mask_start_idx:mask_start_idx + n_masks]
            image_masks = image_masks.numpy().astype(np.uint8)  # 转换为numpy数组
            
            masks_batch.append(image_masks)
            n_masks_batch.append(n_masks)
            
            mask_start_idx += n_masks
        
        return {
            'masks': masks_batch,
            'n_masks': n_masks_batch
        }
    
    def get_all_prompts(self, images: torch.Tensor) -> dict:
        """
        一次性获取所有类型的提示
        
        Args:
            images: 输入图像张量 (B, C, H, W)
            
        Returns:
            prompts: 包含所有提示类型的字典
        """
        centers_batch, labels_batch = self.extract_centers(images)
        boxes_batch = self.extract_boxes(images)
        masks_batch, n_masks_batch = self.extract_masks(images)
        
        return {
            'centers': centers_batch,
            'labels': labels_batch,
            'boxes': boxes_batch,
            'masks': masks_batch,
            'n_masks': n_masks_batch
        }


