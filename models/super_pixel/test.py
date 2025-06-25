import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.feature import canny
from skimage.color import rgb2gray
from superpixel import SuperpixelExtractor
# from superpixel_configs import felzenszwalb_parameters_dict, slic_parameters_dict, quickshift_parameters_dict, watershed_parameters_dict, seeds_parameters_dict

slic_parameters_dict = {
    "n_segments": 200,  # 100  # The (approximate) number of labels in the segmented output image.
    "compactness": 10,
    # Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. We recommend exploring possible values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
    "sigma": 1,  # 0,  # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
    "start_label": 0,
    "min_size_factor": 0.5,  # Proportion of the minimum segment size to be removed with respect to the supposed segment size `depth*width*height/n_segments`
    "max_num_iter": 10,  # Maximum number of iterations of k-means
    "enforce_connectivity": True,  # Whether the generated segments are connected or not
}

# felzenszwalb_parameters_dict = {
#     "scale": 600,  # Higher scale means less and larger segments
#     "sigma": 0.8,  # is the diameter of a Gaussian kernel, used for smoothing the image prior to segmentation.
#     "min_size": 400,  # Minimum component size. Enforced using postprocessing.
# }

def test_single_image_all_algorithms(img_path):
    scale_list = [10, 50, 100, 300, 600]
    sigma_list = [0.2, 0.4, 0.6, 0.8, 1]

    img = imread(img_path)
    if img.shape[-1] > 3:
        img = img[:, :, :3]
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

    img_for_vis = img / 255.0 if img.max() > 1 else img
    
    extractor = SuperpixelExtractor("felzenszwalb")
    for scale in scale_list:
        cols = len(sigma_list)
        rows = 1
        plt.figure(figsize=(4 * cols, 4 * rows))
        for idx, sigma in enumerate(sigma_list):
            print(f"scale={scale}, sigma={sigma}")
            felz_params = slic_parameters_dict.copy()
            felz_params["scale"] = scale
            felz_params["sigma"] = sigma
            try:
                _, n_masks, _, assigned = extractor(img_tensor, felz_params)
                segments = assigned[0].numpy()
                vis_image = mark_boundaries(img_for_vis, segments)
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(vis_image)
                plt.title(f"scale={scale}, sigma={sigma}")
                plt.axis('off')
            except Exception as e:
                print(f"[ERROR] scale={scale}, sigma={sigma}: {e}")

        plt.tight_layout()
        plt.savefig(f"out1/scale={scale}.png", dpi=300, bbox_inches='tight')
        plt.close()


# 调用主函数
image_id = "235"
image_path = f"/home/liw324/code/Segment/LKSeg/data/EarthVQA/Train/images_png/{image_id}.png"
test_single_image_all_algorithms(image_path)