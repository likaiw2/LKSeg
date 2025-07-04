import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from models.sam import SamPredictor, sam_model_registry

def point_prompt_example():
    # 1. 加载图像
    image = np.array(Image.open("path/to/your/image.jpg"))
    
    # 2. 加载SAM模型
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    # 3. 设置图像
    predictor.set_image(image)
    
    # 4. 定义点提示
    # 假设我们想分割图像中的一个物体
    # 点坐标格式为 [x, y]
    # 点标签: 1表示前景点，0表示背景点
    
    # 例如，我们有2个前景点和1个背景点
    point_coords = np.array([
        [300, 400],  # 前景点1，位于物体内部
        [320, 450],  # 前景点2，位于物体内部
        [500, 375]   # 背景点，位于物体外部
    ])
    
    point_labels = np.array([1, 1, 0])  # 两个前景点(1)，一个背景点(0)
    
    # 5. 使用点提示预测掩码
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True  # 返回多个掩码预测
    )
    
    # 6. 可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # 绘制点提示
    for i, (coord, label) in enumerate(zip(point_coords, point_labels)):
        color = 'green' if label == 1 else 'red'
        plt.scatter(coord[0], coord[1], color=color, s=50)
    
    # 选择得分最高的掩码
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    # 绘制掩码轮廓
    plt.contour(best_mask, colors='blue', levels=[0.5], alpha=0.7)
    
    # 绘制半透明掩码
    plt.imshow(best_mask, alpha=0.5, cmap='jet')
    
    plt.title(f"SAM Segmentation with Point Prompts (Score: {scores[best_mask_idx]:.3f})")
    plt.axis('off')
    plt.show()
    
    # 7. 打印点提示的内部处理过程
    print("\n点提示的内部处理过程:")
    print("1. 输入点坐标被转换为模型输入尺寸")
    print("2. 点坐标偏移0.5以对准像素中心")
    print("3. 使用位置编码对点坐标进行编码")
    print("4. 根据点标签添加不同的嵌入向量:")
    print("   - 前景点(标签=1)添加前景点嵌入")
    print("   - 背景点(标签=0)添加背景点嵌入")
    print("5. 如果没有框提示，添加一个填充点(标签=-1)")
    print("6. 所有点嵌入被连接形成稀疏嵌入")
    
    # 8. 展示点嵌入的形状
    with torch.no_grad():
        # 转换为PyTorch张量
        point_coords_torch = torch.from_numpy(point_coords).float().unsqueeze(0)  # [1, 3, 2]
        point_labels_torch = torch.from_numpy(point_labels).int().unsqueeze(0)    # [1, 3]
        
        # 获取原始图像尺寸
        original_size = image.shape[:2]
        
        # 转换点坐标到模型输入尺寸
        input_size = predictor.transform.get_preprocess_shape(*original_size, long_side_length=1024)
        point_coords_transformed = predictor.transform.apply_coords(point_coords, original_size)
        point_coords_torch = torch.from_numpy(point_coords_transformed).float().unsqueeze(0)
        
        # 使用PromptEncoder处理点提示
        sparse_embeddings, _ = predictor.model.prompt_encoder(
            points=(point_coords_torch, point_labels_torch),
            boxes=None,
            masks=None
        )
        
        print(f"\n点提示输入形状:")
        print(f"点坐标: {point_coords_torch.shape}")  # [1, 3, 2]
        print(f"点标签: {point_labels_torch.shape}")  # [1, 3]
        
        print(f"\n点嵌入输出形状:")
        print(f"稀疏嵌入: {sparse_embeddings.shape}")  # [1, 4, embed_dim] (包含填充点)

if __name__ == "__main__":
    point_prompt_example()