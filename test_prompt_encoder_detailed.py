import torch
import torch.nn as nn
from models.sam.modeling.prompt_encoder import PromptEncoder

def inspect_prompt_encoder_internals():
    """详细检查PromptEncoder的内部结构和中间特征形状"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个PromptEncoder用于测试
    img_size = 1024
    embed_dim = 256
    image_embedding_size = (img_size // 16, img_size // 16)  # 64x64
    
    model = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=(img_size, img_size),
        mask_in_chans=16,
    ).to(device)
    
    # 创建钩子函数来捕获中间特征
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = {
                'input_shape': [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x) for x in input],
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else 
                               [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x) for x in output]
            }
        return hook
    
    # 注册钩子
    hooks = []
    
    # 注册PE层钩子
    hooks.append(model.pe_layer.register_forward_hook(hook_fn('pe_layer')))
    
    # 注册mask_downscaling层的钩子
    for i, layer in enumerate(model.mask_downscaling):
        hooks.append(layer.register_forward_hook(hook_fn(f'mask_downscaling_{i}')))
    
    # 创建测试输入
    batch_size = 1
    
    # 点提示: [B, N, 2], [B, N]
    point_coords = torch.randint(0, img_size, (batch_size, 5, 2)).float().to(device)
    point_labels = torch.randint(0, 2, (batch_size, 5)).to(device)
    
    # 框提示: [B, 2, 2]
    boxes = torch.randint(0, img_size, (batch_size, 2, 2)).float().to(device)
    
    # 掩码提示: [B, 1, H, W]
    masks = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float().to(device)
    
    print(f"Point coords shape: {point_coords.shape}")
    print(f"Point labels shape: {point_labels.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Masks shape: {masks.shape}")
    
    # 运行模型
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model(
            points=(point_coords, point_labels),
            boxes=boxes,
            masks=masks
        )
    
    print(f"\nOutput shapes:")
    print(f"Sparse embeddings shape: {sparse_embeddings.shape}")
    print(f"Dense embeddings shape: {dense_embeddings.shape}")
    
    # 测试get_dense_pe方法
    dense_pe = model.get_dense_pe()
    print(f"Dense PE shape: {dense_pe.shape}")
    
    # 打印中间特征形状
    print("\n===== Intermediate Feature Shapes =====")
    for name, feature in features.items():
        print(f"\n{name}:")
        print(f"  Input shape: {feature['input_shape']}")
        print(f"  Output shape: {feature['output_shape']}")
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 打印模型结构
    print("\n===== Model Structure =====")
    print(model)
    
    # 分析各个组件
    print("\n===== Component Analysis =====")
    print(f"Embed dimension: {model.embed_dim}")
    print(f"Image embedding size: {model.image_embedding_size}")
    print(f"Input image size: {model.input_image_size}")
    print(f"Mask input size: {model.mask_input_size}")
    print(f"Number of point embeddings: {model.num_point_embeddings}")
    
    # 测试各种输入组合
    print("\n===== Testing Different Input Combinations =====")
    
    # 只有点提示
    with torch.no_grad():
        sparse_only_points, dense_only_points = model(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )
    print(f"Only points - Sparse shape: {sparse_only_points.shape}")
    
    # 只有框提示
    with torch.no_grad():
        sparse_only_boxes, dense_only_boxes = model(
            points=None,
            boxes=boxes,
            masks=None
        )
    print(f"Only boxes - Sparse shape: {sparse_only_boxes.shape}")
    
    # 只有掩码提示
    with torch.no_grad():
        sparse_only_masks, dense_only_masks = model(
            points=None,
            boxes=None,
            masks=masks
        )
    print(f"Only masks - Dense shape: {dense_only_masks.shape}")

if __name__ == "__main__":
    inspect_prompt_encoder_internals()