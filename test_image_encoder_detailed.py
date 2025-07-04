import torch
import torch.nn as nn
from models.sam.modeling.image_encoder import ImageEncoderViT, Block, Attention, PatchEmbed

def inspect_image_encoder_internals():
    """详细检查ImageEncoderViT的内部结构和中间特征形状"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个小型的ImageEncoderViT用于测试
    model = ImageEncoderViT(
        depth=2,  # 只使用2层以简化输出
        embed_dim=256,
        img_size=512,
        patch_size=16,
        num_heads=8,
        global_attn_indexes=[0, 1],
        window_size=14,
        out_chans=128,
    ).to(device)
    
    # 创建钩子函数来捕获中间特征
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = {
                'input_shape': [tuple(x.shape) for x in input],
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else [tuple(x.shape) for x in output]
            }
        return hook
    
    # 注册钩子
    hooks = []
    hooks.append(model.patch_embed.register_forward_hook(hook_fn('patch_embed')))
    
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(hook_fn(f'block_{i}')))
        hooks.append(block.attn.register_forward_hook(hook_fn(f'block_{i}_attn')))
        hooks.append(block.mlp.register_forward_hook(hook_fn(f'block_{i}_mlp')))
    
    for i, layer in enumerate(model.neck):
        hooks.append(layer.register_forward_hook(hook_fn(f'neck_{i}')))
    
    # 创建输入并运行模型
    x = torch.randn(1, 3, 512, 512).to(device)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
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
    
    # 分析patch_embed的行为
    print("\n===== PatchEmbed Analysis =====")
    patch_embed = model.patch_embed
    print(f"Kernel size: {patch_embed.proj.kernel_size}")
    print(f"Stride: {patch_embed.proj.stride}")
    print(f"Padding: {patch_embed.proj.padding}")
    
    # 计算patch数量
    img_size = 512
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    print(f"Expected number of patches: {num_patches} ({img_size // patch_size}x{img_size // patch_size})")
    
    # 检查pos_embed的形状
    if model.pos_embed is not None:
        print(f"Position embedding shape: {model.pos_embed.shape}")

if __name__ == "__main__":
    inspect_image_encoder_internals()