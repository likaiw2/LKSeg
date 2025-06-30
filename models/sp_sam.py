import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from models.segment_anything.modeling import ImageEncoderViT, PromptEncoder
import matplotlib.pyplot as plt
from models.super_pixel.superpixel import SuperpixelExtractor

class SPSamDecoder(nn.Module):
    """
    Multi-class decoder inspired by Mask2Former, enhanced with Prompt token fusion.
    """
    def __init__(
        self,
        transformer_dim: int,
        num_classes: int,
        num_queries: int = 100,
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        mask_dim: int = 256,
    ):
        super().__init__()
        
        self.query_embed = nn.Embedding(num_queries, transformer_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_transformer_layers
        )
        
        self.class_embed = nn.Linear(transformer_dim, num_classes + 1)
        self.mask_embed = MLP(transformer_dim, mask_dim, mask_dim, 3)
        self.mask_predictor = nn.Conv2d(mask_dim, mask_dim, kernel_size=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> dict:
        bs = image_embeddings.shape[0]
        h, w = image_embeddings.shape[-2:]
        
        image_embeddings_flat = image_embeddings.flatten(2).permute(0, 2, 1)  # B, HW, C
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # 融合 Prompt token
        if prompt_embeddings is not None and prompt_embeddings.shape[1] > 0:
            queries = torch.cat([prompt_embeddings, queries], dim=1)

        decoder_output = self.transformer_decoder(
            tgt=queries,
            memory=image_embeddings_flat,
        )
        
        outputs_class = self.class_embed(decoder_output)
        mask_features = self.mask_embed(decoder_output)

        image_embeddings_proc = self.mask_predictor(image_embeddings)
        image_embeddings_proc_flat = image_embeddings_proc.flatten(2)  # B, C, HW

        masks = torch.einsum(
            "bqc,bchw->bqhw",
            mask_features,
            image_embeddings_proc
        )

        return {
            "pred_logits": outputs_class,
            "pred_masks": masks,
        }

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class SPSam(nn.Module):
    def __init__(
        self,
        num_classes,
        image_encoder_args = None,  # 添加参数来配置image_encoder
        prompt_encoder_args = None,  # 添加参数来配置prompt_encoder
        mask_decoder = SPSamDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        transformer_dim: int = 256,
        img_size: int = 1024,  # 添加默认图像大小
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size  # 存储图像大小
        
        # 创建image_encoder实例
        if image_encoder_args is None:
            image_encoder_args = {
                "depth": 12,
                "embed_dim": 768,
                "img_size": self.img_size,
                "mlp_ratio": 4,
                "norm_layer": nn.LayerNorm,
                "num_heads": 12,
                "patch_size": 16,
                "qkv_bias": True,
                "use_rel_pos": True,
                "global_attn_indexes": [2, 5, 8, 11],
                "window_size": 14,
                "out_chans": transformer_dim,
            }
        self.image_encoder = ImageEncoderViT(**image_encoder_args)
        
        # 创建prompt_encoder实例
        if prompt_encoder_args is None:
            prompt_encoder_args = {
                "embed_dim": transformer_dim,
                "image_embedding_size": (self.img_size // 16, self.img_size // 16),
                "input_image_size": (self.img_size, self.img_size),
                "mask_in_chans": 16,
            }
        self.prompt_encoder = PromptEncoder(**prompt_encoder_args)
        
        # 创建mask_decoder实例
        self.mask_decoder = mask_decoder(transformer_dim=transformer_dim, num_classes=self.num_classes)
        
        # 添加辅助头 (Auxiliary Head)
        self.aux_head = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(transformer_dim // 2, self.num_classes, kernel_size=1)
        )
        
        self.superpixel_extractor = SuperpixelExtractor("slic")
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h  # 使用self.img_size而不是self.image_encoder.img_size
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, images, point_coords=None, point_labels=None, boxes=None):
        
        # 1. 预处理后的图像尺寸
        preprocessed_images = self.preprocess(images)

        # 2. 提取superpixel mask，注意输入需要归一化到0~1
        images_for_sp = images.clone()
        images_for_sp = torch.clamp(images_for_sp / 255.0, 0, 1)

        sp_masks, _, _, _ = self.superpixel_extractor(images_for_sp)

        # 3. 处理superpixel mask，生成mask prompt
        # 这里简单假设每个superpixel对应一个mask prompt
        # sp_masks: [NUM_MASKS, H, W]，我们转为[B, NUM_MASKS, H, W]
        batch_size = images.shape[0]
        num_masks = sp_masks.shape[0]
        mask_prompt = sp_masks.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(images.device)
        
        # 4. 获取图像特征
        image_embeddings = self.image_encoder(preprocessed_images)

        # 5. 生成辅助预测
        # 使用图像特征生成辅助预测
        aux_logits = self.aux_head(image_embeddings)
        
        # 6. 处理提示编码
        if (point_coords is not None and point_labels is not None) or boxes is not None:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=boxes,
                masks=mask_prompt,
            )
        else:
            sparse_embeddings = None

        # 7. 解码生成主要预测
        outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prompt_embeddings=sparse_embeddings,
        )
        
        # 8. 在训练模式下返回主要预测和辅助预测
        if self.training:
            outputs["aux_logits"] = aux_logits
            return outputs
        else:
            return outputs


# ======= 测试代码示例：三种 Prompt 联合输入 + 可视化 =======
if __name__ == "__main__":
    from segment_anything.modeling import ImageEncoderViT, PromptEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
    ).to(device)

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ).to(device)

    mask_decoder = SPSamDecoder(transformer_dim=256, num_classes=6).to(device)

    model = SPSam(image_encoder, prompt_encoder, mask_decoder).to(device)
    model.eval()

    # 构造随机图像
    dummy_image = torch.randn(1, 3, 1024, 1024).to(device)

    # 构造 Point Prompt
    point_coords = torch.tensor([[[512, 512]]], dtype=torch.float, device=device)
    point_labels = torch.tensor([[1]], dtype=torch.int, device=device)

    # 构造 Box Prompt
    boxes = torch.tensor([[[256, 256, 768, 768]]], dtype=torch.float, device=device)

    # 构造 Mask Prompt
    mask_inputs = torch.zeros(1, 1, 1024, 1024, device=device)
    mask_inputs[:, :, 300:700, 300:700] = 1.0

    # 推理
    with torch.no_grad():
        outputs = model(dummy_image, point_coords, point_labels, boxes, mask_inputs)

    print(f"类别预测 logits 形状: {outputs['pred_logits'].shape}")
    print(f"掩码预测形状: {outputs['pred_masks'].shape}")

    # 可视化前4个 mask 结果
    pred_masks = outputs["pred_masks"].sigmoid().cpu().numpy()

    for i in range(min(4, pred_masks.shape[1])):
        plt.figure()
        plt.imshow(dummy_image[0].permute(1, 2, 0).cpu().numpy(), alpha=0.5)
        plt.imshow(pred_masks[0, i], cmap="jet", alpha=0.5)
        plt.title(f"Mask {i}")
        plt.axis("off")
        plt.save()
