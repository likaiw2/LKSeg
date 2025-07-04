import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from models.sam.modeling import ImageEncoderViT, PromptEncoder
import matplotlib.pyplot as plt
from models.super_pixel.superpixel import SuperpixelExtractor

class SPSamDecoder(nn.Module):
    """
    Multi-class decoder inspired by Mask2Former, using query-based mask prediction.
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

        self.num_classes = num_classes
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
        self.mask_predictor = nn.Conv2d(transformer_dim, mask_dim, kernel_size=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        target_size: tuple = None,
    ) -> dict:
        bs = image_embeddings.shape[0]
        
        # 展平图像特征用于Transformer处理
        image_embeddings_flat = image_embeddings.flatten(2).permute(0, 2, 1)  # B, HW, C

        # 准备查询嵌入
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # 融合Prompt token
        if prompt_embeddings is not None and prompt_embeddings.shape[1] > 0:
            queries = torch.cat([prompt_embeddings, queries], dim=1)

        # 通过Transformer解码器处理查询
        decoder_output = self.transformer_decoder(
            tgt=queries,
            memory=image_embeddings_flat,
        )

        # 生成类别预测和掩码嵌入
        outputs_class = self.class_embed(decoder_output)
        mask_features = self.mask_embed(decoder_output)

        # 处理图像特征
        image_embeddings_proc = self.mask_predictor(image_embeddings)

        # 生成掩码
        masks = torch.einsum(
            "bqc,bchw->bqhw",
            mask_features,
            image_embeddings_proc
        )

        # 上采样掩码到目标尺寸
        if target_size is not None:
            masks = F.interpolate(
                masks,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        return {
            "pred_masks": masks,
            "pred_logits": outputs_class,  # 重命名为pred_logits以保持一致性
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
        image_encoder_args = None,
        prompt_encoder_args = None,
        mask_decoder = SPSamDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        transformer_dim: int = 256,
        img_size: int = 1024,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 创建image_encoder实例
        if image_encoder_args is None:
            image_encoder_args = {
                "depth": 6,
                "embed_dim": 512,
                "img_size": self.img_size,
                "mlp_ratio": 4,
                "norm_layer": nn.LayerNorm,
                "num_heads": 8,
                "patch_size": 16,
                "qkv_bias": True,
                "use_rel_pos": True,
                "global_attn_indexes": [1, 3, 5],
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
        
        self.superpixel_extractor = SuperpixelExtractor("slic")
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, images, point_coords=None, point_labels=None, boxes=None):
        preprocessed_images = self.preprocess(images)

        original_h, original_w = images.shape[-2:]
        target_size = (original_h, original_w)

        # 获取图像特征
        image_embeddings = self.image_encoder(preprocessed_images)

        # 提取超像素掩码
        images_for_sp = images.clone()
        images_for_sp = torch.clamp(images_for_sp / 255.0, 0, 1)

        sp_masks, n_pred_masks, _, _ = self.superpixel_extractor(images_for_sp)
        
        # 输出每个图像的超像素数量
        batch_size = images.shape[0]
        # for b in range(batch_size):
            # print(f"Image {b}: {n_pred_masks[b]} superpixels")
        
        # 输出总超像素数量
        # total_superpixels = sum(n_pred_masks)
        # print(f"Total superpixels across batch: {total_superpixels}")
        # print(f"Superpixel masks tensor shape: {sp_masks.shape}")
        
        # 处理超像素掩码 - 每个批次分别处理
        all_sparse_embeddings = []
        
        # 检查是否有点或框提示
        has_point_prompt = point_coords is not None and point_labels is not None
        has_box_prompt = boxes is not None
        
        # 为每个批次处理提示
        for b in range(batch_size):
            batch_sparse_embeddings = []
            
            # 1. 处理点和框提示（如果有）
            if has_point_prompt or has_box_prompt:
                # 提取当前批次的点和框
                batch_points = (
                    point_coords[b:b+1] if has_point_prompt else None,
                    point_labels[b:b+1] if has_point_prompt else None
                )
                batch_boxes = boxes[b:b+1] if has_box_prompt else None
                
                # 获取点和框的嵌入
                point_box_embeddings, _ = self.prompt_encoder(
                    points=batch_points,
                    boxes=batch_boxes,
                    masks=None
                )
                batch_sparse_embeddings.append(point_box_embeddings)
            
            # 2. 处理超像素掩码提示
            if sp_masks.numel() > 0:
                # 计算当前批次的超像素掩码起始和结束索引
                start_idx = sum(n_pred_masks[:b]) if b > 0 else 0
                end_idx = start_idx + n_pred_masks[b]
                
                # print(f"Processing superpixels for image {b}: indices {start_idx} to {end_idx-1}")
                
                # 获取当前批次的超像素掩码
                batch_sp_masks = sp_masks[start_idx:end_idx]
                
                # 为每个超像素掩码单独获取嵌入
                for mask_idx in range(len(batch_sp_masks)):
                    # 将单个超像素掩码转换为正确的形状 [1, 1, H, W]
                    single_mask = batch_sp_masks[mask_idx].float().unsqueeze(0).unsqueeze(0).to(images.device)
                    
                    # 获取掩码嵌入
                    mask_embeddings, _ = self.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=single_mask
                    )
                    batch_sparse_embeddings.append(mask_embeddings)
            
            # 合并当前批次的所有嵌入
            if batch_sparse_embeddings:
                # 连接所有嵌入，形状为 [1, N, embed_dim]
                batch_embeddings = torch.cat(batch_sparse_embeddings, dim=1)
                all_sparse_embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        if all_sparse_embeddings:
            sparse_embeddings = torch.cat(all_sparse_embeddings, dim=0)
            # print(f"Final sparse embeddings shape: {sparse_embeddings.shape}")
        else:
            sparse_embeddings = None
            # print("No sparse embeddings generated")

        # 解码生成预测
        outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prompt_embeddings=sparse_embeddings,
            target_size=target_size,
        )

        return outputs


# ======= 测试代码示例：三种 Prompt 联合输入 + 可视化 =======
if __name__ == "__main__":
    pass
