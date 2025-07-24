# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ImageEncoder import SwinTransformer
from SemSegHead import MaskFormerHead
from PixelDecoder import BasePixelDecoder
from TransformerDecoder import StandardTransformerDecoder

from _criterion import SetCriterion
from _matcher import HungarianMatcher


class SPMaskFormer(nn.Module):
    """
    SP-MaskFormer model with Swin Transformer backbone, FPN pixel decoder,
    transformer decoder, and semantic segmentation head.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool = True,
        panoptic_on: bool = False,
        instance_on: bool = False,
        test_topk_per_image: int = 100,
    ):
        """
        Args:
            backbone: Swin Transformer backbone
            sem_seg_head: semantic segmentation head with FPN pixel decoder and transformer decoder
            criterion: loss computation module
            num_queries: number of object queries
            object_mask_threshold: threshold for object mask filtering
            overlap_threshold: threshold for overlap filtering
            size_divisibility: input size divisibility requirement
            sem_seg_postprocess_before_inference: whether to postprocess before inference
            pixel_mean: per-channel mean for normalization
            pixel_std: per-channel std for normalization
            semantic_on: whether to output semantic segmentation
            panoptic_on: whether to output panoptic segmentation
            instance_on: whether to output instance segmentation
            test_topk_per_image: keep top-k instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # inference settings
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def build_model(cls, cfg):
        """Build model from config without detectron2 dependency"""
        # Build Swin Transformer backbone
        backbone = SwinTransformer(
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            out_features=cfg.out_features,
        )
        
        # Build semantic segmentation head
        sem_seg_head = MaskFormerHead(
            input_shape=backbone.output_shape(),
            num_classes=cfg.num_classes,
            pixel_decoder=BasePixelDecoder(
                input_shape=backbone.output_shape(),
                conv_dim=cfg.conv_dim,
                mask_dim=cfg.mask_dim,
            ),
            transformer_decoder=StandardTransformerDecoder(
                in_channels=cfg.mask_dim,
                num_classes=cfg.num_classes,
                num_queries=cfg.num_queries,
                nheads=cfg.nheads,
                dim_feedforward=cfg.dim_feedforward,
                dec_layers=cfg.dec_layers,
            ),
        )
        
        # Build criterion
        matcher = HungarianMatcher(
            cost_class=cfg.class_weight,
            cost_mask=cfg.mask_weight,
            cost_dice=cfg.dice_weight,
        )
        
        criterion = SetCriterion(
            num_classes=cfg.num_classes,
            matcher=matcher,
            weight_dict=cfg.weight_dict,
            eos_coef=cfg.no_object_weight,
        )

        return cls(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=cfg.num_queries,
            object_mask_threshold=cfg.object_mask_threshold,
            overlap_threshold=cfg.overlap_threshold,
            size_divisibility=cfg.size_divisibility,
            sem_seg_postprocess_before_inference=cfg.sem_seg_postprocess_before_inference,
            pixel_mean=cfg.pixel_mean,
            pixel_std=cfg.pixel_std,
            semantic_on=cfg.semantic_on,
            instance_on=cfg.instance_on,
            panoptic_on=cfg.panoptic_on,
            test_topk_per_image=cfg.test_topk_per_image,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of DatasetMapper.
                Each item in the list contains the inputs for one image.
        Returns:
            list[dict]: each dict has the results for one image.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = self.pad_images(images)

        # Extract features using Swin Transformer backbone
        features = self.backbone(images)
        
        # Process through semantic segmentation head (FPN + Transformer)
        outputs = self.sem_seg_head(features)

        if self.training:
            # Prepare targets for training
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # Compute losses
            losses = self.criterion(outputs, targets)
            return losses
        else:
            # Inference
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            # Upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image in zip(
                mask_cls_results, mask_pred_results, batched_inputs
            ):
                height = input_per_image.get("height", images.shape[-2])
                width = input_per_image.get("width", images.shape[-1])
                
                # Semantic segmentation inference
                if self.semantic_on:
                    r = self.semantic_inference(mask_cls_result, mask_pred_result)
                    r = F.interpolate(r.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
                    processed_results.append({"sem_seg": r})

            return processed_results

    def pad_images(self, images):
        """Pad images to make them divisible by size_divisibility"""
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        
        # Pad to be divisible by size_divisibility
        stride = self.size_divisibility
        max_size = list(max_size)
        max_size[1] = (max_size[1] + (stride - 1)) // stride * stride
        max_size[2] = (max_size[2] + (stride - 1)) // stride * stride

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        
        return batched_imgs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # Pad ground truth masks
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
