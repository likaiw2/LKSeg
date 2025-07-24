import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backbone import SwinTransformer
from pixel_decoder import BasePixelDecoder, MSDeformAttnPixelDecoder
from transformer_decoder import StandardTransformerDecoder, MultiScaleMaskedTransformerDecoder
from mask_former_head import MaskFormerHead
from mask2former import Mask2Former
from utils import ShapeSpec


def create_swin_backbone():
    """Create Swin Transformer backbone"""
    backbone = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_features=["res2", "res3", "res4", "res5"]
    )
    return backbone


def create_pixel_decoder(input_shape, decoder_type="base"):
    """Create pixel decoder"""
    if decoder_type == "base":
        return BasePixelDecoder(
            input_shape=input_shape,
            conv_dim=256,
            mask_dim=256,
            norm="GN"
        )
    elif decoder_type == "msdeform":
        return MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
        )


def create_transformer_decoder(decoder_type="standard"):
    """Create transformer decoder"""
    if decoder_type == "standard":
        return StandardTransformerDecoder(
            in_channels=256,
            mask_classification=True,
            num_classes=150,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dropout=0.1,
            dim_feedforward=2048,
            enc_layers=6,
            dec_layers=6,
            pre_norm=False,
            deep_supervision=True,
            mask_dim=256,
            enforce_input_project=False,
        )
    elif decoder_type == "multiscale":
        return MultiScaleMaskedTransformerDecoder(
            in_channels=256,
            mask_classification=True,
            num_classes=150,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
        )


def test_backbone():
    """Test backbone functionality"""
    print("=" * 50)
    print("Testing Swin Transformer Backbone...")
    
    try:
        backbone = create_swin_backbone()
        print(f"✅ Backbone created successfully!")
        print(f"Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        print(f"Input shape: {x.shape}")
        
        with torch.no_grad():
            features = backbone(x)
            
        print("Output features:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
            
        # Get output shapes
        output_shapes = backbone.output_shape()
        print("Output shapes:")
        for name, shape in output_shapes.items():
            print(f"  {name}: channels={shape.channels}, stride={shape.stride}")
            
        print("✅ Backbone test passed!")
        return backbone, output_shapes
        
    except Exception as e:
        print(f"❌ Backbone test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_pixel_decoder(input_shape, decoder_type="base"):
    """Test pixel decoder functionality"""
    print("=" * 50)
    print(f"Testing {decoder_type.upper()} Pixel Decoder...")
    
    try:
        pixel_decoder = create_pixel_decoder(input_shape, decoder_type)
        print(f"✅ Pixel decoder created successfully!")
        print(f"Parameters: {sum(p.numel() for p in pixel_decoder.parameters()):,}")
        
        # Create mock features
        features = {}
        for name, shape in input_shape.items():
            # Calculate spatial size based on stride
            spatial_size = 224 // shape.stride
            features[name] = torch.randn(2, shape.channels, spatial_size, spatial_size)
            print(f"Mock feature {name}: {features[name].shape}")
        
        with torch.no_grad():
            mask_features, transformer_features, multi_scale_features = pixel_decoder(features)
            
        print("Pixel decoder outputs:")
        print(f"  Mask features: {mask_features.shape}")
        print(f"  Transformer features: {transformer_features}")
        print(f"  Multi-scale features: {len(multi_scale_features)} scales")
        for i, feat in enumerate(multi_scale_features):
            print(f"    Scale {i}: {feat.shape}")
            
        print("✅ Pixel decoder test passed!")
        return pixel_decoder
        
    except Exception as e:
        print(f"❌ Pixel decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_transformer_decoder(decoder_type="standard"):
    """Test transformer decoder functionality"""
    print("=" * 50)
    print(f"Testing {decoder_type.upper()} Transformer Decoder...")
    
    try:
        transformer_decoder = create_transformer_decoder(decoder_type)
        print(f"✅ Transformer decoder created successfully!")
        print(f"Parameters: {sum(p.numel() for p in transformer_decoder.parameters()):,}")
        
        # Create mock inputs
        if decoder_type == "standard":
            # Standard decoder expects single feature map
            x = torch.randn(2, 256, 56, 56)
            mask_features = torch.randn(2, 256, 56, 56)
            print(f"Input feature: {x.shape}")
            print(f"Mask features: {mask_features.shape}")
            
            with torch.no_grad():
                outputs = transformer_decoder(x, mask_features)
                
        elif decoder_type == "multiscale":
            # Multi-scale decoder expects list of features
            multi_scale_features = [
                torch.randn(2, 256, 7, 7),
                torch.randn(2, 256, 14, 14),
                torch.randn(2, 256, 28, 28)
            ]
            mask_features = torch.randn(2, 256, 56, 56)
            print(f"Multi-scale features:")
            for i, feat in enumerate(multi_scale_features):
                print(f"  Scale {i}: {feat.shape}")
            print(f"Mask features: {mask_features.shape}")
            
            with torch.no_grad():
                outputs = transformer_decoder(multi_scale_features, mask_features)
        
        print("Transformer decoder outputs:")
        for key, value in outputs.items():
            if key == "aux_outputs":
                print(f"  {key}: {len(value)} auxiliary outputs")
                if len(value) > 0:
                    for aux_key, aux_value in value[0].items():
                        print(f"    {aux_key}: {aux_value.shape}")
            else:
                print(f"  {key}: {value.shape}")
                
        print("✅ Transformer decoder test passed!")
        return transformer_decoder
        
    except Exception as e:
        print(f"❌ Transformer decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_mask_former_head(input_shape):
    """Test MaskFormer head"""
    print("=" * 50)
    print("Testing MaskFormer Head...")
    
    try:
        # Create components
        pixel_decoder = create_pixel_decoder(input_shape, "base")
        transformer_decoder = create_transformer_decoder("multiscale")
        
        head = MaskFormerHead(
            input_shape=input_shape,
            num_classes=150,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            transformer_in_feature="multi_scale_pixel_decoder",
        )
        
        print(f"✅ MaskFormer head created successfully!")
        print(f"Parameters: {sum(p.numel() for p in head.parameters()):,}")
        
        # Create mock features
        features = {}
        for name, shape in input_shape.items():
            spatial_size = 224 // shape.stride
            features[name] = torch.randn(2, shape.channels, spatial_size, spatial_size)
        
        with torch.no_grad():
            outputs = head(features)
            
        print("MaskFormer head outputs:")
        for key, value in outputs.items():
            if key == "aux_outputs":
                print(f"  {key}: {len(value)} auxiliary outputs")
            else:
                print(f"  {key}: {value.shape}")
                
        print("✅ MaskFormer head test passed!")
        return head
        
    except Exception as e:
        print(f"❌ MaskFormer head test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_full_mask2former():
    """Test complete Mask2Former model"""
    print("=" * 50)
    print("Testing Complete Mask2Former Model...")
    
    try:
        # Create backbone
        backbone = create_swin_backbone()
        input_shape = backbone.output_shape()
        
        # Create head
        pixel_decoder = create_pixel_decoder(input_shape, "msdeform")
        transformer_decoder = create_transformer_decoder("multiscale")
        
        head = MaskFormerHead(
            input_shape=input_shape,
            num_classes=150,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            transformer_in_feature="multi_scale_pixel_decoder",
        )
        
        # Create full model
        model = Mask2Former(
            backbone=backbone,
            sem_seg_head=head,
            criterion=None,  # No criterion for inference
            num_queries=100,
            object_mask_threshold=0.25,
            overlap_threshold=0.8,
            metadata=None,
            size_divisibility=32,
            sem_seg_postprocess_before_inference=True,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            test_topk_per_image=100,
        )
        
        print(f"✅ Complete Mask2Former created successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference mode
        model.eval()
        
        # Create mock batch input (inference format)
        batched_inputs = [
            {
                "image": torch.randn(3, 224, 224),
                "height": 224,
                "width": 224,
            },
            {
                "image": torch.randn(3, 224, 224), 
                "height": 224,
                "width": 224,
            }
        ]
        
        print(f"Input batch size: {len(batched_inputs)}")
        print(f"Input image shape: {batched_inputs[0]['image'].shape}")
        
        with torch.no_grad():
            # Test inference
            outputs = model(batched_inputs)
            
        print("Mask2Former inference outputs:")
        print(f"  Number of results: {len(outputs)}")
        for i, result in enumerate(outputs):
            print(f"  Result {i}:")
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
                elif isinstance(value, tuple) and len(value) == 2:
                    print(f"    {key}: ({value[0].shape}, {len(value[1])} segments)")
                else:
                    print(f"    {key}: {type(value)}")
                    
        print("✅ Complete Mask2Former test passed!")
        return model
        
    except Exception as e:
        print(f"❌ Complete Mask2Former test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_training_mode():
    """Test Mask2Former in training mode"""
    print("=" * 50)
    print("Testing Training Mode...")
    
    try:
        # Create model with criterion
        backbone = create_swin_backbone()
        input_shape = backbone.output_shape()
        
        pixel_decoder = create_pixel_decoder(input_shape, "msdeform")
        transformer_decoder = create_transformer_decoder("multiscale")
        head = MaskFormerHead(
            input_shape=input_shape,
            num_classes=150,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            transformer_in_feature="multi_scale_pixel_decoder",
        )
        
        # Create a mock criterion that accepts the expected inputs
        class MockCriterion:
            def __init__(self):
                self.weight_dict = {"loss_ce": 1.0, "loss_mask": 1.0}
            
            def __call__(self, outputs, targets):
                # Return mock losses
                return {
                    "loss_ce": torch.tensor(1.0),
                    "loss_mask": torch.tensor(2.0),
                }
        
        model = Mask2Former(
            backbone=backbone,
            sem_seg_head=head,
            criterion=MockCriterion(),
            num_queries=100,
            object_mask_threshold=0.25,
            overlap_threshold=0.8,
            metadata=None,
            size_divisibility=32,
            sem_seg_postprocess_before_inference=True,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            test_topk_per_image=100,
        )
        
        model.train()  # Set to training mode
        
        # Create training batch without instances (will use targets=None)
        batched_inputs = [
            {
                "image": torch.randn(3, 224, 224),
                "height": 224,
                "width": 224,
            }
        ]
        
        # Test forward pass
        outputs = model(batched_inputs)
        
        print("Training outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")
        
        print("✅ Training mode test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training mode test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting Mask2Former Tests...")
    print("=" * 80)
    
    # Test 1: Backbone
    backbone, input_shape = test_backbone()
    if backbone is None or input_shape is None:
        print("❌ Backbone test failed, stopping tests")
        return
    
    # Test 2: Pixel Decoders
    base_decoder = test_pixel_decoder(input_shape, "base")
    if base_decoder is None:
        print("❌ Base pixel decoder test failed")
    
    # Test 3: Transformer Decoders  
    standard_decoder = test_transformer_decoder("standard")
    if standard_decoder is None:
        print("❌ Standard transformer decoder test failed")
        
    multiscale_decoder = test_transformer_decoder("multiscale")
    if multiscale_decoder is None:
        print("❌ Multi-scale transformer decoder test failed")
    
    # Test 4: MaskFormer Head
    head = test_mask_former_head(input_shape)
    if head is None:
        print("❌ MaskFormer head test failed")
    
    # Test 5: Complete Model
    model = test_full_mask2former()
    if model is None:
        print("❌ Complete Mask2Former test failed")
    
    # Test 6: Training Mode
    if not test_training_mode():
        print("❌ Training mode test failed")
    
    print("=" * 80)
    print("All tests completed!")
    
    # Summary
    success_count = sum([
        backbone is not None,
        base_decoder is not None,
        standard_decoder is not None,
        multiscale_decoder is not None,
        head is not None,
        model is not None,
        test_training_mode()
    ])
    
    print(f"✅ {success_count}/7 tests passed")
    
    if success_count == 7:
        print("🎉 All Mask2Former components working correctly!")
    else:
        print("⚠️  Some components need attention")


if __name__ == "__main__":
    main()
