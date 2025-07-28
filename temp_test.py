import torch
import torch.nn as nn
import sys
import os

# Add the project root to Python path
sys.path.append('/home/likai/code/LKSeg')

def test_device_and_dtype():
    """Test device and dtype consistency"""
    print("=" * 50)
    print("Testing Device and Dtype Consistency")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test basic tensor operations
    print("\n" + "=" * 30)
    print("Basic Tensor Tests")
    print("=" * 30)
    
    # Create test tensors
    x_cpu = torch.randn(2, 3, 224, 224)
    print(f"CPU tensor - dtype: {x_cpu.dtype}, device: {x_cpu.device}")
    
    x_cuda = x_cpu.to(device)
    print(f"CUDA tensor - dtype: {x_cuda.dtype}, device: {x_cuda.device}")
    
    # Test model creation
    print("\n" + "=" * 30)
    print("Model Creation Tests")
    print("=" * 30)
    
    try:
        # Import and test backbone
        from models.sp_mask2former.backbone import SwinTransformer
        
        backbone = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            out_features=["res2", "res3", "res4", "res5"]
        )
        
        print(f"✅ Backbone created successfully")
        
        # Check backbone parameters
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                print(f"Param {name}: dtype={param.dtype}, device={param.device}")
                break  # Just check first parameter
        
        # Move backbone to device
        backbone = backbone.to(device)
        print(f"✅ Backbone moved to {device}")
        
        # Check after moving to device
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                print(f"After .to(device) - Param {name}: dtype={param.dtype}, device={param.device}")
                break
        
        # Test forward pass
        print("\n" + "=" * 30)
        print("Forward Pass Tests")
        print("=" * 30)
        
        test_input = torch.randn(1, 3, 224, 224)
        print(f"Test input - dtype: {test_input.dtype}, device: {test_input.device}")
        
        test_input = test_input.to(device)
        print(f"Test input after .to(device) - dtype: {test_input.dtype}, device: {test_input.device}")
        
        backbone.eval()
        with torch.no_grad():
            features = backbone(test_input)
            print(f"✅ Backbone forward pass successful")
            
            for name, feat in features.items():
                print(f"Feature {name}: shape={feat.shape}, dtype={feat.dtype}, device={feat.device}")
        
    except Exception as e:
        print(f"❌ Backbone test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test pixel decoder
    print("\n" + "=" * 30)
    print("Pixel Decoder Tests")
    print("=" * 30)
    
    try:
        from models.sp_mask2former.pixel_decoder import BasePixelDecoder
        from models.sp_mask2former.utils import ShapeSpec
        
        # Create input shape
        input_shape = {}
        for name, feat in features.items():
            input_shape[name] = ShapeSpec(channels=feat.shape[1], stride=4 * (2 ** list(features.keys()).index(name)))
        
        pixel_decoder = BasePixelDecoder(
            input_shape=input_shape,
            conv_dim=256,
            mask_dim=256,
            norm="GN"
        )
        
        print(f"✅ Pixel decoder created successfully")
        
        # Check pixel decoder parameters
        for name, param in pixel_decoder.named_parameters():
            if param.requires_grad:
                print(f"PixelDecoder param {name}: dtype={param.dtype}, device={param.device}")
                break
        
        pixel_decoder = pixel_decoder.to(device)
        print(f"✅ Pixel decoder moved to {device}")
        
        # Check after moving to device
        for name, param in pixel_decoder.named_parameters():
            if param.requires_grad:
                print(f"After .to(device) - PixelDecoder param {name}: dtype={param.dtype}, device={param.device}")
                break
        
        # Test forward pass
        pixel_decoder.eval()
        with torch.no_grad():
            mask_features, transformer_features, multi_scale_features = pixel_decoder(features)
            print(f"✅ Pixel decoder forward pass successful")
            print(f"Mask features: shape={mask_features.shape}, dtype={mask_features.dtype}, device={mask_features.device}")
        
    except Exception as e:
        print(f"❌ Pixel decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 50)
    print("Testing Data Loading")
    print("=" * 50)
    
    try:
        # Import config
        from config.loveda_sp_mask2former import config
        
        # Test train loader
        train_loader = config.train_loader
        print(f"✅ Train loader created successfully")
        
        # Get one batch
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Test device transfer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = batch['img'].to(device).float()
            mask = batch['gt_semantic_seg'].to(device)
            
            print(f"After .to(device):")
            print(f"  img: shape={img.shape}, dtype={img.dtype}, device={img.device}")
            print(f"  mask: shape={mask.shape}, dtype={mask.dtype}, device={mask.device}")
            
            break  # Only test first batch
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_model():
    """Test full model"""
    print("\n" + "=" * 50)
    print("Testing Full Model")
    print("=" * 50)
    
    try:
        from config.loveda_sp_mask2former import config
        
        model = config.net
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"✅ Model created successfully")
        
        # Check model parameters before moving to device
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Model param {name}: dtype={param.dtype}, device={param.device}")
                break
        
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        
        # Check model parameters after moving to device
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"After .to(device) - Model param {name}: dtype={param.dtype}, device={param.device}")
                break
        
        # Test with dummy input
        model.eval()
        
        # Create dummy batched_inputs
        batched_inputs = [
            {
                "image": torch.randn(3, 224, 224, device=device, dtype=torch.float32),
                "height": 224,
                "width": 224,
            }
        ]
        
        print(f"Test input dtype: {batched_inputs[0]['image'].dtype}")
        print(f"Test input device: {batched_inputs[0]['image'].device}")
        
        with torch.no_grad():
            outputs = model(batched_inputs)
            print(f"✅ Full model forward pass successful")
            print(f"Output type: {type(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pixel_decoder_detailed():
    """详细测试pixel decoder"""
    print("\n" + "=" * 30)
    print("Detailed Pixel Decoder Tests")
    print("=" * 30)
    
    try:
        from models.sp_mask2former.backbone import SwinTransformer
        from models.sp_mask2former.pixel_decoder import BasePixelDecoder
        from models.sp_mask2former.utils import ShapeSpec
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建backbone并获取features
        backbone = SwinTransformer(
            img_size=224, patch_size=4, in_chans=3, num_classes=1000,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, out_features=["res2", "res3", "res4", "res5"]
        ).to(device)
        
        test_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            features = backbone(test_input)
        
        # 创建input_shape
        input_shape = {}
        for name, feat in features.items():
            input_shape[name] = ShapeSpec(channels=feat.shape[1], stride=4 * (2 ** list(features.keys()).index(name)))
            print(f"Input shape {name}: channels={feat.shape[1]}, stride={4 * (2 ** list(features.keys()).index(name))}")
        
        # 创建pixel decoder
        pixel_decoder = BasePixelDecoder(
            input_shape=input_shape,
            conv_dim=256,
            mask_dim=256,
            norm="GN"
        )
        
        print(f"✅ Pixel decoder created")
        
        # 检查pixel decoder的组件
        print(f"Lateral convs: {len(pixel_decoder.lateral_convs)}")
        print(f"Output convs: {len(pixel_decoder.output_convs)}")
        print(f"Mask features: {pixel_decoder.mask_features}")
        
        # 检查每个lateral conv
        for i, conv in enumerate(pixel_decoder.lateral_convs):
            print(f"Lateral conv {i}: {conv}")
            if conv is None:
                print(f"❌ Lateral conv {i} is None!")
        
        # 检查每个output conv  
        for i, conv in enumerate(pixel_decoder.output_convs):
            print(f"Output conv {i}: {conv}")
            if conv is None:
                print(f"❌ Output conv {i} is None!")
        
        pixel_decoder = pixel_decoder.to(device)
        print(f"✅ Pixel decoder moved to device")
        
        # 尝试forward pass
        pixel_decoder.eval()
        with torch.no_grad():
            try:
                mask_features, transformer_features, multi_scale_features = pixel_decoder(features)
                print(f"✅ Pixel decoder forward pass successful")
            except Exception as e:
                print(f"❌ Pixel decoder forward failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Detailed pixel decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("Starting Comprehensive Tests...")
    
    # Test 1: Device and dtype consistency
    if not test_device_and_dtype():
        print("❌ Device/dtype test failed, stopping")
        return
    
    # Test 2: Data loading
    if not test_data_loading():
        print("❌ Data loading test failed, stopping")
        return
    
    # Test 3: Full model
    if not test_full_model():
        print("❌ Full model test failed")
        return
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)

def main():
    """Run detailed tests"""
    print("Starting Detailed Pixel Decoder Tests...")
    test_pixel_decoder_detailed()

if __name__ == "__main__":
    main()
