import torch
import torch.nn as nn
import sys
import os

# Add the project root to Python path
sys.path.append('/home/likai/code/LKSeg')

def create_mock_shape_spec(channels, stride):
    """Create a mock ShapeSpec object"""
    class MockShapeSpec:
        def __init__(self, channels, stride):
            self.channels = channels
            self.stride = stride
    return MockShapeSpec(channels, stride)

def create_mock_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm=None, activation=None):
    """Create a mock Conv2d to replace detectron2.layers.Conv2d"""
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if activation is not None:
        return nn.Sequential(conv, nn.ReLU())
    return conv

def mock_get_norm(norm_type, channels):
    """Mock get_norm function"""
    if norm_type == "BN":
        return nn.BatchNorm2d(channels)
    elif norm_type == "GN":
        return nn.GroupNorm(32, channels)
    return None

def mock_c2_xavier_fill(module):
    """Mock weight initialization"""
    if hasattr(module, 'weight'):
        nn.init.xavier_uniform_(module.weight)

# Mock detectron2 dependencies
import types
mock_weight_init = types.ModuleType('weight_init')
mock_weight_init.c2_xavier_fill = mock_c2_xavier_fill

sys.modules['detectron2.config'] = types.ModuleType('config')
sys.modules['detectron2.layers'] = types.ModuleType('layers')
sys.modules['detectron2.modeling'] = types.ModuleType('modeling')
sys.modules['fvcore.nn.weight_init'] = mock_weight_init

# Add mock functions to modules
sys.modules['detectron2.layers'].Conv2d = create_mock_conv2d
sys.modules['detectron2.layers'].get_norm = mock_get_norm
sys.modules['detectron2.layers'].ShapeSpec = create_mock_shape_spec

# Mock registry
class MockRegistry:
    def register(self):
        def decorator(cls):
            return cls
        return decorator
    
    def get(self, name):
        return None

sys.modules['detectron2.modeling'].SEM_SEG_HEADS_REGISTRY = MockRegistry()

# Mock configurable decorator
def configurable(func):
    return func

sys.modules['detectron2.config'].configurable = configurable

# Now import the BasePixelDecoder
from models.SP_maskformer.PixelDecoder import BasePixelDecoder

def test_base_pixel_decoder():
    """Test BasePixelDecoder functionality"""
    print("Testing BasePixelDecoder...")
    
    # Create mock input shapes (simulating backbone output)
    input_shape = {
        "res2": create_mock_shape_spec(channels=256, stride=4),
        "res3": create_mock_shape_spec(channels=512, stride=8), 
        "res4": create_mock_shape_spec(channels=1024, stride=16),
        "res5": create_mock_shape_spec(channels=2048, stride=32),
    }
    
    # Create model
    try:
        model = BasePixelDecoder(
            input_shape=input_shape,
            conv_dim=256,
            mask_dim=256,
            norm="BN"
        )
        print("✅ Model created successfully!")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Create mock feature maps
    batch_size = 2
    features = {
        "res2": torch.randn(batch_size, 256, 56, 56),   # stride 4
        "res3": torch.randn(batch_size, 512, 28, 28),   # stride 8
        "res4": torch.randn(batch_size, 1024, 14, 14),  # stride 16
        "res5": torch.randn(batch_size, 2048, 7, 7),    # stride 32
    }
    
    model.eval()
    with torch.no_grad():
        try:
            # Test forward_features method
            mask_features, transformer_features, multi_scale_features = model.forward_features(features)
            
            print("✅ Forward pass successful!")
            print(f"Mask features shape: {mask_features.shape}")
            print(f"Transformer features: {transformer_features}")
            print(f"Multi-scale features count: {len(multi_scale_features)}")
            
            for i, feat in enumerate(multi_scale_features):
                print(f"  Scale {i}: {feat.shape}")
            
            # Test full forward method
            output = model.forward(features)
            print(f"Full forward output: tuple with {len(output)} elements")
            
            # Process each element in the tuple
            for i, item in enumerate(output):
                if item is None:
                    print(f"  Element {i}: None")
                elif isinstance(item, torch.Tensor):
                    print(f"  Element {i}: tensor {item.shape}")
                elif isinstance(item, list):
                    print(f"  Element {i}: list with {len(item)} tensors")
                    for j, tensor in enumerate(item):
                        print(f"    [{j}]: {tensor.shape}")
                else:
                    print(f"  Element {i}: {type(item)}")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test model components
    print(f"\nModel structure:")
    print(f"- Input features: {model.in_features}")
    print(f"- Conv dim: {model.mask_dim}")
    print(f"- Mask dim: {model.mask_dim}")
    print(f"- Feature levels: {model.maskformer_num_feature_levels}")
    print(f"- Lateral convs: {len(model.lateral_convs)}")
    print(f"- Output convs: {len(model.output_convs)}")
    
    print(f"\nBasePixelDecoder test completed!")

if __name__ == "__main__":
    test_base_pixel_decoder()
