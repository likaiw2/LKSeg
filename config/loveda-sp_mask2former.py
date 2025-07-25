from torch.utils.data import DataLoader
from tools.losses import *
from data_reader.loveda_dataset import LoveDATrainDataset,CLASSES
from models.sp_mask2former.sp_mask2former import SPSam
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

from models.sp_mask2former.backbone import SwinTransformer
from models.sp_mask2former.pixel_decoder import BasePixelDecoder, MSDeformAttnPixelDecoder
from models.sp_mask2former.transformer_decoder import StandardTransformerDecoder, MultiScaleMaskedTransformerDecoder
from models.sp_mask2former.mask_former_head import SPMaskFormerHead
from models.sp_mask2former.sp_mask2former import Mask2Former
from models.sp_mask2former.utils import ShapeSpec

# ------------------------------------------
# Training Hyperparameters
# ------------------------------------------
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 9e-3
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

# ------------------------------------------
# Logging and Saving Settings
# ------------------------------------------
save_path = "out"
model_name = "sp_mask2former"
dataset_name = "loveda"
weights_path = f"{save_path}/model_weights/{dataset_name}/{model_name}_{present_time}"
log_name = f'{dataset_name}-{model_name}'
check_val_every_n_epoch = 1
save_top_k = 1
save_last = True
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2

# ------------------------------------------
# Model and Optimizer Settings
# ------------------------------------------
monitor = 'val_mIoU'
monitor_mode = 'max'

pretrained_ckpt_path = None
resume_ckpt_path = None #"model_weights/loveda/delta-0817l0.8lr/delta-0817l0.8lr.ckpt"  # whether continue training with the checkpoint, default None

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

#  define the network
# Create model with criterion
backbone = create_swin_backbone()
input_shape = backbone.output_shape()

pixel_decoder = create_pixel_decoder(input_shape, "msdeform")
transformer_decoder = create_transformer_decoder("multiscale")
head = SPMaskFormerHead(
    input_shape=input_shape,
    num_classes=100,
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

net = Mask2Former(
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

# define the loss
from tools.losses import QuerySegmentationLoss
loss = QuerySegmentationLoss(num_classes=num_classes, ignore_index=ignore_index)
use_aux_loss = False  # 设置为False，因为我们现在只有一条预测路径

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# ------------------------------------------
# Dataloader Settings
# ------------------------------------------
train_dataset = LoveDATrainDataset(data_root='data/LoveDA/Train',
                                   superpixel=True)
val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val',
                                 superpixel=True)
test_dataset = val_dataset

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)