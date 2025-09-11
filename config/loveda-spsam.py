from torch.utils.data import DataLoader
from tools.losses import *
from data_reader.loveda_dataset import LoveDATrainDataset
from models.spsam import SPSam, build_spsam
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
import torch
import torch.nn.functional as F

present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# ------------------------------------------
# Training Hyperparameters
# ------------------------------------------
CLASSES = LoveDATrainDataset().CLASSES
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 2  # 较小的batch size，因为SAM比较大
val_batch_size = 2
lr = 1e-4  # 较小的学习率，因为使用预训练的SAM
weight_decay = 0.01
backbone_lr = 1e-5  # SAM backbone使用更小的学习率
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

# ------------------------------------------
# SPSam特定参数
# ------------------------------------------
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # SAM权重路径
sam_model_type = "vit_h"  # 可选: "vit_h", "vit_l", "vit_b"
n_segments = 100  # 超像素数量
compactness = 10.0  # 超像素紧密度
points_per_batch = 32  # 每批处理的点数
pred_iou_thresh = 0.5  # IoU阈值
multimask_output = False  # 训练时建议False

# ------------------------------------------
# 损失函数权重
# ------------------------------------------
loss_weights = {
    "loss_ce": 2.0,      # 分类损失权重
    "loss_mask": 5.0,    # 掩码损失权重  
    "loss_dice": 5.0,    # Dice损失权重
}

# Hungarian匹配器参数
matcher_costs = {
    "cost_class": 2.0,   # 分类成本
    "cost_mask": 5.0,    # 掩码成本
    "cost_dice": 5.0,    # Dice成本
}

# ------------------------------------------
# Logging and Saving Settings
# ------------------------------------------
save_path = "out"
model_name = "spsam"
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
resume_ckpt_path = None

# ------------------------------------------
# 构建SPSam模型
# ------------------------------------------
def create_spsam_model():
    """创建SPSam模型"""
    # 构建基础SPSam模型
    spsam = build_spsam(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        n_segments=n_segments,
        compactness=compactness,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 更新模型参数
    spsam.num_classes = num_classes
    spsam.points_per_batch = points_per_batch
    spsam.pred_iou_thresh = pred_iou_thresh
    spsam.multimask_output = multimask_output
    
    # 重新初始化分类头以匹配类别数
    spsam.class_embed = torch.nn.Linear(256, num_classes + 1)
    
    # 更新损失函数权重
    if hasattr(spsam, 'criterion'):
        spsam.criterion.weight_dict = loss_weights
        # 更新匹配器成本
        if hasattr(spsam.criterion, 'matcher'):
            spsam.criterion.matcher.cost_class = matcher_costs["cost_class"]
            spsam.criterion.matcher.cost_mask = matcher_costs["cost_mask"] 
            spsam.criterion.matcher.cost_dice = matcher_costs["cost_dice"]
    
    return spsam

# 创建网络
net = create_spsam_model()

# ------------------------------------------
# 损失函数（SPSam内置损失）
# ------------------------------------------
# SPSam使用内置的criterion，这里定义一个兼容的损失函数用于验证
class SPSamCompatibleLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, pred_logits, targets):
        """兼容性损失函数，用于验证阶段"""
        return self.ce_loss(pred_logits, targets)

loss = SPSamCompatibleLoss(ignore_index=ignore_index)
use_aux_loss = False

# ------------------------------------------
# 优化器设置
# ------------------------------------------
def setup_optimizer(model):
    """设置优化器，对SAM backbone使用较小学习率"""
    # 分层参数设置
    sam_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'sam_model' in name:
            # SAM模型参数使用较小学习率
            sam_params.append(param)
        else:
            # 其他参数（分类头等）使用正常学习率
            other_params.append(param)
    
    # 创建参数组
    param_groups = [
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': sam_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay}
    ]
    
    base_optimizer = torch.optim.AdamW(param_groups)
    return Lookahead(base_optimizer)

optimizer = setup_optimizer(net)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# ------------------------------------------
# 数据加载器设置
# ------------------------------------------
train_dataset = LoveDATrainDataset(
    data_root='data/LoveDA/Train',
    output_size=[1024, 1024]
)

val_dataset = LoveDATrainDataset(
    data_root='data/LoveDA/Val',
    output_size=[1024, 1024]  # 与训练阶段保持一致
)

test_dataset = val_dataset

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=2,  # 较少的worker，因为SPSam处理较慢
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# ------------------------------------------
# 模型特定设置
# ------------------------------------------
def print_model_info():
    """打印模型信息"""
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"SPSam Model Info:")
    print(f"  SAM Model Type: {sam_model_type}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Superpixel Segments: {n_segments}")
    print(f"  Points per Batch: {points_per_batch}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Loss Weights: {loss_weights}")

# 在训练开始时调用
if __name__ == "__main__":
    print_model_info()