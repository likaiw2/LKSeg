from torch.utils.data import DataLoader
from tools.losses import *
from torch import nn
from data_reader.loveda_dataset import LoveDATrainDataset
from models.MFANet import MFANet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=0):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, output, target):
        # 确保目标张量中的值在有效范围内
        # 打印目标张量的最小值和最大值，用于调试
        min_val = target.min().item()
        max_val = target.max().item()
        # print(f"Target min: {min_val}, max: {max_val}, output classes: {output.size(1)}")
        
        return F.cross_entropy(output, target, weight=self.weight, ignore_index=self.ignore_index)


# ------------------------------------------
# Training Hyperparameters
# ------------------------------------------
max_epoch = 100
ignore_index = 0
train_batch_size = 4
val_batch_size = 4
lr = 9e-3
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
classes = LoveDATrainDataset.CLASSES
num_classes = len(classes)


# ------------------------------------------
# Logging and Saving Settings
# ------------------------------------------
save_path = "out"
model_name = "mfanet"
dataset_name = "loveda"
weights_path = f"{save_path}/model_weights/loveda/{model_name}_{present_time}"
log_name = f'{dataset_name}-{model_name}'
check_val_every_n_epoch = 10
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

#  define the network
net = MFANet(num_classes=num_classes)

# define the loss
loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# ------------------------------------------
# Dataloader Settings
# ------------------------------------------
train_dataset = LoveDATrainDataset(data_root='data/LoveDA/Train')
val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val')
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

