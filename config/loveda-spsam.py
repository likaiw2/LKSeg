from torch.utils.data import DataLoader
from tools.losses import *
from data_reader.loveda_dataset import LoveDATrainDataset,CLASSES
from models.sp_sam import SPSam
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 2
lr = 9e-3
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

save_path = "#out"
weights_name = "sp_sam"
weights_path = f"model_weights/loveda/{weights_name}_{present_time}"
# test_weights_name = "sfanet"
log_name = f'loveda-{weights_name}-{present_time}'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"model_weights/loveda/delta-0817l0.8lr/delta-0817l0.8lr.ckpt"  # whether continue training with the checkpoint, default None
# strategy = 'None'
# strategy = 'ddp'  # default None, if you want to use ddp, please set the gpus to 2 or more

#  define the network
net = SPSam(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True


# train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')
train_dataset = LoveDATrainDataset(data_root='data/LoveDA/Train')

val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val', 
                                 mosaic_ratio=0.0)

# test_dataset = LoveDATestDataset()
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

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

