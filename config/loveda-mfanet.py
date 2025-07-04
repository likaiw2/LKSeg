from torch.utils.data import DataLoader
from tools.losses import *
from data_reader.loveda_dataset import *
from models.MFANet import MFANet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

class CrossEntropy2d(nn.Module):
    """ 2D version of the cross entropy loss """
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropy2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        dim = input.dim()
        if dim == 2:
            return F.cross_entropy(input, target, weight=self.weight, 
                                  size_average=self.size_average, 
                                  ignore_index=self.ignore_index)
        elif dim == 4:
            output = input.view(input.size(0), input.size(1), -1)
            output = torch.transpose(output, 1, 2).contiguous()
            output = output.view(-1, output.size(2))
            target = target.view(-1)
            return F.cross_entropy(output, target, weight=self.weight, 
                                  size_average=self.size_average,
                                  ignore_index=self.ignore_index)
        else:
            raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


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
model_name = "mfanet"
dataset_name = "loveda"
weights_path = f"{save_path}/model_weights/loveda/{model_name}_{present_time}"
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

#  define the network
net = MFANet(num_classes=num_classes)

# define the loss
loss = CrossEntropy2d(ignore_index=ignore_index)
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

