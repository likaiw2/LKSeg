from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.loveda_dataset import *
from network.models.SFANet import SFANet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 9e-3
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

save_path = "out"
weights_name = "sfanet"
weights_path = f"model_weights/loveda/{weights_name}_{present_time}"
test_weights_name = "sfanet"
log_name = f'loveda-{weights_name}-{present_time}'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 5
pretrained_ckpt_path = "/home/liw324/code/Segment/SFA-Net/model_weights/loveda/sfanet_20250421_183707/sfanet_epoch34_mIoU_0.5365.pth" # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"model_weights/loveda/delta-0817l0.8lr/delta-0817l0.8lr.ckpt"  # whether continue training with the checkpoint, default None
# strategy = 'None'
strategy = 'ddp'  # default None, if you want to use ddp, please set the gpus to 2 or more

#  define the network
net = SFANet(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


# train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')
train_dataset = LoveDATrainDataset(transform=train_aug, 
                                   data_root='data/LoveDA/Train')

val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val', 
                                 mosaic_ratio=0.0,
                                 transform=val_aug)

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

