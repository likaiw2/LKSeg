from torch.utils.data import DataLoader
from tools.losses import *
from data_reader.earthvqa_dataset import *
from models.Semantic_FPN import Semantic_FPN
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import datetime
present_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# training hparam
max_epoch = 45
check_val_every_n_epoch = 5
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
weights_path = f"model_weights/earthvqa/{weights_name}_{present_time}"
test_weights_name = "sfanet"
log_name = f'earthvqa-{weights_name}-{present_time}'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 5
pretrained_ckpt_path = None
# pretrained_ckpt_path = "/home/liw324/code/Segment/SFA-Net/model_weights/earthvqa/sfanet_20250507_231832/sfanet_best.pth" # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"model_weights/earthvqa/delta-0817l0.8lr/delta-0817l0.8lr.ckpt"  # whether continue training with the checkpoint, default None
# strategy = 'None'
strategy = 'ddp'  # default None, if you want to use ddp, please set the gpus to 2 or more

#  define the network
net = Semantic_FPN(num_classes=num_classes)

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

def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


# train_dataset = earthvqaTrainDataset(transform=train_aug, data_root='data/earthvqa/train_val')
train_dataset = EarthVQADataset(transform=train_aug, 
                                data_root='data/EarthVQA/Train'
                                )

val_dataset = EarthVQADataset(transform=val_aug,
                              data_root='data/EarthVQA/Val', 
                              )

# test_dataset = earthvqaTestDataset()
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

