import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
import random
import wandb
from tools.cfg import py2cfg
from tqdm import tqdm

os.environ["WANDB_MODE"] = "offline"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", 
        # required=True,
        # default='config/loveda/sfanet.py',
        default='config/earthvqa/sfanet.py',
        )
    return parser.parse_args()

class Supervision_Train(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

def train_one_epoch(model, loader, optimizer, loss_fn, device,epoch):
    model.train()
    total_loss = 0
    metrics = Evaluator(num_class=model.config.num_classes)
    for batch in tqdm(loader, desc=f"Train Epoch {epoch}"):
        img = batch['img'].to(device)
        mask = batch['gt_semantic_seg'].to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_mask = nn.Softmax(dim=1)(pred[0]).argmax(dim=1)
        for i in range(mask.size(0)):
            metrics.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

    avg_loss = total_loss / len(loader)
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    wandb.log({'train_loss': avg_loss, 'train_mIoU': mIoU, 'train_OA': OA})
    return avg_loss

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics = Evaluator(num_class=model.config.num_classes)
    with torch.no_grad():
        for batch in loader:
            img = batch['img'].to(device)
            mask = batch['gt_semantic_seg'].to(device)

            pred = model(img)
            loss = loss_fn(pred, mask)
            total_loss += loss.item()

            pred_mask = nn.Softmax(dim=1)(pred).argmax(dim=1)
            for i in range(mask.size(0)):
                metrics.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

    avg_loss = total_loss / len(loader)
    mIoU = np.nanmean(metrics.Intersection_over_Union())
    OA = np.nanmean(metrics.OA())
    wandb.log({'val_loss': avg_loss, 'val_mIoU': mIoU, 'val_OA': OA})
    return avg_loss, mIoU, OA

def main():
    args = get_args()
    
    best_mIoU = 0
    best_ckpt_path = ""

    config = py2cfg(args.config_path)
    
    seed_everything(42)

    # 初始化 wandb
    wandb.init(project=config.log_name, 
               config=vars(config),    
               )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supervision_Train(config).to(device)
    if config.pretrained_ckpt_path:
        model.load_state_dict(torch.load(config.pretrained_ckpt_path, map_location=device))

    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler
    train_loader = config.train_loader
    val_loader = config.val_loader

    # 训练/验证循环
    for epoch in range(1, config.max_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config.loss, device,epoch)
        val_loss,mIoU,OA = validate(model, val_loader, config.loss, device)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # 保存最新权重
        os.makedirs(config.weights_path, exist_ok=True)
        latest_ckpt_path = os.path.join(config.weights_path, f"{config.weights_name}_latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        # 保存最佳权重
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_ckpt_path = os.path.join(config.weights_path, f"{config.weights_name}_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved at {best_ckpt_path} with mIoU={best_mIoU:.4f}")

        print(f"Epoch {epoch} done. train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mIoU={mIoU:.4f}, val_OA={OA:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()