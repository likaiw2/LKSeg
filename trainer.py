import os
import wandb
import torch
from tqdm import tqdm
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt

from Codes.Volume_Impainting.LKNet.model.models import InpaintSADirciminator, InpaintSANet, PConvUNet
from Volume_Inpainting.LKNet.model.models import InpaintSADirciminator, InpaintSANet
from model.models import *
from configs.config import get_cfg_defaults  # 导入获取默认配置的函数
import utils.tools as tools
import utils.losses as losses

import time

class _Trainer:
    '''
    初始化参数，步长，数据集信息
    
    The father class of trainer, it will have 2 sons: Unet_trainer and GAN_trainer
    
    _Trainer is aim to make the process of data clear
    '''
    def __init__(self, cfg):
        self.opt = cfg
        self.model_name = f"{self.opt.RUN.MODEL}{self.opt.RUN.ADD_INFO}"
        
        # 初始化wandb，用于实验跟踪和可视化以及wandb
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        cfg.freeze()

        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME,
                        resume=self.opt.TRAIN.RESUME,
                        notes=self.opt.WANDB.LOG_DIR,
                        config=self.opt,
                        mode=self.opt.WANDB.MODE)
        
        # 设置数据集
        self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
                                     volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape=self.opt.DATASET.TARGET_SHAPE,
                                     mask_type=self.opt.RUN.TYPE,
                                     data_type=np.float32)

        self.data_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=self.opt.TRAIN.BATCH_SIZE,
                                           shuffle=self.opt.DATASET.SHUFFLE,
                                           num_workers=self.opt.SYSTEM.NUM_WORKERS)
        
        self.data_iter = iter(self.data_loader)
        
        self.device = torch.device(self.opt.SYSTEM.DEVICE)
        
        self.interval_start = self.opt.TRAIN.INTERVAL_START
        self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
        self.save_path = self.opt.PATH.SAVE_PATH
        self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
        self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
        print("total iter: ", self.interval_total)
    
    def train(self):
        print("ERROR: Please rewrite train function!")
        
    
    
class Unet_Trainer:
    '''
    初始化模型和优化器以权重
    
    It is a class for Unet trainer. 
    
    I modified the input and the steps in each train instance, which make the progress readable
    '''
    def __init__(self, cfg, net=PConvUNet(),loss_function=losses.InpaintingLoss3D()):
        # 初始化名称
        self.opt = cfg
        self.model_name = f"{self.opt.RUN.MODEL}{self.opt.RUN.ADD_INFO}"
        
        # 初始化wandb，用于实验跟踪和可视化以及wandb
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        cfg.freeze()

        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME,
                        resume=self.opt.TRAIN.RESUME,
                        notes=self.opt.WANDB.LOG_DIR,
                        config=self.opt,
                        mode=self.opt.WANDB.MODE)
        
        # 设置数据集
        self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
                                     volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape=self.opt.DATASET.TARGET_SHAPE,
                                     mask_type=self.opt.RUN.TYPE,
                                     data_type=np.float32)

        self.data_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=self.opt.TRAIN.BATCH_SIZE,
                                           shuffle=self.opt.DATASET.SHUFFLE,
                                           num_workers=self.opt.SYSTEM.NUM_WORKERS)
        # 初始化步长和基本信息
        self.data_iter = iter(self.data_loader)
        
        self.device = torch.device(self.opt.SYSTEM.DEVICE)
        
        self.interval_start = self.opt.TRAIN.INTERVAL_START
        self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
        self.save_path = self.opt.PATH.SAVE_PATH
        self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
        self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
        print("total iter: ", self.interval_total)
        
        # 初始化模型和优化器
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.opt.WEIGHT.LEARN_RATE)
        self.loss_function = loss_function.to(self.device)
        
        if self.opt.RUN.LOAD_PTH:
            if os.path.exists(self.opt.PATH.PTH_LOAD_PATH):
                loaded_state = torch.load(self.opt.PATH.PTH_LOAD_PATH)
                self.model.load_state_dict(loaded_state["model"])
                self.optimizer.load_state_dict(loaded_state["optimizer"])
                print("Weight load success!")
            else:
                print("load weights failed!")
                
    def train(self):
        assert False,("ERROR: Please rewrite train function!")
    def __get_input(self):
        assert False,("ERROR: Please rewrite get_input() function!")
    def __get_output(self):
        assert False,("ERROR: Please rewrite get_output() function!")
    def __get_loss(self):
        assert False,("ERROR: Please rewrite get_loss() function!")
    def __do_back_process(self):
        assert False,("ERROR: Please rewrite do_back_process() function!")
    def save_pth(self):
        assert False,("ERROR: Please rewrite save_pth() function!")
    def save_data(self):
        assert False,("ERROR: Please rewrite save_data() function!")


class GAN_Trainer:
    '''
    It is a class for GAN trainer. 
    
    I modified the input and the steps in each train instance, which make the progress readable
    '''
    def __init__(self, cfg, net_G=InpaintSANet, net_D=InpaintSADirciminator, loss_function=None):
        self.opt = cfg
        self.model_name = f"{self.opt.RUN.MODEL}{self.opt.RUN.ADD_INFO}"
        
        # 初始化wandb，用于实验跟踪和可视化以及wandb
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        cfg.freeze()

        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME,
                        resume=self.opt.TRAIN.RESUME,
                        notes=self.opt.WANDB.LOG_DIR,
                        config=self.opt,
                        mode=self.opt.WANDB.MODE)
        
        # 设置数据集
        self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
                                     volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
                                     target_shape=self.opt.DATASET.TARGET_SHAPE,
                                     mask_type=self.opt.RUN.TYPE,
                                     data_type=np.float32)
        self.data_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=self.opt.TRAIN.BATCH_SIZE,
                                           shuffle=self.opt.DATASET.SHUFFLE,
                                           num_workers=self.opt.SYSTEM.NUM_WORKERS)
        
        self.data_iter = iter(self.data_loader)
        self.device = torch.device(self.opt.SYSTEM.DEVICE)
        self.interval_start = self.opt.TRAIN.INTERVAL_START
        self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
        self.save_path = self.opt.PATH.SAVE_PATH
        self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
        self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
        print("total iter: ", self.interval_total)
        
        # 设置生成器鉴别器以及它们的优化器
        self.net_G = net_G
        self.net_D = net_D
        
        # 创建模型组件
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)
        self.net_G_opt = torch.optim.Adam(net_G.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)
        self.net_D_opt = torch.optim.Adam(net_D.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)
        
        self.net_G_opt = torch.optim.Adam(
            net_G.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)
        self.net_D_opt = torch.optim.Adam(
            net_D.parameters(), lr=self.opt.TRAIN.LEARN_RATE, weight_decay=0.0)

        # 加载模型参数
        if self.opt.RUN.LOAD_PTH:
            if os.path.exists(self.opt.PATH.PTH_LOAD_PATH):
                loaded_state = torch.load(self.opt.PATH.PTH_LOAD_PATH)
                self.model.load_state_dict(loaded_state["net_G"])
                self.model.load_state_dict(loaded_state["net_D"])
                self.optimizer.load_state_dict(loaded_state["net_G_opt"])
                self.optimizer.load_state_dict(loaded_state["net_D_opt"])
                print("Weight load success!")
            else:
                print("load weights failed!")
        
        
        
    def train(self):
        assert False,("ERROR: Please rewrite train function!")
    def __get_input(self):
        assert False,("ERROR: Please rewrite get_input() function!")
    def __get_out_G(self):
        assert False,("ERROR: Please rewrite get_output() function!")
    def __get_out_D(self):
        assert False,("ERROR: Please rewrite get_loss() function!")
    def __get_loss_G(self):
        assert False,("ERROR: Please rewrite do_back_process() function!")
    def __get_loss_G(self):
        assert False,("ERROR: Please rewrite do_back_process() function!")
    def __update_G(self):
        assert False,("ERROR: Please rewrite do_back_process() function!")
    def __update_D(self):
        assert False,("ERROR: Please rewrite do_back_process() function!")
    def save_pth(self):
        assert False,("ERROR: Please rewrite save_pth() function!")
    def save_data(self):
        assert False,("ERROR: Please rewrite save_data() function!")


# ------------------------------以下是具体的类，以上都是父类
class PConvUNet_Trainer(Unet_Trainer):
    def __init__(self, cfg, net=PConvUNet(), loss_function=losses.InpaintingLoss3D()):
        super().__init__(cfg, net, loss_function)
    
    def train(self):
        global_iter = 0
        for epoch_idx in tqdm(range(self.opt.TRAIN.EPOCH_TOTAL),unit="epoch"):
            _loss = []
            for gt, mask in tqdm(self.data_loader,unit="iter",leave=False):
                
                input,mask,gt = self.__get_input(groundtruth=gt,mask=mask)
                
                output_raw,output_mask,output_completed = self.__get_output(input=input,mask=mask)
                
                # 计算并在wandb记录loss
                loss_dict,loss = self.__get_loss(input, mask, output=output_raw, ground_truth=gt)
                _loss.append(loss.item())

                self.__do_back_process(loss)

                if self.opt.RUN.SAVE_PTH:
                    # 第一次迭代保存，最后一次迭代保存，中间的话看指定的保存间隔
                    if (global_iter + 1) % self.opt.TRAIN.ITER_SAVE == 0 or (global_iter + 1) == self.interval_total or global_iter == 0:
                        self.save_pth(global_iter)
                        
                        data_save_list=[gt,mask,input,output_raw,output_mask,output_completed]
                        self.save_data(epoch_idx,global_iter,data_save_list)
                

                global_iter += 1
            average_loss=np.average(np.array(_loss))
            print("\n",average_loss)
    
    def __get_input(self,groundtruth,mask):
        groundtruth = groundtruth.to(self.device)
        mask = mask.to(self.device)
        input = groundtruth*mask                 # 制作输入图像
        
        return input,mask,groundtruth
    
    def __get_output(self,input,mask):
        output_raw, output_mask = self.model(input, mask)                 # 制作输出
        output_completed=output_raw*(1-output_mask)+input*output_mask
        
        return output_raw,output_mask,output_completed
    
    def __get_loss(self,input, mask, output, ground_truth):
        loss_dict = self.loss_function(input, mask, output, ground_truth)    # 求损失
        
        # 加权计算并输出损失
        loss = 0.0
        lambda_dict = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
        for key, coef in lambda_dict.items():
            value = coef * loss_dict[key]
            loss += value
        
        self.wandb.log({"iter_loss": loss.item(),
                        "valid_loss": loss_dict["valid"],
                        "hole_loss": loss_dict["hole"],
                        "prc_loss": loss_dict["prc"],
                        "style_loss": loss_dict["style"]
                        }, commit=False)
                    
        return loss_dict,loss
    
    def save_pth(self,global_iter):
        # save weights
        fileName = f"{self.pth_save_path}/{self.model_name}_{global_iter+1}iter.pth"
        os.makedirs(self.pth_save_path) if not os.path.exists(self.pth_save_path) else None

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(), }, fileName)
    
    def save_data(self,epoch_idx,global_iter,data_save_list):
        for i in range(self.opt.TRAIN.BATCH_SIZE):
            # save images
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_ground_truth",
                        volume=data_save_list[0][i])
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_mask",
                        volume=data_save_list[1][i])
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_input",
                        volume=data_save_list[2][i])
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_output_raw",
                        volume=data_save_list[3][i])
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_output_completed",
                        volume=data_save_list[4][i])
            tools.saveRAW(dataSavePath=f"{self.save_path}/{self.model_name}_{epoch_idx}epoch_{global_iter+1}iter",
                        fileName=f"b{i}_output_mask",
                        volume=data_save_list[5][i])
    
    def __do_back_process(self,loss):
        self.optimizer.zero_grad()          # 重置梯度
        loss.backward()                     # 计算梯度
        self.optimizer.step()               # 根据梯度和优化器的参数更新参数

# SAGAN_Trainer还没修改完
class SAGAN_Trainer(GAN_Trainer):
    def __init__(self, cfg, net_G=InpaintSANet, net_D=InpaintSADirciminator):
        super().__init__(cfg, net_G, net_D)

        # 定义损失函数
        self.recon_loss = losses.ReconLoss(*([1.2, 1.2, 1.2, 1.2]))
        self.G_loss = losses.SNGenLoss(0.005)
        self.D_loss = losses.SNDisLoss()
        
    def train(self):
        # netG = self.net_G
        # netD = self.net_D
        GANLoss = self.gan_loss
        ReconLoss=self.recon_loss
        # DLoss = self.dis_loss
        optG = self.net_G_opt
        optD = self.net_D_opt
        dataloader = self.data_loader
        device = self.device
        
        global_iter = 0
        for epoch_idx in tqdm(range(self.opt.TRAIN.EPOCH_TOTAL),unit="epoch"):
            for gt, mask in tqdm(self.data_loader,unit="iter",leave=False):
                # 更新鉴别器
                ground_truth,mask = self.__get_input(gt,mask)
                out_coarse,out_recon,out_final = self.__get_output_G(ground_truth,mask)
                pred_real, pred_fake = self.__get_out_D(ground_truth,mask,out_final)
                loss_d = self.__get_loss_D(pred_real, pred_fake)
                self.__update_D()
                
                # Optimize Generator
                optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
                pred_neg = netD(neg_imgs)
                # pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
                g_loss = GANLoss(pred_neg)
                r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

                whole_loss = g_loss + r_loss

                # Update the recorder for losses
                losses['g_loss'].update(g_loss.item(), imgs.size(0))
                losses['r_loss'].update(r_loss.item(), imgs.size(0))
                losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
                whole_loss.backward()

                optG.step()
                
                
    def __get_input(self,ground_truth,mask):
        ground_truth = ground_truth.to(self.device)
        masks = masks.to(self.device)
        return ground_truth,masks
        
    def __get_output_G(self,ground_truth, mask):
        # 粗糙的imgs和精修的imgs
        coarse_imgs, recon_imgs = self.netG(imgs, mask)
        # 制作补全后的图像，非挖空区域是原图，挖空区域是补全后的图片
        complete_imgs = recon_imgs * (1-mask) + imgs * mask   # mask is 0 on masked region
        
        return coarse_imgs,recon_imgs,complete_imgs

    def __get_out_D(self,ground_truth,mask,out_final):
        # 制作正面图像和负面图像并把它们合并到一起，送进鉴别器
        real_imgs = torch.cat([ground_truth, mask, torch.full_like(mask, 1.)], dim=1)
        fake_imgs = torch.cat([out_final, mask, torch.full_like(mask, 1.)], dim=1)
        real_fake_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

        # 鉴别器执行操作
        pred_real_fake = self.netD(real_fake_imgs)
        pred_real, pred_fake = torch.chunk(pred_real_fake, 2, dim=0)    # 分别读取鉴别器对正面样本的反应和负面样本的反应
        
        return pred_real, pred_fake
    
    def __get_loss_D(self,pred_pos,pred_neg):
        # 求损失并进行反向传播
        d_loss = self.D_loss(pred_pos, pred_neg)
        return d_loss
    
    def __update_D(self,d_loss):
        self.net_D_opt.zero_grad()
        d_loss.backward()
        self.net_D_opt.step()

    def __get_loss_G(self):
        return super().__get_loss_G()

class DCGAN_Ttrainer(GAN_Trainer):
    def __init__(self, cfg, net_G=DCGAN_ResUNet(in_channel=1, n_classes=1, dp_prob=0.2), net_D=DCGAN_Discriminator, loss_function=None):
        super().__init__(cfg, net_G, net_D, loss_function)
            
            
# class P2P_Trainer:
#     def __init__(self,cfg,net_G, net_D) -> None:
#         self.opt = cfg
#         self.model_name = f"{self.opt.RUN.MODEL}"
#         # info = f" [Step: {self.num_step}/{self.opt.TRAIN.NUM_TOTAL_STEP} ({100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP}%)] "
#         # print(info)

#         # 设置数据集
#         self.dataset = tools.DataSet(data_path=self.opt.PATH.DATA_PATH,
#                                      volume_shape=self.opt.DATASET.ORIGIN_SHAPE,
#                                      target_shape=self.opt.DATASET.TARGET_SHAPE,
#                                      mask_type=self.opt.RUN.TYPE,
#                                      data_type=np.float32)
#         self.data_loader = data.DataLoader(dataset=self.dataset,
#                                            batch_size=self.opt.TRAIN.BATCH_SIZE,
#                                            shuffle=self.opt.DATASET.SHUFFLE,
#                                            num_workers=self.opt.SYSTEM.NUM_WORKERS)
#         self.data_iter = iter(self.data_loader)
        
#         # 初始化变量
#         self.device = torch.device(self.opt.SYSTEM.DEVICE)
#         self.interval_start = self.opt.TRAIN.INTERVAL_START
#         self.interval_save = self.opt.TRAIN.INTERVAL_SAVE
        
#         self.save_path = self.opt.PATH.SAVE_PATH
#         self.pth_save_path = self.opt.PATH.PTH_SAVE_PATH
#         self.interval_total = self.opt.TRAIN.EPOCH_TOTAL*len(self.dataset)
#         print("total iter: ", self.interval_total)
        
#         gen = p2pUNet(input_channels=1, output_channels=1).to(self.device)
#         gen_opt = torch.optim.Adam(gen.parameters(), lr=self.opt.)
#         disc = Discriminator(input_dim + real_dim).to(device)
#         disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
        
#     def run(self):
#         pass



if __name__ == '__main__':
    trainer = UnetTrainer(cfg)
    loader = trainer.data_loader
    imgs, mask = next(iter(loader))
    print(imgs.shape)
    print(mask.shape)
