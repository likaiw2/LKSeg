# import glog as log  # Google的日志库
import trainer
import sys
# from tester import Tester  # 导入测试器类
from configs.config import get_cfg_defaults  # 导入获取默认配置的函数
import model.models as models
import torch

# 自动识别在哪跑，并获取默认配置
cfg = get_cfg_defaults()
def get_cfg():
    plat = sys.platform
    if plat=="darwin":                                  # MacOS
        print("MacOS")
        cfg.merge_from_file("/Users/wanglikai/Codes/Volume_Inpainting/LKNet/configs/macbook.yaml")
        
    elif plat=="linux":                                 # linux server
        print("linux server")
        # cfg.merge_from_file("configs/linuxserver.yaml")
        cfg.merge_from_file("/root/autodl-tmp/Diode/Codes/Volume_Impainting/LKNet/configs/linuxsever_autoDL.yaml")
        
    elif (plat=="win32" or plat=="cygwin"):             # windows
        print("windows")
        cfg.merge_from_file(r"Volume_Inpainting\LKNet\configs\windows.yaml")
        
    else:
        print("can't judge platform automatically,please check yaml path")
        # cfg.merge_from_file(r"Volume_Inpainting\VCNet_modify\configs\windows.yaml")
        cfg.merge_from_file(None)
        
get_cfg()

# clear cuda memory
torch.cuda.empty_cache()
torch.manual_seed(0)

Unet_model = {
    "PconvUnet": models.PConvUNet,
    "PconvUnet_ori": models.PConvUNet2,
}

GAN_model = {
    "SAGAN": {"GEN" : models.InpaintSANet,
              "DIS" : models.InpaintSADirciminator},
    "Pix2Pix":{"GEN" : models.p2pUNet,
              "DIS" : models.p2pDiscriminator},
}


if cfg.RUN.TYPE=="train":
    # 创建训练器对象并开始训练
    # 判断用GAN还是Unet来训练
    if cfg.RUN.MODEL in Unet_model:
        print("Unet!")
        trainer = trainer.UnetTrainer(cfg,
                                      model=Unet_model[cfg.RUN.MODEL]())
        trainer.run()
    elif cfg.RUN.MODEL in GAN_model:
        print("GAN!")
        if cfg.RUN.MODEL=="SAGAN":
            trainer = trainer.SAGAN_Trainer(cfg,
                                            net_G=GAN_model[cfg.RUN.MODEL]["GEN"](), 
                                            net_D=GAN_model[cfg.RUN.MODEL]["DIS"]())
        elif cfg.RUN.MODEL=="Pix2Pix":
            trainer = trainer.P2P_Trainer(cfg,
                                          net_G=GAN_model[cfg.RUN.MODEL]["GEN"](), 
                                          net_D=GAN_model[cfg.RUN.MODEL]["DIS"]())
            
        elif cfg.RUN.MODEL=="DCGAN":
            trainer = trainer.DCGAN_Trainer(cfg,
                                            net_G=GAN_model[cfg.RUN.MODEL]["GEN"](), 
                                            net_D=GAN_model[cfg.RUN.MODEL]["DIS"]())
        trainer.run()
    else:
        assert True,"Check your model name!"
    