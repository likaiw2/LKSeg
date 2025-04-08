import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils.tools as tools
from model.blocks import * 
from torchvision import models


class UNet_v2(nn.Module):
    def __init__(self, down_mode=3, up_mode=3):
        super(UNet_v2, self).__init__()
        # input = ["batch_size",1,128,128,128] 
        
        self.activate_fun = nn.LeakyReLU()
        # Conv + ReLU (down sample)
        self.down_sample_1 = DownSampleBlock(in_channels=1,  out_channels=32)
        self.down_sample_2 = DownSampleBlock(in_channels=32, out_channels=64)
        self.down_sample_3 = DownSampleBlock(in_channels=64, out_channels=128)
        
        # dilated conv + RB 作者表述不清不楚， 邮件询问后，得到三个 dilated RB 一模一样
        self.mid_1 = Dilated_Block(in_channels=256,out_channels=256)
        self.mid_2 = Dilated_Block(in_channels=256,out_channels=256)
        self.mid_3 = Dilated_Block(in_channels=256,out_channels=256)

        # upsample
        if up_mode == 1:
            # use transposition conv
            self.up_sample_4 = UpSampleBlock_T_conv(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_T_conv(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_T_conv(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_T_conv(in_channels=32,out_channels=1)
        
        if up_mode == 2:
            # use Voxel Shuffle +31conv
            self.up_sample_4 = UpSampleBlock_VS(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_VS(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_VS(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_VS(in_channels=32,out_channels=1)
        
        if up_mode == 3:
            # use Trilinear + 11conv
            self.up_sample_4 = UpSampleBlock_Trilinear(in_channels=256,out_channels=128)
            self.up_sample_3 = UpSampleBlock_Trilinear(in_channels=128,out_channels=64)
            self.up_sample_2 = UpSampleBlock_Trilinear(in_channels=64,out_channels=32)
            self.up_sample_1 = UpSampleBlock_Trilinear(in_channels=32,out_channels=1)

        self.final_activate_fun = nn.tanh()
        # self.final_activate_fun = tools.Swish(0.5)
        
    def forward(self, x,mask, test_mode=False, dataSavePath="/home/dell/storage/WANGLIKAI/VCNet/output"):
        res_0 = x
        x = torch.cat([x, mask], dim=1)
        
        # down_sample_1     2,128,128->32,64,64
        out = self.down_sample_1(x)
        # print("down_sample_1:",out.shape)
        res_1 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_64",f"down_64_{i}",out[0, i, :, :, :])
        
        # down_sample_2     32,64,64->64,32,32
        out = self.down_sample_2(out)
        # print("down_sample_2:",out.shape)
        res_2 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_32",f"down_32{i}",out[0, i, :, :, :])
        
        # down_sample_3     64,32,32->128,16,16
        out = self.down_sample_3(out)
        # print("down_sample_3:",out.shape)
        res_3 = out
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#down_16",f"down_16{i}",out[0, i, :, :, :])

        # mid conv + RB 作者表述不清不楚，目前暂定三个 dilated RB 一模一样
        out=self.mid_1(out)
        out=self.mid_2(out)
        out=self.mid_3(out)
        
        # up_sample_3       128,16,16->64,32,32
        out=self.up_sample_3(out,res_3)
        # print("layer3_conv",out.shape)
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#up_32",f"up_32_{i}",out[0, i, :, :, :])
        
        # up_sample_2       64,32,32->32,64,64
        out=self.up_sample_2(out,res_2)
        # print("layer2_conv",out.shape)
        if test_mode:
            for i in range(32):
                Volume_Inpainting.VCNet_modify.utils.tools.saveRawFile10(f"{dataSavePath}/#up_64",f"up_64_{i}",out[0, i, :, :, :])
        
        # up_sample_1       32,64,64->1,128,128
        out=self.up_sample_1(out,res_1)
        # print("up_sample_1:",out.shape)
        
        
        out=self.final_activate_fun(self.up_bn1(self.up_res_conv_1(out)))
        # out=self.final_activate_fun(self.up_bn1(self.up_res_conv_1(out)))
        # print("layer1_conv(final)",out.shape)
        
        return out
    
class Dis_VCNet(nn.Module):
    def __init__(self):
        super(Dis_VCNet,self).__init__()

        self.activate_fun = nn.ReLU(inplace=True)   # 原地修改数据，可以节省空间

        self.start_conv = nn.Conv3d(in_channels=1,   out_channels=32, kernel_size=1)
        self.down_1_conv = nn.Conv3d(in_channels=1,   out_channels=32,  kernel_size=4, dilation=1,  stride=2)
        self.down_2_conv = nn.Conv3d(in_channels=32,  out_channels=64,  kernel_size=4, dilation=1,  stride=2)
        self.down_3_conv = nn.Conv3d(in_channels=64,  out_channels=128,  kernel_size=4, dilation=1,  stride=2)
        self.down_4_conv = nn.Conv3d(in_channels=128,   out_channels=1,  kernel_size=4, dilation=1,  stride=2)
        self.avg = nn.AdaptiveAvgPool3d(output_size=1)
        
        self.pool1 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=4,stride=2)
        self.pool4 = nn.MaxPool3d(kernel_size=4,stride=2)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(1)
        self.activate_fun = nn.LeakyReLU()

    def forward(self,x):
        # out = self.activate_fun(self.start_conv(x))
        # print("x:",x.shape)
        out = self.activate_fun(self.down_1_conv(x))
        # out = self.pool1(out)
        out = self.activate_fun(self.down_2_conv(out))
        # out = self.pool2(out)
        out = self.activate_fun(self.down_3_conv(out))
        # out = self.pool3(out)
        out = self.activate_fun(self.down_4_conv(out))
        # out = self.pool4(out)
        out = self.avg(out)
        out = self.activate_fun(out)
        
        return out
    
# partial conv
class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=1, upsampling_mode='trilinear'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsample_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,bn=False, activ=None, conv_bias=True)
        
        # self.dec_7 = PCBActiv(512, 512, activ='leaky')
        # self.dec_6 = PCBActiv(512, 512, activ='leaky')
        # self.dec_5 = PCBActiv(512, 512, activ='leaky')
        # self.dec_4 = PCBActiv(512, 256, activ='leaky')
        # self.dec_3 = PCBActiv(256, 128, activ='leaky')
        # self.dec_2 = PCBActiv(128, 64, activ='leaky')
        # self.dec_1 = PCBActiv(64, 1,bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        
        # 初始输入和遮罩
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        
        # 编码器
        h_dict['h_1'], h_mask_dict['h_1'] = self.enc_1(h_dict['h_0'], h_mask_dict['h_0'])
        h_dict['h_2'], h_mask_dict['h_2'] = self.enc_2(h_dict['h_1'], h_mask_dict['h_1'])
        h_dict['h_3'], h_mask_dict['h_3'] = self.enc_3(h_dict['h_2'], h_mask_dict['h_2'])
        h_dict['h_4'], h_mask_dict['h_4'] = self.enc_4(h_dict['h_3'], h_mask_dict['h_3'])
        h_dict['h_5'], h_mask_dict['h_5'] = self.enc_5(h_dict['h_4'], h_mask_dict['h_4'])
        h_dict['h_6'], h_mask_dict['h_6'] = self.enc_6(h_dict['h_5'], h_mask_dict['h_5'])
        # h_dict['h_7'], h_mask_dict['h_7'] = self.enc_7(h_dict['h_6'], h_mask_dict['h_6'])
        
        # 保存一下，第七层就是最底层
        # h, h_mask = h_dict['h_7'], h_mask_dict['h_7']
        h, h_mask = h_dict['h_6'], h_mask_dict['h_6']

        # 解码器
        # h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_6')
        # h,h_mask = self.dec_7(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_5')
        h, h_mask = self.dec_6(h, h_mask)
        h=h*(1-h_mask_dict['h_5'])+h_dict['h_5']*h_mask_dict['h_5']
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_4')
        h, h_mask = self.dec_5(h, h_mask)
        h=h*(1-h_mask_dict['h_4'])+h_dict['h_4']*h_mask_dict['h_4']

        

        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_3')
        h, h_mask = self.dec_4(h, h_mask)
        h=h*(1-h_mask_dict['h_3'])+h_dict['h_3']*h_mask_dict['h_3']
        
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_2')
        h, h_mask = self.dec_3(h, h_mask)
        h=h*(1-h_mask_dict['h_2'])+h_dict['h_2']*h_mask_dict['h_2']
       
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_1')
        h, h_mask = self.dec_2(h, h_mask)
        h=h*(1-h_mask_dict['h_1'])+h_dict['h_1']*h_mask_dict['h_1']
        

        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_0')
        h, h_mask = self.dec_1(h, h_mask)
        
        h_final=h*(1-input_mask)+input*input_mask

        return h_final, h_mask+input_mask
    
    def up_sample(self,h,h_mask,h_dict,h_mask_dict,layer_name):
        
        # print("before interpolate:")
        # print(h.shape)
        # print(h_mask.shape)
        h = F.interpolate(h, scale_factor=2, mode=self.upsample_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsample_mode)
        # # h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.
        
        # print("after interpolate:")
        # print(h.shape)
        # print(h_mask.shape)
        
        # print(f"h_dict {layer_name} :")
        # print(h_dict[layer_name].shape)
        # print(h_mask_dict[layer_name].shape)
        
        # h = h + h_dict[layer_name]
        # h_mask = h_mask+ h_mask_dict[layer_name]
        
        h = torch.cat([h, h_dict[layer_name]], dim=1)
        h_mask = torch.cat([h_mask, h_mask_dict[layer_name]], dim=1)
        
        # print("final:")
        # print(h.shape)
        # print(h_mask.shape)
        # print()
        
        return h,h_mask

class PConvUNet2(nn.Module):
    def __init__(self, layer_size=7, input_channels=1, upsampling_mode='trilinear'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsample_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,bn=False, activ=None, conv_bias=True)
        
        # self.dec_7 = PCBActiv(512, 512, activ='leaky')
        # self.dec_6 = PCBActiv(512, 512, activ='leaky')
        # self.dec_5 = PCBActiv(512, 512, activ='leaky')
        # self.dec_4 = PCBActiv(512, 256, activ='leaky')
        # self.dec_3 = PCBActiv(256, 128, activ='leaky')
        # self.dec_2 = PCBActiv(128, 64, activ='leaky')
        # self.dec_1 = PCBActiv(64, 1,bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        
        # 初始输入和遮罩
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        
        # 编码器
        h_dict['h_1'], h_mask_dict['h_1'] = self.enc_1(h_dict['h_0'], h_mask_dict['h_0'])
        h_dict['h_2'], h_mask_dict['h_2'] = self.enc_2(h_dict['h_1'], h_mask_dict['h_1'])
        h_dict['h_3'], h_mask_dict['h_3'] = self.enc_3(h_dict['h_2'], h_mask_dict['h_2'])
        h_dict['h_4'], h_mask_dict['h_4'] = self.enc_4(h_dict['h_3'], h_mask_dict['h_3'])
        h_dict['h_5'], h_mask_dict['h_5'] = self.enc_5(h_dict['h_4'], h_mask_dict['h_4'])
        h_dict['h_6'], h_mask_dict['h_6'] = self.enc_6(h_dict['h_5'], h_mask_dict['h_5'])
        # h_dict['h_7'], h_mask_dict['h_7'] = self.enc_7(h_dict['h_6'], h_mask_dict['h_6'])
        
        # 保存一下，第七层就是最底层
        # h, h_mask = h_dict['h_7'], h_mask_dict['h_7']
        h, h_mask = h_dict['h_6'], h_mask_dict['h_6']

        # 解码器
        # h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_6')
        # h,h_mask = self.dec_7(h, h_mask)
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_5')
        h, h_mask = self.dec_6(h, h_mask)
        # h=h*(1-h_mask_dict['h_5'])+h_dict['h_5']*h_mask_dict['h_5']
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_4')
        h, h_mask = self.dec_5(h, h_mask)
        # h=h*(1-h_mask_dict['h_4'])+h_dict['h_4']*h_mask_dict['h_4']

    
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_3')
        h, h_mask = self.dec_4(h, h_mask)
        # h=h*(1-h_mask_dict['h_3'])+h_dict['h_3']*h_mask_dict['h_3']
        
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_2')
        h, h_mask = self.dec_3(h, h_mask)
        # h=h*(1-h_mask_dict['h_2'])+h_dict['h_2']*h_mask_dict['h_2']
       
        
        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_1')
        h, h_mask = self.dec_2(h, h_mask)
        # h=h*(1-h_mask_dict['h_1'])+h_dict['h_1']*h_mask_dict['h_1']
        

        h,h_mask = self.up_sample(h,h_mask,h_dict,h_mask_dict,'h_0')
        h, h_mask = self.dec_1(h, h_mask)
        # print(h.shape)
        # print(h_mask.shape)
        
        # h_final=h*(1-input_mask)+input*input_mask

        return h, h_mask*input_mask
    
    def up_sample(self,h,h_mask,h_dict,h_mask_dict,layer_name):
        
        # print("before interpolate:")
        # print(h.shape)
        # print(h_mask.shape)
        h = F.interpolate(h, scale_factor=2, mode=self.upsample_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsample_mode)
        # # h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.
        
        # print("after interpolate:")
        # print(h.shape)
        # print(h_mask.shape)
        
        # print(f"h_dict {layer_name} :")
        # print(h_dict[layer_name].shape)
        # print(h_mask_dict[layer_name].shape)
        
        # h = h + h_dict[layer_name]
        # h_mask = h_mask+ h_mask_dict[layer_name]
        
        h = torch.cat([h, h_dict[layer_name]], dim=1)
        h_mask = torch.cat([h_mask, h_mask_dict[layer_name]], dim=1)
        
        # print("final:")
        # print(h.shape)
        # print(h_mask.shape)
        # print()
        
        return h,h_mask

# gated conv
class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=3):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(
            #input is 3*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv3dWithActivation(n_in_channel, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),
            # downsample 128
            GatedConv3dWithActivation(cnum, 2*cnum, 4, 2, padding=tools.get_pad(256, 4, 2)),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            #downsample to 64
            GatedConv3dWithActivation(2*cnum, 4*cnum, 4, 2, padding=tools.get_pad(128, 4, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=tools.get_pad(64, 3, 1, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=tools.get_pad(64, 3, 1, 4)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=tools.get_pad(64, 3, 1, 8)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=tools.get_pad(64, 3, 1, 16)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            # upsample
            GatedDeConv3dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            #Self_Attn(2*cnum, 'relu'),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedDeConv3dWithActivation(2, 2*cnum, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),

            GatedConv3dWithActivation(cnum, cnum//2, 3, 1, padding=tools.get_pad(256, 3, 1)),
            #Self_Attn(cnum//2, 'relu'),
            GatedConv3dWithActivation(cnum//2, 1, 3, 1, padding=tools.get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv3dWithActivation(n_in_channel, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),
            # downsample
            GatedConv3dWithActivation(cnum, cnum, 4, 2, padding=tools.get_pad(256, 4, 2)),
            GatedConv3dWithActivation(cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            # downsample
            GatedConv3dWithActivation(2*cnum, 2*cnum, 4, 2, padding=tools.get_pad(128, 4, 2)),
            GatedConv3dWithActivation(2*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=tools.get_pad(64, 3, 1, 2)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=tools.get_pad(64, 3, 1, 4)),
            #Self_Attn(4*cnum, 'relu'),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=tools.get_pad(64, 3, 1, 8)),
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=tools.get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=True)
        self.refine_upsample_net = nn.Sequential(
            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),

            GatedConv3dWithActivation(4*cnum, 4*cnum, 3, 1, padding=tools.get_pad(64, 3, 1)),
            GatedDeConv3dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedConv3dWithActivation(2*cnum, 2*cnum, 3, 1, padding=tools.get_pad(128, 3, 1)),
            GatedDeConv3dWithActivation(2, 2*cnum, cnum, 3, 1, padding=tools.get_pad(256, 3, 1)),

            GatedConv3dWithActivation(cnum, cnum//2, 3, 1, padding=tools.get_pad(256, 3, 1)),
            #Self_Attn(cnum, 'relu'),
            GatedConv3dWithActivation(cnum//2, 1, 3, 1, padding=tools.get_pad(256, 3, 1), activation=None),
        )


    def forward(self, imgs, masks, img_exs=None):
        # print(imgs.shape)
        # print(masks.shape)
        # Coarse
        # make masked mage 
        
        masked_imgs =  imgs * masks
        if img_exs == None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        # print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        
        x = torch.clamp(x, -1., 1.)     # 将tensor元素限制到(-1,1)区间
        coarse_x = x
        
        # Refine
        masked_imgs = imgs * masks + coarse_x * (1 - masks)
        if img_exs is None:
            input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        # print(masked_imgs.size(), masks.size(), input_imgs.size())
        # print("input_imgs:",input_imgs.shape)
        x = self.refine_conv_net(input_imgs)
        # print("x:",x.shape)
        # x,attention = self.refine_attn(x)
        # print(x.size(), attention.size())
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(3, 2*cnum, 4, 2, padding=tools.get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=tools.get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=tools.get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=tools.get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=tools.get_pad(16, 5, 2)),
            # SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=tools.get_pad(8, 5, 2)),
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=tools.get_pad(8, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x

# pix2pix
class p2pUNet(nn.Module):
    #Generator
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(p2pUNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        # self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True,s1=(2,2,3))
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        # self.contract4 = ContractingBlock(hidden_channels * 8,k1=(2,2,1),s1=(2,2,1))
        self.contract4 = ContractingBlock(hidden_channels * 8,k1=(1,2,1),s1=(1,2,1))
        # self.contract5 = ContractingBlock(hidden_channels * 16,k1=(1,1,1),s1=(1,1,1))
        self.contract5 = ContractingBlock(hidden_channels * 16, k1=(1,1,1),s1=(1,1,1))
        self.contract6 = ContractingBlock(hidden_channels * 32,k1=(1,1,1),s1=(1,1,1))
        self.expand0 = ExpandingBlock(hidden_channels * 64,s1=1)
        self.expand1 = ExpandingBlock(hidden_channels * 32,s1=1)
        # self.expand1 = ExpandingBlock(hidden_channels * 32, s1=(1,2,1))
        # self.expand2 = ExpandingBlock(hidden_channels * 16,s1=(2,2,1))
        self.expand2 = ExpandingBlock(hidden_channels * 16,s1=(1,2,1))
        # self.expand3 = ExpandingBlock(hidden_channels * 8,s1=(2,2,3))
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        # self.deconv=nn.ConvTranspose3d(1,1,kernel_size=(9,9,6),stride=1,padding=0,dilation=(2,6,4))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        # print("X3.shape:"+str(x3.shape))
        x4 = self.contract4(x3)
        # print("X4.shape:" + str(x4.shape))
        x5 = self.contract5(x4)
        # print("X5.shape:" + str(x5.shape))
        x6 = self.contract6(x5)
        # print("X6.shape:" + str(x6.shape))

        # print("X6.shape:"+str(x6.shape))
        # print(x5.shape)
        # x6=x6.squeeze(0)
        x7 = self.expand0(x6, x5)
        # print("X7.shape:" + str(x7.shape))
        x8 = self.expand1(x7, x4)
        # print("X8.shape:" + str(x8.shape))
        x9 = self.expand2(x8, x3)
        # print("X9.shape:" + str(x9.shape))
        x10 = self.expand3(x9, x2)
        # print("X10.shape:" + str(x10.shape))
        x11 = self.expand4(x10, x1)
        # print("X11.shape:" + str(x11.shape))
        x12 = self.expand5(x11, x0)
        # print("X12.shape:" + str(x12.shape))
        # print("X12 shape"+str(x12.shape))
        xn = self.downfeature(x12)
        # print("Xn shape" + str(xn.shape))
        # xn=self.deconv(xn)

        return self.sigmoid(xn)

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
class p2pDiscriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(p2pDiscriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        # self.final = nn.Conv2d(hidden_channels * 16, None, kernel_size=None)
        self.final = nn.Conv3d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        # print(x.shape)
        # print(y.shape)
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

# DCGAN
class DCGAN_ResUNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(DCGAN_ResUNet, self).__init__()
        # self.imsize = imsize

        self.activation = F.leaky_relu

        self.pool1 = nn.MaxPool3d(2)
        # self.pool1 = nn.MaxPool3d((3,3,3),(1,1,1),(2,2,3))
        self.pool2 = nn.MaxPool3d(2)
        # self.pool3 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(3, stride=(2, 2, 2), padding=1)
        # self.pool4 = nn.MaxPool3d(2)

        # hidden_channel = 32
        hidden_channel = 16
        self.conv_block1_64 = UNetConvBlock(in_channel, hidden_channel)
        self.conv_block64_128 = residualUnit(hidden_channel, hidden_channel*2)
        self.conv_block128_256 = residualUnit(hidden_channel*2, hidden_channel*4)
        self.conv_block256_512 = residualUnit(hidden_channel*4, hidden_channel*8)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        # self.up_block512_256 = UNetUpResBlock_223(hidden_channel*8, hidden_channel*4)
        self.up_block512_256 = UNetUpResBlock(hidden_channel*8, hidden_channel*4)
        self.up_block256_128 = UNetUpResBlock(hidden_channel*4, hidden_channel*2)
        self.up_block128_64 = UNetUpResBlock(hidden_channel*2, hidden_channel)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(hidden_channel, n_classes, 1, stride=1)

    # def forward(self, x, res_x):
    def forward(self, x):
        res_x = x
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)             #(hc,80,112,84)
        # print ('block1.shape: ', block1.shape)
        pool1 = self.pool1(block1)                  #(hc,40,56,42)
        # print ('pool1.shape: ', block1.shape)
        pool1_dp = self.Dropout(pool1)              #(hc,40,56,42)
        # print ('pool1_dp.shape: ', pool1_dp.shape)
        block2 = self.conv_block64_128(pool1_dp)    #(hc*2,40,56,42)
        # print ('block2.shape: ', block2.shape)
        pool2 = self.pool2(block2)                  #(hc*2,20,28,21)
        # print ('pool2.shape: ', pool2.shape)
        pool2_dp = self.Dropout(pool2)              #(hc*2,20,28,21)
        # print ('pool2_dp.shape: ', pool2_dp.shape)

        block3 = self.conv_block128_256(pool2_dp)   #(hc*4,20,28,21)
        # print ('block3.shape: ', block3.shape)
        
        pool3 = self.pool3(block3)                  #(hc*4,10,14,7)
        # print ('pool3.shape: ', pool3.shape)

        pool3_dp = self.Dropout(pool3)              #(hc*4,10,14,7)
        # print ('pool3_dp.shape: ', pool3_dp.shape)

        block4 = self.conv_block256_512(pool3_dp)   #(hc*8,10,14,7)
        # print ('block4.shape: ', block4.shape)
        
        # pool4 = self.pool4(block4)
        # pool4_dp = self.Dropout(pool4)
        # # block5 = self.conv_block512_1024(pool4_dp)
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
        # print ('up2.shape: ', up2.shape)

        # up3 = self.up_block256_128(up2, block2)
        up3 = self.up_block256_128(block3, block2)  #如果不用512-256的话启用这个
        # print ('up3.shape: ', up3.shape)

        up4 = self.up_block128_64(up3, block1)
        # print ('up4.shape: ', up4.shape)

        last = self.last(up4)
        
        out = last
        # print ('res_x.shape is ', res_x.shape, ' and last.shape is ', last.shape)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print ('out.shape is ',out.shape)
        return out

class DCGAN_Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(DCGAN_Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        # self.final = nn.Conv2d(hidden_channels * 16, None, kernel_size=None)
        self.final = nn.Conv3d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn





