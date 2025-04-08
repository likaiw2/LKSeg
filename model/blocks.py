import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils.tools as tools



class VoxelShuffle(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_factor):
        super(VoxelShuffle,self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(in_channels,out_channels*(upscale_factor**3),kernel_size=3,stride=1,padding=1)
    
    def forward(self,x):
        x = self.voxel_shuffle(self.conv(x),self.upscale_factor)
        return x
    
    def voxel_shuffle(input, upscale_factor):
        batch_size, channels, in_height, in_width, in_depth = input.size()
        channels = channels // upscale_factor ** 3

        out_height = in_height * upscale_factor
        out_width = in_width * upscale_factor
        out_depth = in_depth * upscale_factor

        input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

        return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class Dilated_Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(Dilated_Block, self).__init__()
        
        # in the VCNet, in_channels and out_channels are the same
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=2,padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=4,padding=4)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=8,padding=8)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.activation = nn.LeakyReLU()
        
        # nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv1.bias,0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv2.bias,0)
        # nn.init.xavier_uniform_(self.conv3.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv3.bias,0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        # out += identity   #这个会报错，很呆，会产生inplace问题
        out = identity + out
        out = self.activation(out)

        return out

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        
        # this conv is for down sample
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=4, dilation=1, stride=2, padding=1)
        # this conv is for add channels
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=4, dilation=1,  stride=2, padding=1)
        self.activation1 = nn.ELU(inplace=True)
        self.activation2 = nn.ELU(inplace=True)

        # nn.init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv1.bias,0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.conv2.bias,0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv1(2)
        out = self.activation2(out)
        out = self.bn(out)
        return out

class UpSampleBlock_T_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_T_conv, self).__init__()
        
        # transposition for upscale
        self.up_t_conv = nn.ConvTranspose3d(in_channels, out_channels,kernel_size=4,stride=2,padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
        
        # nn.init.xavier_uniform_(self.up_t_conv.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.up_t_conv.bias,0)

        
    def forward(self,x):
        out=self.activation(self.bn(self.up_t_conv(x)))
        return out

class UpSampleBlock_VS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_VS, self).__init__()
    
        # voxel shuffle
        self.up_VS = VoxelShuffle(in_channels, out_channels,  upscale_factor=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU()
            
    def forward(self, x):
        out=self.activation(self.bn(self.up_VS(x)))

        return out

class UpSampleBlock_Trilinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock_Trilinear, self).__init__()
        
        # trilinear + 11conv for upscale
        self.up_conv11 = nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1,  stride=1, padding=0)
        self.up_tri_linear = nn.Upsample(scale_factor=2,mode="trilinear",align_corners=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ELU()
        
        # nn.init.xavier_uniform_(self.up_conv11.weight, gain = np.sqrt(2.0))
        # nn.init.constant_(self.up_conv11.bias,0)
        
    def forward(self,x,res):
        out=x+res
        out=self.up_tri_linear(out)
        out=self.up_conv11(out)
        out=self.activation(out)
        out=self.bn(out)

        return out



#---------------------------------- partial conv --------------------------
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size,stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        # print(input.dtype)
        # print(mask.dtype)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm3d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


#---------------------------------- gated conv ----------------------------
class gated_conv(torch.nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(gated_conv, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv3d = torch.nn.utils.spectral_norm(self.conv3d)
        self.mask_conv3d = torch.nn.utils.spectral_norm(self.mask_conv3d)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)
        #return torch.clamp(mask, -1, 1)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm3d(x)
        else:
            return x

class GatedConv3dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv3dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv3d(input)
        
        mask = self.mask_conv3d(input)
        
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
            
        if self.batch_norm:
            return self.batch_norm3d(x)
        else:
            return x

class GatedDeConv3dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv3dWithActivation, self).__init__()
        self.conv3d = GatedConv3dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv3d(x)
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B , C , D , W , H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,channel,depth,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,depth*width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,depth*width*height)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,depth*width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,channel,depth,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv3d = torch.nn.utils.spectral_norm(self.conv3d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv3d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x
        
        
        
        
#---------------------------------- pix2pix ---------------------------------
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def     __init__(self, input_channels, use_dropout=False, use_bn=True,s1=2,k1=2):
        super(ContractingBlock, self).__init__()
        #self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=k1, stride=s1)
        if use_bn:
            #self.batchnorm = nn.BatchNorm2d(input_channels * 2)
            # self.batchnorm = nn.BatchNorm3d(input_channels * 2)
            self.instancenorm=nn.InstanceNorm3d(input_channels*2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True,s1=2):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=s1, mode='trilinear', align_corners=True)
        # self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        # self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            # self.batchnorm = nn.BatchNorm2d(input_channels // 2)
            self.batchnorm = nn.BatchNorm3d(input_channels // 2)
            self.instancenorm=nn.InstanceNorm3d(input_channels//2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        # print(x.shape)
        #[2048,1,1,1]

        x = self.upsample(x)
        # print("After Expanding:"+str(x.shape))
        # print(x.shape)
        x = self.conv1(x)
        # print("After Conv1:" + str(x.shape))
        # print(x.shape)
        #使用裁剪操作将上一步生成的数据裁剪过后与现有生成数据拼接
        skip_con_x = crop(skip_con_x, x.shape)
        # print("skip_con_x.shape:"+str(skip_con_x.shape))
        # print("x.shape:" + str(x.shape))
        x = torch.cat([x, skip_con_x], axis=1)
        # print("After concact:"+x.shape)
        x = self.conv2(x)
        # print("X AFTER CONV2:"+str(x.shape))
        if self.use_bn and x.shape[3] >1 :
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn :
            x = self.instancenorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net -
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x



