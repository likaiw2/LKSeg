import numpy as np
import torch
import torch.nn as nn
import utils.tools as tools
import torch.nn.functional as F


# LOSS Function Class
class WeightedMSELoss(nn.Module):
    # $\mathcal{L}_{\mathrm{rec}}^G=\frac{1}{n} \sum_{j=1}^n\left\|\mathbf{M}_j^C \odot\left(G\left(\mathbf{V}_{M, j}^C\right)-\mathbf{V}_j^C\right)\right\|_2$
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, ground_truth, net_output,mask):
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        iter_norm = []
        
        for i in range(batch_size):
            V_ground_truth = ground_truth[i][0]
            V_net_output = net_output[i][0]
            V_mask = mask[i][0]
            diff = V_net_output - V_ground_truth
            valuable_part = V_mask * diff
            # valuable_part_norm = np.linalg.norm(valuable_part,ord=2)
            valuable_part_norm = torch.linalg.norm(valuable_part,dim=1,ord=2).cpu().detach().numpy()
            iter_norm.append(valuable_part_norm)
            
        iter_norm = np.array(iter_norm)
        
        loss_WeightedMSE = torch.mean(torch.tensor(iter_norm,requires_grad=True).cuda())

        return loss_WeightedMSE
    
class AdversarialGLoss(nn.Module):
    # $\mathcal{L}_{\mathrm{adv}}^G=\frac{1}{n} \sum_{j=1}^n\left[\log D\left(\mathbf{M}_j^C \odot G\left(\mathbf{V}_{M, j}^C\right)+\left(\mathbf{1}-\mathbf{M}_j^C\right) \odot \mathbf{V}_j^C\right)\right]$
    def __init__(self, discriminator):
        super(AdversarialGLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, ground_truth, net_output,mask):
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        iter_norm = []
        
        for i in range(batch_size):
            # V_ground_truth = ground_truth[i][0]
            # V_net_output = net_output[i][0]
            # V_mask = mask[i]
            
            mixed_data = mask * net_output + (1 - mask) * ground_truth
            dis_output = self.discriminator(mixed_data)
            
            log_dis = torch.log10(dis_output).cpu().detach().numpy()
            
            iter_norm.append(log_dis)
            
        iter_norm = np.array(iter_norm)
        
        adversarial_loss = torch.mean(torch.tensor(iter_norm,requires_grad=True).cuda())

        return adversarial_loss

class AdversarialDLoss(nn.Module):
    def __init__(self, discriminator):
        super(AdversarialDLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, ground_truth, net_output,mask):
        v_mask=mask
        batch_size = ground_truth.shape[0]
        # print(mask.shape)
        exp1 = []
        exp2 = []
        
        for i in range(batch_size):

            # print("net_output:",net_output.shape)
            log_dis1 = torch.log10(self.discriminator(ground_truth)).cpu().detach().numpy()
            exp1.append(log_dis1)
            
            mixed_data = mask * net_output + (1 - mask) * ground_truth
            # print("mixed_data:",mixed_data.shape)
            dis_output = self.discriminator(mixed_data)
            
            log_dis2 = torch.log10(1 - dis_output).cpu().detach().numpy()
            exp2.append(log_dis2)
            
        exp1 = np.array(exp1)
        exp2 = np.array(exp2)
        
        adversarial_loss1 = torch.mean(torch.tensor(exp1,requires_grad=True).cuda())
        adversarial_loss2 = torch.mean(torch.tensor(exp2,requires_grad=True).cuda())
        adversarial_loss = adversarial_loss1 + adversarial_loss2

        return adversarial_loss


class InpaintingLoss3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        # self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        # if output.shape[1] == 3:
        #     feat_output_comp = self.extractor(output_comp)
        #     feat_output = self.extractor(output)
        #     feat_gt = self.extractor(gt)
        # elif output.shape[1] == 1:
        #     feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
        #     feat_output = self.extractor(torch.cat([output]*3, 1))
        #     feat_gt = self.extractor(torch.cat([gt]*3, 1))
        # else:
        #     raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        # for i in range(3):
        #     loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
        #     loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        # for i in range(3):
        #     loss_dict['style'] += self.l1(self.gram_matrix(feat_output[i]),
        #                                   self.gram_matrix(feat_gt[i]))
        #     loss_dict['style'] += self.l1(self.gram_matrix(feat_output_comp[i]),
        #                                   self.gram_matrix(feat_gt[i]))

        loss_dict['tv'] = self.total_variation_loss(output_comp)

        return loss_dict
    
    def gram_matrix(self,feat):
        # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram
    
    def total_variation_loss(self,image):
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss


class L1Loss(nn.Module):
    # 2D 3D 都可用
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))

class MSELoss(nn.Module):
    # 2D 3D 都可用
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, prediction, target):
        return torch.mean((prediction - target) ** 2)

class TotalVariationLoss3D(nn.Module):
    def __init__(self):
        super(TotalVariationLoss3D, self).__init__()
    
    def forward(self, x):
        h = torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:]))
        w = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        d = torch.mean(torch.abs(x[:, :, :-1, :-1] - x[:, :, 1:, 1:]))
        return h + w + d

class AdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super(AdversarialLoss, self).__init__()
        self.discriminator = discriminator
    
    def forward(self, prediction, target, is_real):
        outputs = self.discriminator(prediction)
        real_outputs = self.discriminator(target)
        if is_real:
            return torch.mean((1 - outputs) ** 2) + torch.mean((1 - real_outputs) ** 2)
        else:
            return torch.mean(outputs ** 2) + torch.mean(real_outputs ** 2)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    def gaussian(self,window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self,window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(self.gaussian(window_size, 1.5).t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self,img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class WMSELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,mask, y_pred, y_true):
        return torch.mean(((1-mask) * (y_true - y_pred)) ** 2)

class TVLoss(torch.nn.Module):
    """
    TV loss
    """

    def __init__(self, weight=1):
        self.weight = weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class CXReconLoss(torch.nn.Module):

    """
    contexutal loss with vgg network
    """

    def __init__(self, feat_extractor, device=None, weight=1):
        super(CXReconLoss, self).__init__()
        self.feat_extractor = feat_extractor
        self.device = device
        if device is not None:
            self.feat_extractor = self.feat_extractor.to(device)
        #self.feat_extractor = self.feat_extractor.cuda()
        self.weight = weight

    def forward(self, imgs, recon_imgs, coarse_imgs=None):
        if self.device is not None:
            imgs = imgs.to(self.device)
            recon_imgs = recon_imgs.to(self.device)
            if coarse_imgs is not None:
                coarse_imgs = coarse_imgs.to(self.device)

        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))

        ori_feats, _ = self.feat_extractor(imgs)
        recon_feats, _ = self.feat_extractor(recon_imgs)
        if coarse_imgs is not None:
            coarse_imgs = F.interpolate(coarse_imgs, (224,224))
            coarse_feats, _ =self.feat_extractor(coarse_imgs)
            return self.weight * (symetric_CX_loss(ori_feats, recon_feats) )
        return self.weight * symetric_CX_loss(ori_feats, recon_feats)


class MaskDisLoss(torch.nn.Module):
    """
    The loss for mask discriminator
    """
    def __init__(self, weight=1):
        super(MaskDisLoss, self).__init__()
        self.weight = weight
        self.leakyrelu = torch.nn.LeakyReLU()
    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(self.leakyrelu(1.-pos)) + torch.mean(self.leakyrelu(1.+neg)))


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)

class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """
    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            #print(masks.view(masks.size(0), -1).mean(1).size(), imgs.size())
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))

class PerceptualLoss(torch.nn.Module):
    """
    Use vgg or inception for perceptual loss, compute the feature distance, (todo)
    """
    def __init__(self, weight=1, layers=[0,9,13,17], feat_extractors=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(feat - recon_feat))
        return self.weight*loss

class StyleLoss(torch.nn.Module):
    """
    Use vgg or inception for style loss, compute the feature distance, (todo)
    """
    def __init__(self, weight=1, layers=[0,9,13,17], feat_extractors=None):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers
    def gram(self, x):
        gram_x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        return torch.bmm(gram_x, torch.transpose(gram_x, 1, 2))

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(self.gram(feat) - self.gram(recon_feat))) / (feat.size(2) * feat.size(3) )
        return self.weight*loss

class ReconLoss(torch.nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """
    def __init__(self, chole_alpha, cunhole_alpha, rhole_alpha, runhole_alpha):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        return self.rhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
                self.runhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
                self.chole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
                self.cunhole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))


