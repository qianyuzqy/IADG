import torch
import torch.nn.functional as F
import torch.distributions as tdist
from torch import nn
import torchvision.models as models
from pdb import set_trace as st
import numpy as np
import math
from utils.cov_settings import *
import sys
sys.path.append("..") 

try:
    from . import base_block
    # from .base_block import Conv_block
    from .base_block import Conv_block_gate
    from .base_block import Basic_block_gate
except Exception:
    from utils.initial import init_weights
from utils.initial import init_weights


class CSA(nn.Module):
    def __init__(self, style_dim, base_style_num, concentration_coeff, eps=1e-5):
        '''
            Args:
                style_dim (int): The dimension of the style vector.
                base_style_num (int): The number of base styles.
                concentration_coeff (float): The concentration coefficient for the Dirichlet distribution.
                eps (float, optional): A small epsilon value used for numerical stability. Defaults to 1e-5.
        '''
        super().__init__()
        self.eps = eps
        self.style_dim = style_dim
        self.base_style_num = base_style_num
        self.concentration_coeff = concentration_coeff
        self.concentration = torch.tensor([self.concentration_coeff] * self.base_style_num, device='cuda')
        self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)
        # Register buffer for proto_mean and proto_std
        self.register_buffer("proto_mean_pos", torch.zeros((self.base_style_num, self.style_dim), requires_grad=False))
        self.register_buffer("proto_std_pos", torch.zeros((self.base_style_num, self.style_dim), requires_grad=False))
        self.register_buffer("proto_mean_neg", torch.zeros((self.base_style_num, self.style_dim), requires_grad=False))
        self.register_buffer("proto_std_neg", torch.zeros((self.base_style_num, self.style_dim), requires_grad=False))

    def forward(self, x, label):
        B,C,H,W = x.size()
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x_var = torch.var(x, dim=[2, 3], keepdim=True)
        x_mean, x_var = x_mean.detach(), x_var.detach()
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)

        combine_weights = self._dirichlet.sample((B,)) # B,C
        
        if label[0]==0:  # real        
            new_mean = combine_weights[0].unsqueeze(dim=0) @ self.proto_mean_pos.data # 1,C = (1,N) @ (N,C)
            new_std = combine_weights[0].unsqueeze(dim=0) @ self.proto_std_pos.data   
        else:            # attack
            new_mean = combine_weights[0].unsqueeze(dim=0) @ self.proto_mean_neg.data # 1,C = (1,N) @ (N,C)
            new_std = combine_weights[0].unsqueeze(dim=0) @ self.proto_std_neg.data

        for i in range(1, x_norm.shape[0]):
            if label[i]==0:
                cur_mean = combine_weights[i].unsqueeze(dim=0) @ self.proto_mean_pos.data
                cur_std = combine_weights[i].unsqueeze(dim=0) @ self.proto_std_pos.data
            else:
                cur_mean = combine_weights[i].unsqueeze(dim=0) @ self.proto_mean_neg.data
                cur_std = combine_weights[i].unsqueeze(dim=0) @ self.proto_std_neg.data

            new_mean = torch.cat((new_mean, cur_mean), dim=0)
            new_std = torch.cat((new_std, cur_std), dim=0)

        x_new = x_norm * new_std.unsqueeze(2).unsqueeze(3) + new_mean.unsqueeze(2).unsqueeze(3)

        return x_new

class CSABlock(nn.Module):

    def __init__(self, style_dim, base_style_num, concentration_coeff, dim, model_initial):
        '''
            Args:
                style_dim (int): The dimension of the style vector.
                base_style_num (int): The number of base styles.
                concentration_coeff (float): The concentration coefficient for the Dirichlet distribution.
                dim (int): the channel numbers of features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(CSABlock, self).__init__()
        self.style_dim = style_dim
        self.base_style_num = base_style_num
        self.concentration_coeff = concentration_coeff

        self.CSA_norm = CSA(style_dim=self.style_dim, base_style_num=self.base_style_num,
                                      concentration_coeff=self.concentration_coeff)

        self.model_initial = model_initial

    def forward(self, x, label):
        out = self.CSA_norm(x, label)
        return x + out


class FeatExtractor(nn.Module):
    def __init__(self, dkg_flag, in_channels=6, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                in_channels (int): the channel numbers of input features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(FeatExtractor, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        self.inc = Conv_block_gate(in_channels, 64, 0, False, self.model_initial)
        self.down1 = Basic_block_gate(64, 128, 1, self.dkg_flag, self.model_initial)
        self.down2 = Basic_block_gate(128, 128, 1, self.dkg_flag, self.model_initial)
        self.down3 = Basic_block_gate(128, 128, 1, self.dkg_flag, self.model_initial)

    def cal_cat_feat(self, x1):
        x1 = self.inc(x1)
        x1_1 = self.down1(x1)
        x1_2 = self.down2(x1_1)
        x1_3 = self.down3(x1_2)

        re_x1_1 = F.adaptive_avg_pool2d(x1_1, 32)
        re_x1_2 = F.adaptive_avg_pool2d(x1_2, 32)
        catfeat = torch.cat([re_x1_1, re_x1_2, x1_3],1)

        return catfeat

    def forward(self, input):

        x1 = self.cal_cat_feat(input)
        fea_x1_x1 = x1
            
        # get outputs
        outputs = {}
        outputs["cat_feat"] = x1
        outputs["out"] = fea_x1_x1

        return outputs

class Shader(nn.Module):
    def __init__(self, dkg_flag, style_dim, base_style_num, concentration_coeff, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                style_dim (int): The dimension of the style vector.
                base_style_num (int): The number of base styles.
                concentration_coeff (float): The concentration coefficient for the Dirichlet distribution.
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Shader, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        # #----------------------------------------------------------------------------
        # CSA Module
        self.CSA_layers = nn.ModuleList([CSABlock(style_dim, base_style_num, concentration_coeff, 384, model_initial) for i in range(2)])
        self.style_dim = style_dim
        self.base_style_num = base_style_num
        self.concentration_coeff = concentration_coeff
        # used
        self.conv_final = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384)
        )
        #------------------------------------------------
        # Covariance Matrix Layer for AIAW Loss
        self.eps = 1e-7
        self.cov_matrix_layer_real = CovMatrix_AIAW_real(dim=384, relax_denom=0)
        self.cov_matrix_layer_fake = CovMatrix_AIAW_fake(dim=384, relax_denom=0)

        # model initial
        init_weights(self.conv_final, init_type=self.model_initial)


    def forward(self, input, label, apply_shade, cal_covstat=False, apply_wt=True):
        # Update Mask for Instance Whitening (AIAW)
        if cal_covstat:
            fea_x1_x1 = input
            fea_org = self.conv_final(input)                   # B, 384, H, W
            for i in range(len(self.CSA_layers)):
                fea_x1_x1 = self.CSA_layers[i](fea_x1_x1, label) # B, 384, H, W
            fea_aug = self.conv_final(fea_x1_x1)                 # B, 384, H, W
            #---------------------------------------------------------------------------------------   
            bs = input.shape[0]
          
            fea_aug_real, fea_aug_fake = fea_aug[:int(bs/2), :, :, :], fea_aug[int(bs/2):, :, :, :]
            fea_org_real, fea_org_fake = fea_org[:int(bs/2), :, :, :], fea_org[int(bs/2):, :, :, :]

            # Update Mask for Instance Whitening for real samples
            fea_org_aug_real = torch.cat((fea_org_real, fea_aug_real), dim=0) # 2B, 384, H, W
            B, C, H, W = fea_org_aug_real.shape  # i-th feature size (B X C X H X W)
            HW = H * W
            fea_org_aug_real = fea_org_aug_real.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
            eye_real, reverse_eye_real = self.cov_matrix_layer_real.get_eye_matrix()
            f_corvariance_real = torch.bmm(fea_org_aug_real, fea_org_aug_real.transpose(1, 2)).div(HW - 1) + (self.eps * eye_real)  # B X C X C / HW
            off_diag_elements_real = f_corvariance_real * reverse_eye_real
            self.cov_matrix_layer_real.set_pair_covariance(torch.var(off_diag_elements_real, dim=0))
            
            # Update Mask for Instance Whitening for fake samples
            fea_org_aug_fake = torch.cat((fea_org_fake, fea_aug_fake), dim=0) # 2B, 384, H, W
            B, C, H, W = fea_org_aug_fake.shape  # i-th feature size (B X C X H X W)
            HW = H * W
            fea_org_aug_fake = fea_org_aug_fake.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
            eye_fake, reverse_eye_fake = self.cov_matrix_layer_fake.get_eye_matrix()
            f_corvariance_fake = torch.bmm(fea_org_aug_fake, fea_org_aug_fake.transpose(1, 2)).div(HW - 1) + (self.eps * eye_fake)  # B X C X C / HW
            off_diag_elements_fake = f_corvariance_fake * reverse_eye_fake
            self.cov_matrix_layer_fake.set_pair_covariance(torch.var(off_diag_elements_fake, dim=0))

            return 0         

        if apply_shade==True:
            if self.training:
                fea_x1_x1 = input
                for i in range(len(self.CSA_layers)):
                    fea_x1_x1 = self.CSA_layers[i](fea_x1_x1, label) # B, 384, H, W
                fea_x2_x2 = self.conv_final(fea_x1_x1) # B, 384, H, W
                fea_aug = fea_x2_x2
                fea_org = self.conv_final(input)
                
                bs = input.shape[0]
                fea_aug_real, fea_aug_fake = fea_aug[:int(bs/2), :, :, :], fea_aug[int(bs/2):, :, :, :]
                fea_org_real, fea_org_fake = fea_org[:int(bs/2), :, :, :], fea_org[int(bs/2):, :, :, :]

                # # Turn on the AIAW and get the generated mask
                if apply_wt==True:
                    eye_real, mask_matrix_real, margin_real, num_remove_cov_real = self.cov_matrix_layer_real.get_mask_matrix()
                    eye_fake, mask_matrix_fake, margin_fake, num_remove_cov_fake = self.cov_matrix_layer_fake.get_mask_matrix()
            else:
                fea_x2_x2 = input

        else:
            fea_x2_x2 = input
            
        # get outputs
        outputs = {}
        outputs["out_feat"] = fea_x2_x2
        if self.training and (apply_wt==True):
            outputs["org_feat_real"] = fea_org_real
            outputs["org_feat_fake"] = fea_org_fake
            outputs["aug_feat_real"] = fea_aug_real
            outputs["aug_feat_fake"] = fea_aug_fake
            outputs["eye_real"] = eye_real
            outputs["eye_fake"] = eye_fake
            outputs["mask_matrix_real"] = mask_matrix_real
            outputs["mask_matrix_fake"] = mask_matrix_fake
            outputs["margin_real"] = margin_real
            outputs["margin_fake"] = margin_fake
            outputs["num_remove_cov_real"] = num_remove_cov_real
            outputs["num_remove_cov_fake"] = num_remove_cov_fake
        return outputs


class FeatEmbedder(nn.Module):
    def __init__(self, dkg_flag, in_channels=384, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                in_channels (int): the channel numbers of input features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(FeatEmbedder, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        if self.dkg_flag==True:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 256, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(256, 512, False, self.model_initial)   
        else:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 256, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(256, 512, False, self.model_initial)        	
        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):

        # Normal brach for Feature Classifier
        x = self.conv_block1(x)
        x = self.max_pool(x)
        x= self.conv_block2(x)
        x = self.max_pool(x)
        x = self.conv_block3(x) # torch.Size([0, 13])
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
            
        x = self.fc(x) # torch.Size([12, 2])
            
        # get outputs
        outputs = {}
        outputs["out"] = x

        return outputs

class DepthEstmator(nn.Module):
    def __init__(self, dkg_flag, in_channels=384, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                in_channels (int): the channel numbers of input features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(DepthEstmator, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        if self.dkg_flag==True:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 64, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(64, 1, False, self.model_initial) 
        else:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 64, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(64, 1, False, self.model_initial)        	

    def forward(self, x):

        # Normal brach for Depth Estimator    
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)    
            
        # get outputs
        outputs = {}
        outputs["out"] = x
            
        return outputs            

class Framework(nn.Module):
    def __init__(self, total_dkg_flag, style_dim, base_style_num, concentration_coeff, in_channels=6,mid_channels=384,model_initial='kaiming'):
        '''
            Args:
                total_dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                style_dim (int): The dimension of the style vector.
                base_style_num (int): The number of base styles.
                concentration_coeff (float): The concentration coefficient for the Dirichlet distribution.
                in_channels (int): the channel numbers of input features
                mid_channels (int): the channel numbers of middle features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Framework, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.model_initial = model_initial
        self.total_dkg_flag = total_dkg_flag
        self.style_dim = style_dim
        self.base_style_num = base_style_num
        self.concentration_coeff = concentration_coeff

        self.FeatExtractor = FeatExtractor(dkg_flag=self.total_dkg_flag, 
                                           in_channels=self.in_channels, model_initial=self.model_initial)
        self.Classifier = FeatEmbedder(dkg_flag=False, in_channels=self.mid_channels)
        self.DepthEstmator = DepthEstmator(dkg_flag=False, in_channels=self.mid_channels)
        self.Shader = Shader(dkg_flag=self.total_dkg_flag, style_dim=self.style_dim,
                                           base_style_num = self.base_style_num,
                                           concentration_coeff=self.concentration_coeff,
                                           model_initial=self.model_initial)
    def forward(self, x, label, apply_shade, cal_covstat, apply_wt):
        if cal_covstat==True:
            outputs_catfeat = self.FeatExtractor(x)
            self.Shader(outputs_catfeat["out"], label, apply_shade, cal_covstat, apply_wt)
            outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat= {}, {}, {}, {}, {}, {}
            return outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat

        if self.training:
            outputs_catfeat = self.FeatExtractor(x)
            outputs_catcls = self.Classifier(outputs_catfeat["out"])
            outputs_catdepth = self.DepthEstmator(outputs_catfeat["out"])
            if apply_shade==True:
                outputs_shadefeat = self.Shader(outputs_catfeat["out"], label, apply_shade, cal_covstat, apply_wt)
                outputs_shadecls = self.Classifier(outputs_shadefeat["out_feat"])
                outputs_shadedepth = self.DepthEstmator(outputs_shadefeat["out_feat"])
            else:
                outputs_shadefeat = {}
                outputs_shadecls = {}
                outputs_shadedepth = {}    
        else:
            outputs_catfeat = self.FeatExtractor(x)
            outputs_catcls = self.Classifier(outputs_catfeat["out"])
            outputs_catdepth = self.DepthEstmator(outputs_catfeat["out"])
            outputs_shadefeat = {}
            outputs_shadecls = {}
            outputs_shadedepth = {}

        return outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat

if __name__ == '__main__':
    in_channels=6
    model_initial='kaiming'

    model = Framework()
    x = torch.randn((2,6,256,256))
    y,depth = model(x)
    st()