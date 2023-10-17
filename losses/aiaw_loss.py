import torch
import torch.nn as nn


class AIAWLoss(nn.Module):
    def __init__(self,reduction=None):
        super(AIAWLoss, self).__init__()
        self.reduction = reduction

    def forward(self, f_map_real, f_map_fake, eye_real, eye_fake, mask_matrix_real, mask_matrix_fake, margin_real, margin_fake, num_remove_cov_real, num_remove_cov_fake):
        '''
         Input:
            f_map_real (Tensor): The feature map for real data, with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
            f_map_fake (Tensor): The feature map for fake data, with the same shape as f_map_real.
            eye_real (Tensor): The identity matrix used for real data.
            eye_fake (Tensor): The identity matrix used for fake data.
            mask_matrix_real (Tensor): The mask matrix for real data, used for selective covariance calculation.
            mask_matrix_fake (Tensor): The mask matrix for fake data, used for selective covariance calculation.
            margin_real (Tensor): Margin for real data.
            margin_fake (Tensor): Margin for fake data.
            num_remove_cov_real (Tensor): The number of covariances to be removed for real data.
            num_remove_cov_fake (Tensor): The number of covariances to be removed for fake data.
         Return:
             loss (Tensor): The AIAW loss
         '''
        f_cov_real, B = get_covariance_matrix(f_map_real, eye_real)
        f_cov_masked_real = f_cov_real * mask_matrix_real

        f_cov_fake, B = get_covariance_matrix(f_map_fake, eye_fake)
        f_cov_masked_fake = f_cov_fake * mask_matrix_fake

        off_diag_sum_real = torch.sum(torch.abs(f_cov_masked_real), dim=(1,2), keepdim=True) - margin_real # B X 1 X 1
        loss_real = torch.clamp(torch.div(off_diag_sum_real, num_remove_cov_real), min=0) # B X 1 X 1
        loss_real = torch.sum(loss_real) / B

        off_diag_sum_fake = torch.sum(torch.abs(f_cov_masked_fake), dim=(1,2), keepdim=True) - margin_fake # B X 1 X 1
        loss_fake = torch.clamp(torch.div(off_diag_sum_fake, num_remove_cov_fake), min=0) # B X 1 X 1
        loss_fake = torch.sum(loss_fake) / B
        
        loss = (loss_real + loss_fake) / 2

        return loss


def get_covariance_matrix(f_map, eye=None):
    '''
     Input:
        f_map (Tensor): The feature map tensor with shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the feature map.
        eye (Tensor, optional): The identity matrix used for covariance calculation. Defaults to None.
     Return:
         f_cor (Tensor): The covariance matrix of the feature map
         B (int): batch size
     '''
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B