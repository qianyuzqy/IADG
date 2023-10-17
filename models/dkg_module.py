import torch
import torch.nn as nn
import torch.nn.functional as F


class DKGModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        """
            k (int): The size of the kernel.
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            m (int, optional): The channel reduction rate. Defaults to 4.
            padding (int, optional): The padding value for convolution. Defaults to None.
            stride (int, optional): The stride value for convolution. Defaults to 1.
        """
        super(DKGModule, self).__init__()
        self.k = k
        self.channel = inplanes 
        self.group = self.channel // 2
        # cov1
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.pad = padding
        self.stride = stride
        # conv2'
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        # conv2
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)

        # conv4
        self.conv_static = nn.Conv2d(self.channel // 2, self.channel // 2, kernel_size=3, dilation=1, padding=1, bias=True)
        self.fuse = nn.Conv2d(self.channel, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, H, W = x.shape                         # [B * C * H * W]
        # print('x.shape', x.shape)
        x1 = x[:, :int(C/2), :, :]                   # [B * C/2 * H * W]
        x2 = x[:, int(C/2):, :, :]                   # [B * C/2 * H * W]
        
        # Kernel Generator----------------------------------------------
        # conv1 + avg_pool
        g = self.avg_pool(x1)                       # [B * C/2 * 1 * 1]
        g_perm = g.permute(0, 2, 1, 3).contiguous() # [B * 1 * C/2 * 1]
        # conv2
        kernel = self.conv_kernel(g_perm)           # [B * k^2 * C/2 * 1]
        kernel = kernel.permute(0, 3, 2, 1)         # [B * 1 * C/2 * k^2]
       
        f_list = torch.split(x1, 1, 0)              # [1 * C/2 * H * W]
        g_list = torch.split(kernel, 1, 0)          # [1 * 1 * C/2 * k^2]
        # Instance-wise Interaction-------------------------------------
        out = []
        for i in range(N):
            f_one = f_list[i] # [1* C/2 * H * W]
            g_one = g_list[i] # [1 * 1 * C/2 * k^2]
            # Dynamic Kenerl with conv2'
            g_k = self.conv_k(g_one)                                    # [1 * 1 * C/2 * k^2]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k) # [C/2 * 1 * k * k]

            # Padding
            if self.pad is None:
                padding = ((self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2)
            else:
                padding = (self.pad, self.pad, self.pad, self.pad)

            f_one = F.pad(input=f_one, pad=padding, mode='constant', value=0) # [1* C/2 * H * W]

            # Dynamic Kernel Interaction
            o = F.conv2d(input=f_one, weight=g_k, stride=self.stride, groups=self.group)
            out.append(o)

        # Output of Keneral Generator branch
        y_res = torch.cat(out, dim=0)  # [B * C/2 * H * W]
        
        y_out = self.conv_static(x2)   # [B * C/2 * H * W]

        y_out = torch.cat([y_res, y_out], dim=1) # [B * C * H * W] 
        y_out = self.fuse(y_out)                 # [B * C' * H * W]

        return y_out