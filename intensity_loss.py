import torch
import torch.nn.functional as F
import numpy as np
#from skimage.feature import local_binary_pattern
from math import exp
import math
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VIF_CON_Loss(torch.nn.Module):
    """docstring for VIF_SSIM_Loss"""

    def __init__(self, sensors, kernal_size=10, num_channels=1, C=9e-4, device='cuda'):
        super(VIF_CON_Loss, self).__init__()
        self.sensors = sensors
        self.kernal_size = kernal_size
        self.num_channels = num_channels
        self.device = device
        self.c = C
        self.gra_kernal = torch.Tensor([[[[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]]]]).to(device)
        self.avg_kernal = torch.ones(num_channels, 1, self.kernal_size, self.kernal_size) / (self.kernal_size) ** 2
        self.avg_kernal = self.avg_kernal.to(device)

    def forward(self, input_images, output_images):
        vis_images, inf_images, fusion_images = input_images[self.sensors[0]], input_images[self.sensors[1]], \
                                                output_images       # input_images传进来的是一个字典，用key来索引
        idx, num_channels, size = output_images.shape[0], output_images.shape[1], output_images.shape[2]
        one_size = torch.ones(idx, num_channels, size // self.kernal_size, size // self.kernal_size).to(self.device)
        # vis_images_mean = F.conv2d(vis_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        # inf_images_mean = F.conv2d(inf_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        # fusion_images_mean = F.conv2d(fusion_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)

        score_pix = torch.abs(vis_images - fusion_images) + 10*(vis_images <= inf_images) * \
                    torch.pow((inf_images - fusion_images), 2)

        return score_pix.mean()

#调用时先实例化：intency = VIF_CON_Loss(['Vis', 'Inf'], C=9e-4, device='cpu')
#使用：g_loss_intency = intency({'Vis': batch_images_vi, 'Inf': batch_images_ir}, fusion_img)

