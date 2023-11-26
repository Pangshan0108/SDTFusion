# from torchvision.models.vgg import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim_loss import ssim
from intensity_loss import VIF_CON_Loss

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    
class L_Grad(nn.Module):                # texture loss
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
        
class L_SSIM(nn.Module):                # ssim loss
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))   # 谁的梯度大谁的权重大
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM


# class L_Intensity(nn.Module):           # intensity loss
#     def __init__(self):
#         super(L_Intensity, self).__init__()
#
#     def forward(self, image_A, image_B, image_fused):
#         intensity_joint = torch.max(image_A, image_B)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = VIF_CON_Loss(['Vis', 'Inf'])
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_inten = 100 * self.L_Inten({'Vis': image_B, 'Inf': image_A}, image_fused)
        loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_inten + loss_gradient + loss_SSIM

        return fusion_loss, loss_gradient, loss_inten, loss_SSIM

