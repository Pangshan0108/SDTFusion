
import argparse
import time
import os
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from train import *
from model.Trans_model import *

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--phase', default='val', type=str, help='test phase')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument("--model_dir", default='./save_model', help="训练模型状态字典路径")
parser.add_argument("--test_ir_dir", default='./images/source_ir', help="测试路径")
parser.add_argument("--test_vi_dir", default='./images/source_vi', help="测试路径")
opt = parser.parse_args()

# data
def all_path(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.tif', '.img', '.jpg', '.png', '.bmp']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list

def all_path_model(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.pkl']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list

def imsave(image,path):
    img = Image.fromarray(image)
    img.save(path)

# def get_img_parts(image, height, width):
#     images = []
#     h_cen = int(np.floor(height / 40))              # 测试图像高度/40再向下取整
#     w_cen = int(np.floor(width / 40))               # 测试图像宽度/40再向下取整
#     for i in range(h_cen-1):
#         for j in range(w_cen-1):
#             img = image[:, :, i*60-20*i:(i+1)*60-20*i, j*60-20*j:(j+1)*60-20*j]         # 用40的步长切成60*60的图像块
#             images.append(img)
#             j+=1
#         i+=1
#     return images, i

def get_img_parts(image, height, width):
    images = []
    h_cen = int(np.floor(height / 54))      # np.floor向下取整
    w_cen = int(np.floor(width / 54))
    h_pad = 54 - (height - (54 * h_cen))    # 未被整除的剩余部分有多少像素
    w_pad = 54 - (width - (54 * w_cen))
    pad_c0 = torch.nn.ReflectionPad2d(padding=(0, w_pad, 0, h_pad))#left, right, top, down
    image1 = pad_c0(image)          # image1高和宽经过填充可以被54整除
    # print(image.shape)
    # print(image1.shape)
    for i in range(h_cen+1):        # h_cen和w_cen是向下取整得到的,经过填充这两个数字应该+1
        for j in range(w_cen+1):
            img = image1[:, :, i*54:(i+1)*54, j*54:(j+1)*54]        # 1,1,54,54
            # print(image.type())
            # pad = torch.nn.ReflectionPad2d(padding=(10, 10, 10, 10))
            pad = torch.nn.ReflectionPad2d(padding=(5, 5, 5, 5))
            img = pad(img)                      # 1,1,64,64
            images.append(img)                  # 切好的块组成的列表
            j+=1
        i+=1
    return images, h_cen+1, img                 # h_cen+1是沿height方向可以切多少个块


def recons_fusion_images(img_lists, q, height, width):
    # img_f = []
    l = len(img_lists)          # img_lists融合图像小块组成的列表
    t = int(l/q)                # q是h_cen+1即沿height方向的切块数,t是沿width方向的切块数
    # h_cen = int(np.floor(height / 60))
    # w_cen = int(np.floor(width / 60))
    img_f = torch.zeros(1, 1, q * 54, t * 54).cuda()    # 定义一个未切块大小的零tensor
    x = 0
    for i in range(q):
        for j in range(t):
            im = img_lists[x]       # 拿一个64*64的小块
            img_f[:, :, i * 54:(i + 1) * 54, j * 54: (j + 1) * 54] += im[:,:,5:59,5:59]     # 跳过5个填充的像素,从5开始取值到59
            x = x + 1
    img_f1 = torch.zeros(1, 1, height, width).cuda()    # 定义一个和原图大小相同的零tensor
    img_f1 += img_f[:, :, 0:height, 0:width]
    return img_f1

TEST_MODE = True if opt.test_mode == 'GPU' else False

model_list = all_path_model(opt.model_dir)
model_num = len(model_list)
model = TransFusion().cuda().eval()

para = sum([np.prod(list(p.size())) for p in model.parameters()])  # para为参数量，参数为浮点型，即一个参数占4个字节
type_size = 4
print('Model {} : params: {:4f}M'.format('TransFusion', para * type_size / 1000 / 1000))   # ×4/1024/1024转化为Mb输出

with torch.no_grad():
    for i in range(model_num):
        model_path = model_list[i]
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))) # 载入第一个模型的状态字典
        ir_list = all_path(opt.test_ir_dir)
        # ir_list.sort()
        ir_list.sort(key=lambda x: int(x.split('/')[3][:-4]))
        vi_list = all_path(opt.test_vi_dir)
        # vi_list.sort()
        vi_list.sort(key=lambda x: int(x.split('/')[3][:-4]))
        files_list = list(zip(ir_list, vi_list))
        files_num = len(files_list)

        for ii in range(0,files_num):
            ir, vi = files_list[ii]

            ir = Image.open(ir).convert('L')  # 读取图片，变化类型为numpy浮点型
            ir = np.array(ir).astype(np.float64)
            vi = Image.open(vi).convert('L')  # 读取图片，变化类型为numpy浮点型
            vi = np.array(vi).astype(np.float64)
            # print (ir.shape)                    # 数组(576, 768)

            ir = (ir - 127.5) / 127.5           # 归一化至-1~1
            vi = (vi - 127.5) / 127.5
            # print(vi.shape)                     # 数组(576, 768)


            label = ir
            image = vi
            # concat_image = concat_img
            label = torch.cuda.FloatTensor(label)       # numpy数组转换为(H,W)二维浮点型gpu张量
            label = torch.unsqueeze(label, 0)
            label = torch.unsqueeze(label, 0)
            # print(label.shape)                          # torch.Size([1, 1, 576, 768])
            image = torch.cuda.FloatTensor(image)
            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)

            height = image.shape[2]
            width = image.shape[3]
            test_img_ir,nn,_ = get_img_parts(label, height, width)      # 切块
            # print(test_img_ir[0].shape)                 # torch.Size([1, 1, 60, 60])
            test_img_vi,mm,_ = get_img_parts(image, height, width)      # 切块
            k = len(test_img_vi)        # 有k个64*64的块


            start = time.perf_counter()

            fusion_results = []

            for m in range(k):
                out = model(test_img_ir[m], test_img_vi[m])     # 拿一个64*64的小块输入模型得到融合图像
                fusion_results.append(out)                          # [1,1,64,64]的小块组成的列表
                
            out_img = recons_fusion_images(fusion_results, nn, height, width)
            
            end = time.perf_counter()
            tim = end - start
            print('cost' + str(tim) + 's')

            out_img = out_img*127.5 + 127.5
            out_img = out_img.squeeze().cpu().detach().numpy()
            out_img = np.clip(out_img,0,255)

            imsave(out_img.astype(np.uint8), "./result/"+ str(ii+1)+".bmp")

