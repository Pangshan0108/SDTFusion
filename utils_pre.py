# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""
from PIL import Image
import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import imageio
import argparse
import numpy as np
import torch
from torch import nn
import cv2

def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:   # 'r'表示只读，hf只在with里起作用
        data = np.array(hf.get('data'))    # 返回numpy的数组类型
        label = np.array(hf.get('label'))
        return data, label

# 提取图片地址
def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
     # 调动训练集
    filenames = os.listdir(dataset)                   #返回指定的文件夹dataset包含的文件的名字的列表，但默认顺序是无序的，需要重新排序
    data_dir = os.path.join(os.getcwd(), dataset)     # os.getcwd()取得当前工作路径的字符串，os.path.join将dataset拼接到当前工作路径的最后
                                                      # data_dir是当前dataset的路径
    data = glob.glob(os.path.join(data_dir, "*.bmp"))  # os.path.join将当前dataset路径与dataset文件夹里的bmp文件拼接上，即可返回文件夹中所有的bmp文件路径
                                                    # 查找bmp文件，存入列表中，glob.glob()返回路径中所有bmp文件的路径列表list
    data.extend(glob.glob(os.path.join(data_dir, "*.tif"))) # glob.glob()返回路径中所有tif文件的路径列表list，extend将list添加到data列表的末尾
    # 将图片按序号排序
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))   #对data列表中图片排序

    # print(data)
    return data             # 返回dataset文件夹中所有图片按顺序排好的路径组成的列表

# 将文件以h5py的形式保存在savepath路径中
def make_data(data, label, data_dir):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """

    savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))  # 当前目录下checkpoint文件夹-data_dir文件夹-train.h5文件名
    if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))): # 如果当前目录下没有checkpoint文件夹-data_dir文件
        os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))        # 在当前目录下建立checkpoint文件夹-data_dir文件

    with h5py.File(savepath, 'w') as hf:        # with打开后面的文件，as表示hf变量就是h5py.File(savepath, 'w')该文件对象，w表示为可写的文件对象
        hf.create_dataset('data', data=data)    # with...as离开with结构后hf自动关闭
        hf.create_dataset('label', data=label)  # 在savepath路径中的train.h5文件中创建2个表，把data和label放进去


# 将图像数据保存为文件
# def imread(path, is_grayscale=True):
#     """
#     Read image using its path.
#     Default value is gray-scale, and image is read by YCbCr format as the paper said.
#     """
#     if is_grayscale:
#         # flatten=True 将颜色层展平为单个灰度图层  mode='YCbCr'将图像转换为YCbCr模式
#         return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)   # 默认是以YCbCr模式读取
#     else:
#         return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    img = Image.open(path).convert('YCbCr')
    if is_grayscale:
        img = img.convert('L')
    
    img = np.array(img)
    
    return img.astype(np.float64)


def input_setup(opt,data_dir,index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path

    # data：取到所有的原始图片的地址组成的列表
    data = prepare_data(dataset=data_dir)


    sub_input_sequence = []
    sub_label_sequence = []
    padding = int(abs(opt.img_size - opt.label_size) / 2)  # 6      ###  原程序中img_size设置为33，label_size设置为21，这里不用再导入opt


    for i in range(len(data)):
        # input_, label_ = preprocess(data[i], opt.scale)
        input_ = (imread(data[i]) - 127.5) / 127.5     # 对所有读入的图像做-1~1的归一化，灰度图0~255，所以这里用127.5
        label_ = input_                         # input_和label_一样

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # 按opt.stride步长采样小patch
        for x in range(0, h - opt.img_size + 1, opt.stride):        # 在0~h-opt.img_size+1范围内，以opt.stride的步长遍历
            for y in range(0, w - opt.img_size + 1, opt.stride):
                sub_input = input_[x:x + opt.img_size, y:y + opt.img_size]  # [120 x 120]
                # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
                sub_label = label_[x + padding:x + padding + opt.label_size,
                            y + padding:y + padding + opt.label_size]  # [120 x 120],本文中padding=0

                sub_input_sequence.append(sub_input)    # 切块的结果放在sub_input_sequence列表中
                sub_label_sequence.append(sub_label)
    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]      把列表转换为numpy二维数组[33,33]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]     把列表转换为numpy二维数组[21,21]
    # print(arrdata.shape)
    make_data(arrdata, arrlabel, data_dir)  # 将data_dir中的图片切块保存在当前目录下checkpoint文件夹-data_dir-train.h5文件

    # if not opt.is_train:
    #     print(nx, ny)
    #     print(h_real, w_real)
    #     return nx, ny, h_real, w_real


# def imsave(image, path):
#     return scipy.misc.imsave(path, image)

def imsave(image, path):
    img = Image.fromarray(image)
    img.save(path)


# 将一组图片进行组合
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return (img * 127.5 + 127.5)


