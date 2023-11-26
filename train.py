import argparse  # argparse 是 Python 内置的一个用于命令项选项与参数解析的模块
import os  # 该模块提供了一些方便使用操作系统相关功能的函数，如操作路径os.path模块
import numpy as np  # numpy是支持数组与矩阵运算的库
import math  # math数学函数运算库
import sys  # sys包含了与Python解释器和它的环境有关的函数
import torch.autograd as autograd  # 自动求导，根据输入和前向传播过程自动构建计算图，并执行反向传播
from torch.autograd import Variable  # Variable可以计算一个计算图，以实现自动求导
import torch.nn as nn  # torch神经网络库，方便创建、训练、保存、恢复神经网络
import torch
from config import *
from model.Trans_model import *
from utils_pre import *
from loss import fusion_loss_vif



gpus = [0]
device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')  # 若使用Gpu,torch.cuda.is_available()会变成1,这句话与后面的.to device配套使用
cuda = True if torch.cuda.is_available() else False


if __name__ == '__main__':      # 只有直接运行该脚本，该条件语句下面的程序才会被执行，如果该脚本是import到其他脚本中的，则这句话下面的语句不被执行
	# wind = SummaryWriter(log_dir='/data-output')
	# count = 0
	generator = TransFusion()               # 实例化类 #
	if cuda:
		generator = generator.to(device)    # 模型丢gpu0

	# Configure data loader 配置数据加载器
	input_setup(opt, "images/source_ir")  ## 当前脚本所在目录data文件夹-Train_ir文件夹，双引和单引没有区别，只是双引里面可以套单引
	input_setup(opt, "images/source_vi")
	data_dir_ir = os.path.join('.', opt.checkpoint_dir, "images/source_ir","train.h5")  # 路径拼接，当前目录下的checkpoint文件夹-Train_ir文件夹-train.h5文件(存放切好块的红外训练图像)
	data_dir_v = os.path.join('./{}'.format(opt.checkpoint_dir), "images/source_vi", "train.h5")  # format()里的字符串放到{}里去
	train_data_ir, train_label_ir = read_data(data_dir_ir)      # 返回切好块的红外训练图像的numpy数组类型
	train_data_v, train_label_v = read_data(data_dir_v)
	# 初始化优化器，模型所有参数和优化器设置参数丢进去
	gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	
	
	for epoch in range(opt.max_epochs):  # epoch=0~max_epochs-1,每次循环默认步长为1
		print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs))  # 第一次循环，print出来Epoch 1/60
		print('-' * 10)  # print出来10个-

		generator.train(True)                                    # 模型变成训练模式
		
		# Each epoch has a training and validation phase
		batch_idxs = len(train_data_ir) // opt.batch_size  # 共有batch_idxs个batch,//表示相除取整
		for idx in range(0, batch_idxs):
			batch_images_ir = train_data_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]  # 红外图像训练集中第一个batch赋值给batch_images_ir
			batch_images_v = train_data_v[idx * opt.batch_size: (idx + 1) * opt.batch_size]
			
			batch_images_ir = np.expand_dims(batch_images_ir, 1)  # 扩展numpy数组的形状，在原维度1(ch)的位置加入一个新维度
			batch_images_v = np.expand_dims(batch_images_v, 1)
			
			batch_images_ir = torch.autograd.Variable(torch.Tensor(batch_images_ir).to(device))  # batch_images_ir是numpy的数组类型转成tensor，丢到gpu，batch_images_ir是浮点型，则torch.Tensor将其转化成浮点型tensor
			batch_images_v = torch.autograd.Variable(torch.Tensor(batch_images_v).to(device))
			
			# Train the generator every 5 steps
			if (idx + 1) % 1 == 0:               # idx + 1除以5的余数如果=0，鉴别器每更新5次，生成器更新一次
				
				# -----------------
				#  Train Generator
				# -----------------
				# Generate a batch of images
				fusion_img = generator(batch_images_ir, batch_images_v)  # 返回生成器的输出，即融合图像
				
				fusion_loss = fusion_loss_vif()
				loss = fusion_loss(batch_images_ir, batch_images_v, fusion_img)     # [fusion_loss, loss_gradient, loss_inten, loss_SSIM]
				g_loss = loss[0]

				gen_optimizer.zero_grad()       # 优化器梯度清零

				g_loss.backward(retain_graph=True)
				gen_optimizer.step()
				# print(1)
				print("[Batch %d/%d] [fusion_loss: %f] [loss_gradient: %f] [loss_inten: %f] [loss_SSIM: %f]"
				      % (idx + 1, batch_idxs, g_loss.item(), loss[1].item(), loss[2].item(), loss[3].item()))                # 只有一个元素的tensor用.item()拿到其数值用于输出
		
	
		torch.save(generator.state_dict(), 'model%d.pkl' % epoch)  # 保存生成器模型中每个epoch中最后一个batch的所有参数在当前路径
		
				
				
"保存整个网络：torch.save(net, ‘net.pkl’)，保存网络的状态信息：torch.save(net.state_dict(), ‘net_params.pkl’)" \
				"包含模型的所有参数，提取网络的方法：torch.load(‘net.pkl’)"



