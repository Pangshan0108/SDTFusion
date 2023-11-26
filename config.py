import warnings

class Config():
	train_ir_dir = './images/source_ir'
	train_vi_dir = './images/source_vi'
	checkpoint_dir = "checkpoint"
	img_size = 64          # 每张红外和可见光图像都切成60*60大小
	label_size = 64        # 融合不像分类，事先没有标签，所以在融合问题的训练中，标签就是源图像，即红外和可见光图像，也切成60*60大小
	stride = 64
	use_gpu = True            # 是否使用gpu
	num_workers = 4          # 加载数据时使用子进程个数
	batch_size = 4         # 批处理大小
	max_epochs = 20         # 训练最大次数
	lr = 0.0001             # 初始学习速率
	weight_decay = 1e-4      # 权重衰减，防止过拟合
	momentum = 0.099        # 梯度下降动量系数
	step_size = 100          # 学习速率调整周期
	gamma = 0.01             # 学习速率调整的加权系数
	b1 = 0.5
	b2 = 0.99
	lambda_gp_v = 10
	lambda_gp_ir = 10
	
def parse(self, kwargs):
	for k, v in kwargs.iteritems():
		if not hasattr(self, k):
			warnings.warn("Warning: opt has not attribute %s" % k)
		setattr(self, k, v)

	print('user config:')
	for k, v in self.__class__.__dict__.iteritems():
		if not k.startswith('__'):
			print(k, getattr(self, k))


Config.parse = parse    # ？？？？？
opt = Config()