
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math

class LeFF(nn.Module):              # 代替传统Transformer块中的mlp
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity() if use_eca else nn.Identity()      # nn.Identity()什么也不做,直接返回输入

    def forward(self, x):           # IN:[B,HW,C]  OUT:[B,HW,C]
        # bs x hw x c
        bs, hw, c = x.size()
        shortcut = x
        hh = int(math.sqrt(hw))     # 将变量hw开平方并取整后赋值给变量hh

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)
        x = x + shortcut

        return x


#####定义深度可分离卷积，包含深度卷积和点卷积
class SepConv2d(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, act_layer=nn.ReLU):  # 普通卷积层默认dilation=1
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)  # 深度卷积groups设置为输出通道数
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    
    def forward(self, x):           # IN:[B,C,H,W] OUT:[B,C,H,W]
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


######## 利用深度可分离卷积Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim=32, heads=8//2, dim_head=4, kernel_size=3, q_stride=1, k_stride=1, v_stride=1):
        super().__init__()
        
        # inner_dim = dim_head * heads            # dim_head是每个头输出维度，总维度为inner_dim(32)
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, dim, kernel_size, q_stride, pad)       # OUT:[B, 32, H, W]
        self.to_k = SepConv2d(dim, dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, dim, kernel_size, v_stride, pad)
    
    def forward(self, x, attn_kv=None):             # attn_kv=None就是自注意力，即qkv都是x映射得到的
        b, n, c, h = *x.shape, self.heads           # x,attn_kv:[B_, N, C] N=Wh*Ww B_=nW*B
        l = int(math.sqrt(n))                       # l,w=win_size
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)      # x:[B_, C, win_size, win_size]
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)                # q:[B_,C,win_size,win_size]
        k = self.to_k(x)
        v = self.to_v(x)
        q1, q2 = torch.split(q, q.size(1)//2, dim=1)    # q1,q2:[B_,C//2,win_size,win_size]
        q1 = rearrange(q1, 'b (h d) l w -> b h (l w) d', h=h)         # [B_,heads,win_size*win_size,dim_heads]
        q2 = rearrange(q2, 'b (h d) l w -> b h (l w) d', h=h)
        k1, _ = torch.split(k, k.size(1)//2, dim=1)
        k1 = rearrange(k1, 'b (h d) l w -> b h (l w) d', h=h)       # k1
        v1, _ = torch.split(v, v.size(1) // 2, dim=1)
        v1 = rearrange(v1, 'b (h d) l w -> b h (l w) d', h=h)       # v1
        
        k_ = self.to_k(attn_kv)
        v_ = self.to_v(attn_kv)
        k2, _ = torch.split(k_, k_.size(1) // 2, dim=1)
        k2 = rearrange(k2, 'b (h d) l w -> b h (l w) d', h=h)       # k2
        v2, _ = torch.split(v_, v_.size(1) // 2, dim=1)
        v2 = rearrange(v2, 'b (h d) l w -> b h (l w) d', h=h)       # v2
        return q1, q2, k1, k2, v1, v2          # [B_, heads, win_size*win_size, dim_heads]
    
######## 利用线性层Embedding for q,k,v ########
class LinearProjection(nn.Module):
    def __init__(self, dim=32, heads =8, dim_head =4, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:     # kv不是x映射的，有指定的输入
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)   # 增加一个第0维，然后在第0维上面复制B_次，其余维度不变
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]        # [B_, self.heads, N, C // self.heads]
        k, v = kv[0], kv[1]     # [B_, self.heads, N, C // self.heads]
        return q,k,v


########### window operation#############
def window_partition(x, win_size, dilation_rate=1):         # IN:[B, H, W, C]  OUT:[B' ,Wh ,Ww ,C]
    B, H, W, C = x.shape
    if dilation_rate !=1:       # 这里用F.unfold对输入切块,切块大小为win_size*win_size,步长为win_size即没有重叠
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):       # IN:[B' ,Wh ,Ww ,C]  OUT:[B, H, W, C]
    # B'= B * H/Wh*W/Ww
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads=8//2, token_projection='conv', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        
        self.dim = dim                          # 输入的通道数
        self.win_size = win_size                # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads // 2            # head_dim是每个头的通道数,为了保证8个头分开两组后,head_dim和之前保持相同,故除以2
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww,最后一维求和后最后一个维度没了
        self.register_buffer("relative_position_index", relative_position_index)  # relative_position_index是相对位置一维坐标值
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):      # IN:[B_, N, C]  OUT:[B_, N, C]  N=Wh*Ww,输入已经切好块
        B_, N, C = x.shape
        q1, q2, k1, k2, v1, v2 = self.qkv(x, attn_kv)  # [B_, self.heads, N, C // self.heads]
        q1 = q1 * self.scale                    # n/2 heads的
        q2 = q2 * self.scale                    # 另外n/2 heads的
        attn1 = (q1 @ k1.transpose(-2, -1))  # k最后两个维度转置,attn:[B_, self.heads, N, N],N=Wh*Ww
        attn2 = (q2 @ k2.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn1.size(-1) // relative_position_bias.size(-1)  # 这里ratio=1
        # print(ratio)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        
        attn1 = attn1 + relative_position_bias.unsqueeze(0)
        attn2 = attn2 + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N * ratio)
            attn1 = self.softmax(attn1)
            attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N * ratio)
            attn2 = self.softmax(attn2)
        else:
            attn1 = self.softmax(attn1)
            attn2 = self.softmax(attn2)
        
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        
        x1 = attn1 @ v1
        x2 = attn2 @ v2
        x = torch.cat((x1, x2), dim=1)              # 按照heads cat,得到共8个头 [B_, heads ,N, dim_heads]
        x = x.transpose(1, 2).reshape(B_, N, C)
        
        x = self.proj(x)        # 自注意力*v之后再线性层一次
        x = self.proj_drop(x)
        return x
    

# Input Projection
class InputProj(nn.Module):             # 卷积+reshape(4d→3d)
    def __init__(self, in_channel=1, out_channel=32, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)     # inplace=True,会直接将计算结果覆盖在原始输入张量上,但梯度计算要基于覆盖后的输出值而不是输入值,会更加复杂应谨慎使用
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):       # IN: [B,C,H,W] OUT: [B H*W C]
        B, C, H, W = x.shape            # 输入的x形状是[B,C,H,W]
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
# Output Projection
class OutputProj(nn.Module):        # 卷积+reshape(3d→4d)
    def __init__(self, in_channel=256, out_channel=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):       # IN: [B H*W C] OUT: [B,C,H,W]
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        
        return x


########### Transformer: Inter-Attn,LeFF #############
class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8//2, win_size=8, shift_size=8//2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='conv', token_mlp='leff'):
        super().__init__()

        self.input_resolution = [input_resolution]      # 必须是可迭代的类型才能放在min里面
        self.win_size = win_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.win_size:  # 如果输入分辨率高或宽比窗口大小还小
            self.shift_size = 0  # 不需要shifted window
            self.win_size = min(self.input_resolution)  # 调整窗口大小
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"  # ""是报错时的提示信息,当不符合条件会抛出一个AssertionError异常
        

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)           # 在LeFF中通道膨胀到4倍再调整回原通道数量
        
        if token_mlp in ['ffn', 'mlp']:
            pass
            # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, y_as_kv, mask=None):        # IN:[B, HW, C] OUT:[B, HW, C]
        B, L, C = x.shape                           # 第一个块输入[4,3600,32]
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
    
        ## input mask
        if mask != None:  # 如果有输入一个mask
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)  # 插值运算使得输入的mask与特征图尺寸相同,再将维度调整为相同顺序
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
    
        ## shift mask
        if self.shift_size > 0:  # 生成一个mask
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:              # 标号
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
            # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            # 如果有输入一个mask则将2个mask加起来，如果没有，则使用生成的shift mask
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
    
        shortcut = x
        x = self.norm1(x)               # trans块中第一个LN
        x = x.view(B, H, W, C)          # [4,60,60,32]第一个块
        y_as_kv = self.norm1(y_as_kv)
        y_as_kv = y_as_kv.view(B, H, W, C)
    
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y_as_kv = torch.roll(y_as_kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y_as_kv = y_as_kv
    
        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        y_as_kv_win = window_partition(shifted_y_as_kv, self.win_size)
        y_as_kv_win = y_as_kv_win.view(-1, self.win_size * self.win_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_kv= y_as_kv_win, mask=attn_mask)  # nW*B, win_size*win_size, C
    
        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H W C
    
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

###trans block + dense connection,输入[B,HW,C],输出[B,HW,2C]
class Trans_denseBlock(nn.Module):
    def __init__(self, dim, input_resolution,drop_path1=0,drop_path2=0, num_heads=8//2, win_size=8):
        super(Trans_denseBlock,self).__init__()
        self.transblock1 = TransformerBlock(dim, input_resolution, num_heads, win_size=8, shift_size=win_size//2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=drop_path1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='conv', token_mlp='leff')
        self.transblock2 = TransformerBlock(dim*2, input_resolution, num_heads, win_size=8, shift_size=win_size//2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=drop_path2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='conv', token_mlp='leff')
        self.conv1 = nn.Conv2d(dim*4, dim*2, 1, 1, 0, bias=False)
    
    def forward(self, x, y, mask=None):
        shortcut_1 = x                                  # [B, HW, C]
        trans1_out = self.transblock1(x, y, mask)          # [B, HW, C], C=32
        shortcut_2 = trans1_out                         # [B, HW, C]
        trans2_in = torch.cat((trans1_out, shortcut_1), dim=-1)     # [B, HW, 2C]
        y_2 = torch.cat((y, y), dim=-1)                 # y的通道数扩展至2倍
        trans2_out = self.transblock2(trans2_in, y_2, mask)              # [B, HW, 2C]
        trans_out = torch.cat((trans2_out, shortcut_1, shortcut_2), dim=-1)      # [B, HW, 4C]
        h = int(math.sqrt(trans_out.shape[1]))          # 对HW开根号
        trans_out = rearrange(trans_out, 'b (h w) c -> b c h w', h=h)       # [B, 4C, H, W]
        out = self.conv1(trans_out)                 # [B, 2C, H, W]
        out = rearrange(out, 'b c h w -> b (h w) c')      # [B, HW, 2C]
        
        return out

##################### Fudion Module ##############################
###双向门控单元,不包括残差相加这一步
class GatedDconv(nn.Module):
    def __init__(self, dim, bias=False):  # 本文dim=256
        super(GatedDconv, self).__init__()
        self.project_in_x = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.project_in_y = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.dwconv_x = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.dwconv_y = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
    
    def forward(self, x, y_weight):  # 这里输入的x,y已经经过Layer Norm之后了, IN:[B,C,H,W],OUT:[B,C,H,W]
        x = self.project_in_x(x)
        y_weight = self.project_in_y(y_weight)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)     # .chunk(2, dim=1)将x经过深度卷积后的特征图按照通道维度分为2个特征图,x1和x2都是hidden_features通道数
        x = self.dwconv_x(x)
        y_weight = self.dwconv_y(y_weight)
        x = F.gelu(y_weight) * x                      # 经过激活函数之后x1尺寸不变,逐元素相乘
        x = self.project_out(x)
        return x


########## 自定义LayerNorm ###########
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):  # normalized_shape是整数类型
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1  # normalized_shape的长度必须为1,否则触发AssertionErro异常,程序执行将停止
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):  # x:[b,(hw),c]
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 沿通道维度求方差，输出尺寸不变
        return x / torch.sqrt(sigma + 1e-5) * self.weight  # x除以标准差，+1e-5避免除以0，*self.weight乘以可学习的缩放尺度因子，使输出更灵活


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)  # 沿通道维度求均值
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 沿通道维度求方差
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias  # x减去均值除以标准差


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):       # IN,OUT:[b,c,h,w]
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


#####################
########融合模块(添加了双向门控单元)
class FusionModule(nn.Module):
    def __init__(self, dim):
        super(FusionModule, self).__init__()
        self.norm = LayerNorm(dim=dim, LayerNorm_type='BiasFree')
        self.Gate_xy = GatedDconv(dim=dim, bias=False)
        # self.norm_y = LayerNorm(dim=dim, LayerNorm_type='BiasFree')
        self.Gate_yx = GatedDconv(dim=dim, bias=False)
    
    def forward(self, x, y):  # 输入的红外和可见光特征图通道均为256, IN:[B, C, H, W] OUT:[B, 2C, H, W]
        shortcut_x = x
        shortcut_y = y
        x = self.norm(x)
        y = self.norm(y)
        x_y_weight = self.Gate_xy(x, y)
        x_y_weight = x_y_weight + shortcut_x
        y_x_weight = self.Gate_yx(y, x)
        y_x_weight = y_x_weight + shortcut_y
        out = torch.cat([x_y_weight, y_x_weight], dim=1)  # [B, 2C, H, W]
        
        return out
##########################################################################

##################### Reconstruction Module ##############################
######## 利用深度可分离卷积Embedding for q,k,v ########
class Rec_ConvProjection(nn.Module):
    def __init__(self, dim, heads, dim_head, kernel_size=3, q_stride=1, k_stride=1, v_stride=1):
        super().__init__()
        
        inner_dim = dim_head * heads  # dim_head是每个头输出维度，总维度为inner_dim
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
    
    def forward(self, x, attn_kv=None):  # attn_kv=None就是自注意力，即qkv都是x映射得到的
        b, n, c, h = *x.shape, self.heads  # x:[nW*B, 6*6, 64]
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))
        if attn_kv is not None:  # attn_kv:[6*6, 64]
            attn_kv = attn_kv.unsqueeze(0).repeat(b, 1, 1)  # attn_kv:[nW*B, 6*6, 64]
        else:
            attn_kv = x
        
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)  # x:[nW*B, 64, 6, 6]
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)  # attn_kv:[nW*B, 64, 6, 6]
        # print(attn_kv)
        q = self.to_q(x)  # q: [nW*B, 64, 6, 6]
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)         # q:[nW*B, 8, 6*6, 8]
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v  # q,k,v: [nW*B, heads, win_size*win_size, dim_head]=[nW*B, 8, 6*6, 8]


class Rec_WindowAttention(nn.Module):  # 没改动
    def __init__(self, dim, win_size, num_heads, token_projection='conv', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        
        self.dim = dim  # 输入的通道数
        self.win_size = [win_size,win_size]  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads  # head_dim是每个头的通道数
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.win_size[0] - 1) * (2 * self.win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww,最后一维求和后最后一个维度没了
        self.register_buffer("relative_position_index", relative_position_index)  # relative_position_index是相对位置一维坐标值
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        if token_projection == 'conv':
            self.qkv = Rec_ConvProjection(dim, num_heads, dim // num_heads)
        else:
            raise Exception("Projection error!")
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):  # IN:[B_, N, C]  OUT:[B_, N, C]  N=Wh*Ww,输入已经切好块
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)          # [B_, heads, N, C // heads]=[nW*B, 8, 6*6, 8]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))        # k最后两个维度转置,attn:[B_, heads, N, N],N=Wh*Ww
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)  # 这里ratio=1
        # print(ratio)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)  # attn:[B_, heads, N, N],N=Wh*Ww
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # 自注意力*v之后再线性层一次
        x = self.proj_drop(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv11_in = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1), nn.LeakyReLU())
        self.conv33 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU())
        self.conv11_out = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.resi = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.active = nn.LeakyReLU()
    
    def forward(self, x):  # IN:[B,in_channel,H,W]  OUT:[B,out_channel,H,W]
        resi = x
        x = self.conv11_out(self.conv33(self.conv11_in(x)))
        resi = self.resi(resi)
        x = self.active(x + resi)
        
        return x

class Modulator(nn.Module):
    def __init__(self, img_size, modulator_dim, VI_dim=64, win_size=8, num_heads=8, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm, token_projection='conv'):
        super(Modulator, self).__init__()
        self.modulator_dim = modulator_dim
        self.modulator = nn.Embedding(img_size * img_size,
                                      modulator_dim)  # 默认使用均匀分布初始化,输出(64*64,modulator_dim)的向量,每行代表特征图中一个像素
        self.cross_modulator = nn.Embedding(win_size * win_size, VI_dim)  # 输出(8*8,dim)的向量
        self.cross_attn = Rec_WindowAttention(VI_dim, win_size, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop, token_projection=token_projection)
        self.norm_cross = norm_layer(VI_dim)
        self.win_size = win_size
    
    def forward(self, x, cross_modulator=False):                # x:[B,HW,64]
        if cross_modulator:                                     # q(输入x)是VI浅层特征图,kv是噪声信号
            B, L, C = x.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            modulator_kv = self.cross_modulator  # [8*8,64]
            x = self.norm_cross(x)
            x = x.reshape(B, H, W, C)
            # partition windows
            x_windows = window_partition(x, self.win_size)
            x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # [nW*B, win_size*win_size, 64]
            # W-MSA
            modulator = self.cross_attn(x_windows, modulator_kv.weight)  # [nW*B, win_size*win_size, 64]
            # merge windows
            modulator = modulator.view(-1, self.win_size, self.win_size, C)  # [nW*B, win_size, win_size, 64]
            modulator = window_reverse(modulator, self.win_size, H, W)  # [B, H, W, C]
            modulator = modulator.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            modulator = modulator.repeat(1, self.modulator_dim//C, 1, 1)    # 将modulator通道数调整为和特征图相同便于相加
        # modulator = modulator + skip
        else:  # 生成和特征图尺寸相同的噪声信号
            modulator = self.modulator.weight  # self.modulator.weight是张量[img_size * img_size, modulator_dim]
            L, C = modulator.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            modulator = modulator.transpose(0, 1).contiguous()  # [modulator_dim, img_size * img_size]
            modulator = modulator.view(C, H, W).unsqueeze(0)  # [1,C,H,W]
        return modulator

######输出Extraction Module可见光分支第一个Trans_denseBlock的VI_Feature
class VI_Feature_Out(nn.Module):
    def __init__(self, img_size=64, drop_path1=0, drop_path2=0):
        super(VI_Feature_Out, self).__init__()
        self.input_project = InputProj(in_channel=1, out_channel=32, kernel_size=3, stride=1, norm_layer=None,
                                       act_layer=nn.LeakyReLU)
        self.extract_vi_1 = Trans_denseBlock(dim=32, input_resolution=img_size, drop_path1=drop_path1,
                                             drop_path2=drop_path2, num_heads=8 // 2)

    def forward(self, vi, ir):
        vi_in = self.input_project(vi)          # [B,HW,32]
        ir_in = self.input_project(ir)
        vi_feature = self.extract_vi_1(vi_in, ir_in)   # [B,HW,64]
            
        return vi_feature

class ReconstructionModule(nn.Module):
    def __init__(self, img_size):
        super(ReconstructionModule, self).__init__()
        ch = [512, 256, 128, 64, 1]
        self.Res1 = ResBlock(in_channel=ch[0], out_channel=ch[1])
        self.Res2 = ResBlock(in_channel=ch[1], out_channel=ch[2])
        self.Res3 = ResBlock(in_channel=ch[2], out_channel=ch[3])
        self.Res4 = ResBlock(in_channel=ch[3], out_channel=ch[4])
        self.modulator1 = Modulator(img_size, modulator_dim=ch[0], num_heads=8, qkv_bias=True, qk_scale=None, drop=0.,
                                    attn_drop=0., norm_layer=nn.LayerNorm, token_projection='conv')
        self.modulator2 = Modulator(img_size, modulator_dim=ch[1], num_heads=8, qkv_bias=True, qk_scale=None, drop=0.,
                                    attn_drop=0., norm_layer=nn.LayerNorm, token_projection='conv')
        self.modulator3 = Modulator(img_size, modulator_dim=ch[2], num_heads=8, qkv_bias=True, qk_scale=None, drop=0.,
                                    attn_drop=0., norm_layer=nn.LayerNorm, token_projection='conv')
        self.modulator4 = Modulator(img_size, modulator_dim=ch[3], num_heads=8, qkv_bias=True, qk_scale=None, drop=0.,
                                    attn_drop=0., norm_layer=nn.LayerNorm, token_projection='conv')
        self.VI_Feature = VI_Feature_Out(img_size)
    
    def forward(self, x, vi, ir):  # x:[B,C,H,W], VI_Feature:[B, HW, 64]
        VI_Feature = self.VI_Feature(vi, ir)
        recon = self.Res1(x + self.modulator1(VI_Feature, cross_modulator=True))   # 这里的cross_modulator控制modulator的噪声信号用哪种
        recon = self.Res2(recon + self.modulator2(VI_Feature, cross_modulator=True))
        recon = self.Res3(recon + self.modulator3(VI_Feature, cross_modulator=True))
        recon = self.Res4(recon + self.modulator4(VI_Feature, cross_modulator=True))
        
        return recon

########################################################################

###总架构
class TransFusion(nn.Module):
    def __init__(self, img_size=64):
        super(TransFusion,self).__init__()
        # build layers
        # Input/Output
        self.input_project = InputProj(in_channel=1, out_channel=32, kernel_size=3, stride=1, norm_layer=None,
                                       act_layer=nn.LeakyReLU)
        self.output_project = OutputProj()
        
        # drop_path
        extrac_dpr = [x.item() for x in torch.linspace(0, 0.1, 6)]
        
        # Extraction Module
        self.extract_ir_1 = Trans_denseBlock(dim=32, input_resolution=img_size, drop_path1=extrac_dpr[0],
                                             drop_path2=extrac_dpr[1], num_heads=8//2)
        self.extract_ir_2 = Trans_denseBlock(dim=64, input_resolution=img_size, drop_path1=extrac_dpr[2],
                                             drop_path2=extrac_dpr[3], num_heads=8//2)
        self.extract_ir_3 = Trans_denseBlock(dim=128, input_resolution=img_size, drop_path1=extrac_dpr[4],
                                             drop_path2=extrac_dpr[5], num_heads=8//2)
        # self.extract_vi_1 = Trans_denseBlock(dim=32, input_resolution=img_size, num_heads=8//2)
        self.vi_feature = VI_Feature_Out(img_size=img_size, drop_path1=extrac_dpr[0], drop_path2=extrac_dpr[1])
        self.extract_vi_2 = Trans_denseBlock(dim=64, input_resolution=img_size, drop_path1=extrac_dpr[2],
                                             drop_path2=extrac_dpr[3], num_heads=8//2)
        self.extract_vi_3 = Trans_denseBlock(dim=128, input_resolution=img_size, drop_path1=extrac_dpr[4],
                                             drop_path2=extrac_dpr[5], num_heads=8//2)
        
        # Fusion Module
        self.fusion_layer = FusionModule(dim=256)      # IN:[B, HW, C]  OUT:[B, 2C, H, W]
        
        # Reconstruction Module
        self.reconstrction_layer = ReconstructionModule(img_size=img_size)
        
    def forward(self, ir, vi):
        ir_in = self.input_project(ir)
        vi_in = self.input_project(vi)
        ir_extract_1 = self.extract_ir_1(ir_in, vi_in)
        vi_extract_1 = self.vi_feature(vi, ir)
        ir_extract_2 = self.extract_ir_2(ir_extract_1, vi_extract_1)
        vi_extract_2 = self.extract_vi_2(vi_extract_1, ir_extract_1)
        ir_extract_3 = self.extract_ir_3(ir_extract_2, vi_extract_2)     # [B, HW, C]
        vi_extract_3 = self.extract_vi_3(vi_extract_2, ir_extract_2)

        ir_extract_out = self.output_project(ir_extract_3)
        vi_extract_out = self.output_project(vi_extract_3)
        
        fusion_out = self.fusion_layer(ir_extract_out, vi_extract_out)
        
        out = self.reconstrction_layer(fusion_out, vi, ir)
        
        return out
