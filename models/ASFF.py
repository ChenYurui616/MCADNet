'''
Descripttion:  三层的ASFF模块，对数据格式要求很高，建议读懂代码再使用，切不可盲目使用
Result:  根据level确定输出数据的维度
Author: Philo
Date: 2023-06-01 10:48:40
LastEditors: Philo
LastEditTime: 2023-06-07 17:25:21
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):  # level只有0、1、2
    def __init__(self, level, multiplier=0.5, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(1024*multiplier), int(1024*multiplier),
                    int(512*multiplier)]
        # print(self.dim)
        
        self.inter_dim = self.dim[self.level] #0:512, 1:256, 2:128
        if level == 0:  #512
            self.stride_level_1 = Conv(int(512*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 256, h, w]--[bs, 512, h/2, w/2]          
            self.stride_level_2 = Conv(int(256*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 128, h, w]--[bs, 512, h/2, w/2]          
            self.expand = Conv(self.inter_dim, int(1024*multiplier), 3, 1)    # size is still      [bs, 512, h, w]--[bs, 512, h, w]
        elif level == 1: #256
            self.stride_level_1 = Conv(int(512*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 256, h, w]--[bs, 512, h/2, w/2]          
            self.stride_level_2 = Conv(int(256*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 128, h, w]--[bs, 512, h/2, w/2]          
            self.expand = Conv(self.inter_dim, int(1024*multiplier), 3, 1)    # size is still      [bs, 512, h, w]--[bs, 512, h, w]
        elif level == 2: #128
            self.compress_level_0 = Conv(int(1024*multiplier), self.inter_dim, 1, 1)   # still  [bs, 512, h, w]--[bs, 128, h, w]
            self.compress_level_1 = Conv(int(512*multiplier), self.inter_dim, 1, 1)  # still    [bs, 256, h, w]--[bs, 128, h, w]
            self.expand = Conv(self.inter_dim, int(512*multiplier), 3, 1)  # still              [bs, 128, h, w]--[bs, 128, h, w]

        # if level == 0:  #512
        #     self.level0_conv256_512_sh = Conv(int(512*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 256, h, w]--[bs, 512, h/2, w/2]          
        #     #self.level0_conv256_512_sh = Conv(int(256*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 128, h, w]--[bs, 512, h/2, w/2]          
        #     self.expand = Conv(self.inter_dim, int(1024*multiplier), 3, 1)    # size is still      [bs, 512, h, w]--[bs, 512, h, w]
        # elif level == 1: #512
        #     self.level1_conv256_512_sh = Conv(int(512*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 256, h, w]--[bs, 512, h/2, w/2]          
        #     #self.level0_conv256_512_sh = Conv(int(256*multiplier), self.inter_dim, 3, 2)  # size is half  [bs, 128, h, w]--[bs, 512, h/2, w/2]          
        #     self.expand = Conv(self.inter_dim, int(1024*multiplier), 3, 1)    # size is still      [bs, 512, h, w]--[bs, 512, h, w]
        # elif level == 2: #256
        #     self.level2_conv512_256 = Conv(int(1024*multiplier), self.inter_dim, 1, 1)   # size still  [bs, 512, h, w]--[bs, 256, h, w]
        #     #self.level2_conv128_256_sh = Conv(int(256*multiplier), self.inter_dim, 3, 2)   # size is half   [bs, 128, h, w]--[bs, 256, h/2, w/2]
        #     self.expand = Conv(self.inter_dim, int(512*multiplier), 3, 1)  # size is still           [bs, 256, h, w]--[bs, 256, h, w]     

        # when adding rfb, we use half number of channels to save memory # 这个地方以下的代码可以好好学一下，然后融会贯通
        compress_c = 8 if rfb else 16  #通道压缩
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(compress_c*3, 3, 1, 1)  # channel is change, size is still
        self.vis = vis

    def forward(self, x): # x为列表数据，[f1, f2, f3] channel:f1<f2<f3  size:f1>f2>f3   
        """
        # 
        256, 512, 1024
        from small -> large
        """
        x_level_0=x[2] #最大特征层 x3 (1, 512, 14, 14)
        x_level_1=x[1] #中间特征层 x2 (1, 512, 14, 14)
        x_level_2=x[0] #最小特征层 x1 (1, 256, 28, 28)

        if self.level == 0:
            level_0_resized = x_level_0 #x3 (1, 512, 14, 14)
            #level_1_resized = self.stride_level_1(x_level_1)  #(1, 256, 28, 28)→(1, 512, 14, 14)
            level_1_resized = x_level_1
            level_2_resized  = self.stride_level_1(x_level_2)  #(1, 256, 28, 28)→(1, 512, 14, 14)
        elif self.level == 1:
            level_0_resized = x_level_0 #x3 (1, 512, 14, 14)
            #level_1_resized = self.stride_level_1(x_level_1)  #(1, 256, 28, 28)→(1, 512, 14, 14)
            level_1_resized = x_level_1
            level_2_resized  = self.stride_level_1(x_level_2)  #(1, 256, 28, 28)→(1, 512, 14, 14)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)   #(1, 512, 28, 28)→(1, 256, 28, 28)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest') #(1, 256, 28, 28)→(1, 256, 14, 14)

            level_1_compressed = self.compress_level_0(x_level_1)  #(1, 512, 14, 14)→(1, 256, 14, 14)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest') #(1, 256, 28, 28)→(1, 256, 14, 14)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized) #(1, 512, 14, 14)→(1, 16, 14, 14)
        level_1_weight_v = self.weight_level_1(level_1_resized) #(1, 512, 14, 14)→(1, 16, 14, 14)
        level_2_weight_v = self.weight_level_2(level_2_resized) #(1, 512, 14, 14)→(1, 16, 14, 14)
        #print(level_0_weight_v.shape, level_1_weight_v.shape, level_2_weight_v.shape)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)  #（1，48，14，14）
        levels_weight = self.weight_levels(levels_weight_v) #（1，48，14，14）→（1，3，14，14）
        levels_weight = F.softmax(levels_weight, dim=1)  #权重张量（1，3，14，14）

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:, :, :]  #加权后的特征图通过元素及加法融合在一起 (1, 512, 14, 14)

        out = self.expand(fused_out_reduced)  #(1, 512, 14, 14)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

# 输入的数据维度为128 * w * h、   256 * w/2 *h/2、    512 * w/4 * h/4  实例化对象时，multiplier=0.5, level决定输出哪个level的尺寸
# x1 = torch.randn(1, 256, 28, 28)
# x2 = torch.randn(1, 512, 14, 14)
# x3 = torch.randn(1, 512, 14, 14)
# net1 = ASFF(level=0,multiplier=0.5)
# net2 = ASFF(level=1,multiplier=0.5)
# net3 = ASFF(level=2,multiplier=0.5)
# print('net1:', net1([x1, x2, x3]).shape)
# print('net2:', net2([x1, x2, x3]).shape)
# print('net3:', net3([x1, x2, x3]).shape)

# 256 * w * h、   512 * w/2 *h/2、    1024 * w/4 * h/4  实例化对象时，multiplier=1, level决定输出哪个level的尺寸

# m1 = torch.randn(1, 256, 128, 128)
# m2 = torch.randn(1, 512, 64, 64)
# m3 = torch.randn(1, 1024, 32, 32)
# net2 = ASFF(level=2,multiplier=1)
# print(net2([m1, m2, m3]).shape)
