# CYY
# 2024.04.18

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math

class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(CoAttLayer, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.all_attention = Grafting(channel_in)   # GCAM→DFE
        self.attention = Grafting1(channel_in)
        #self.training = True

    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        if self.training:
            correlation_maps = F.conv2d(x5, weight=seeds)  # B,B,H,W
        else:
            correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))  # B,B,H,W
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps
    
    def forward(self, x):
        loadN = 2
        x = self.conv(x) + x #残差连接，增强网络的特征表示能力
        B, C, H, W = x.shape
        # if self.training:
        #     channel_per_class = x.shape[0] // loadN   #channel_per_class:5
        #     x_per_class_corr_list = []

        #     x_per_class_1 = x[0:channel_per_class]
        #     x_per_class_2 = x[channel_per_class:x.shape[0]]

        #     x_w = self.all_attention(x_per_class_1,x_per_class_2)
        # else:
        x_w = self.attention(x)
            
        x_w = torch.max(x_w, -1).values
        x_w = F.softmax(x_w, dim=-1)  # B, HW

        norm0 = F.normalize(x, dim=1)
        # norm = norm0.view(B, C, -1)
        # max_indices = max_indices0.expand(B, C, -1)
        # seeds = torch.gather(norm, 2, max_indices).unsqueeze(-1)
        x_w = x_w.unsqueeze(1)
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w) #1求x_w在最后一个维度的最大值，2在第3个维度增加一个大小为1的维度，3扩展到x_w相同的形状
        #[b,1,hw] [b,1] [b,1,1] [b,1,hw]
        mask = torch.zeros_like(x_w) #.cuda() #创建一个与x_w具有相同形状和数据类型的全0张量，并将这个张量移动到GPU
        mask[x_w == x_w_max] = 1  #把mask中所有x_w == x_w_max 的元素都赋值为1，生成一个标记了x_w中最大值的位置的掩码（mask）
        mask = mask.view(B, 1, H, W)
        
        seeds = norm0 * mask #选出了原始特征中最具代表性的种子seeds  
        #seeds[b, c, h, w]
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        #[b, c, 1, 1]
        cormap = self.correlation(norm0, seeds)
        x51 = x * cormap
        proto1 = torch.mean(x51, (0, 2, 3), True)
        return x, proto1, x*proto1+x51, mask

class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(512)
        self.lny = nn.LayerNorm(512)
        self.bn = nn.BatchNorm2d(8)
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = self.lny(y)
        
        B, N, C = x.shape
        y_qv= self.qv(y).reshape(B, N, 2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #[2, 196, 1, 8, 64]
        y_q, y_v = y_qv[0], y_qv[1]
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv= self.qv(x).reshape(B, N, 2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        x_k = self.k(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_k = y_k[0]
        attn_1 = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn_1 = attn_1.softmax(dim=-1)
        attn_2 = (y_q @ x_k.transpose(-2, -1)) * self.scale
        attn_2 = attn_2.softmax(dim=-1)
        x = (attn_1 @ x_v).transpose(1, 2).reshape(B, N, C)
        y = (attn_2 @ y_v).transpose(1, 2).reshape(B, N, C)
        weighted_x = torch.cat((x,y),dim=0) #b,hw,c
        return weighted_x
        
class Grafting1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(512)
        self.lny = nn.LayerNorm(512)
        self.bn = nn.BatchNorm2d(8)
    
    def forward(self, x):
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        x = self.lnx(x)
        
        B, N, C = x.shape
        x_qv= self.qv(x).reshape(B, N, 2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        x_k = self.k(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn_1 = (x_q @ x_k.transpose(-2, -1)) * self.scale
        attn_1 = attn_1.softmax(dim=-1)
        x = (attn_1 @ x_v).transpose(1, 2).reshape(B, N, C)
        return x






