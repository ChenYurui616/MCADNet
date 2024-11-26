#demo
import torch
import torch.nn as nn
import torch.utils.data as Data
import math
import numpy as np

#ECAAttention
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16, act_layer=nn.GELU,drop=0.,b=1, gama=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size
 
        padding = kernel_size // 2

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  #b,64,8,8

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False),
            nn.Dropout(drop)
        )
 
    def forward(self, inputs):

        max_out = self.mlp(self.max_pool(inputs))
        avg_out = self.mlp(self.avg_pool(inputs))
        ch_att_value = self.sigmoid(max_out + avg_out)
        ch_out = ch_att_value*inputs
        
        return ch_out             #b,512,8,8


    
class SpatialAttention(nn.Module):
    def __init__(self,channel,kernel_size=7,reduction=16):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.conv = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)    
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.conv_1 = nn.Conv2d(channel, 1, kernel_size, padding=padding, bias=False)
        self.conv2_1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):  

        _,_,h,w = x.size()

        x_h = self.avg_h(x) 
        x_w = self.avg_w(x).permute(0, 1, 3, 2) #交换长宽

        #x_cat_conv_relu = self.relu(self.conv1(torch.cat((x_h, x_w), 3)))
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.relu(y)

        #x_cat_conv_split_h, x_cat_conv_split_w = y.split([self.h, self.w], 3)
        x_h,x_w = torch.split(y, [h, w], dim=2)
        s_h = self.sigmoid(self.conv(x_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid(self.conv(x_w))
        avg_out = x * s_h * s_w
        
        avg_out = self.conv_1(avg_out)

        #avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        #spatial_out = self.sigmoid(self.conv1(torch.cat([max_out, avg_out], dim=1)))
        out = torch.cat([max_out, avg_out], dim=1)
        spatial_out = self.conv2_1(out)
        spatial_out = self.sigmoid(spatial_out)

        #outputs = self.conv1(outputs)  # 30,1,50,30
        return spatial_out  

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention(channel=512, reduction=16)

    def forward(self, x):
        ch_out = self.channel_attention(x)
        sp_att_value = self.spatial_attention(ch_out)
        out = ch_out * sp_att_value

        return out
