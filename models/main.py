import torch
from torch import nn
import torch.nn.functional as F
from models.Swin import Swintransformer
from util import *

from models.decoder import Decoder
from models.CrossAttention import CrossAttLayer
from models.DPG import AttLayer
from models.DFE import AugAttentionModule
from models.MCM import MCM
from models.ASFF import ASFF
from models.SCattention import CBAM
from models.graft_dpg import CoAttLayer
#from models.cloformer import EfficientBlock


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class YYNet(nn.Module):

    def __init__(self, mode=''):
        super(YYNet, self).__init__()
        self.gradients = None
        self.backbone = Swintransformer()
        self.backbone.load_state_dict(
             torch.load('/hy-tmp/pth/swin224.pth')['model'],strict = False) 
        self.mode = mode

        self.net0 =  ASFF(level=0,multiplier=0.5)
        self.net1 =  ASFF(level=1,multiplier=0.5)
        self.net2 =  ASFF(level=2,multiplier=0.5)

        self.crossattn = CrossAttLayer(512)   #[bs, 512, 14, 14]---[bs, 512, 14, 14]
        self.mcm = MCM(512)
        self.aug = AugAttentionModule()
        self.fusion = AttLayer(512) #DPG:
        self.enhance = CoAttLayer(512)
        self.cbam = CBAM(512)

        # self.cloformer1_2 = EfficientBlock(128, 256, 4, [1, 1, 1, 1], [7, 5, 3], 7, 7, 4, 2)
        # self.cloformer2_3 = EfficientBlock(256, 512, 4, [1, 1, 1, 1], [7, 5, 3], 7, 7, 4, 2)

        
        self.decoder = Decoder()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):
        N, _, H, W = x.size()

        y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)

        s1,s2,s3,s4 = self.backbone(y)

        return s1,s2,s3,s4

    def _forward(self, x, gt):
        B, _, H, W = x.size()
        #encoder
        s1,s2,s3,s4= self.featextract(x)  #s1:[bs, 128, 56, 56] s2:[bs, 256, 28, 28] s3:[bs, 512, 14, 14] s4:[bs, 512, 14, 14]
        
        feat, proto, weighted_x5, cormap = self.enhance(s4)  #weighted_x5 [bs, 512, 14, 14]
        feataug = self.cbam(weighted_x5)
        preds = self.decoder(feataug, s4, s3, s2, s1, H, W)       

        if self.training:
            gt = F.interpolate(gt, size=weighted_x5.size()[2:], mode='bilinear', align_corners=False)
            feat_pos, proto_pos, weighted_x5_pos, cormap_pos = self.enhance(s4 * gt)
            feat_neg, proto_neg, weighted_x5_neg, cormap_neg = self.enhance(s4*(1-gt))
            return preds, proto, proto_pos, proto_neg
        
        return preds


class YY(nn.Module):
    def __init__(self, mode=''):
        super(YY, self).__init__()
        set_seed(123)
        self.yynet = YYNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.yynet.set_mode(self.mode)

    def forward(self, x, gt):
        ########## Co-SOD ############
        preds = self.yynet(x, gt)
        return preds
    

    
