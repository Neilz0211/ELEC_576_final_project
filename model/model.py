from operator import mod
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .swin_transformer import SwinTransformer
from .swin_transformer3d import SwinTransformer3D
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
#from .attention_bottleneck import AttenionBottleneck
from .attention_decoder import AttentionDecoder
from utils import load_checkpoint 

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 decoder_variant: str = 'recurrent',
                 pretrained_backbone: bool = False,
                 pretrain_img_size: int = 224):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50', 'swin', 'swin3d']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            decoder_params = ([16, 24, 40, 128], [80, 40, 32, 16], 'mobilenetv3')
        elif variant == 'swin':
            self.backbone = SwinTransformer(pretrain_img_size=pretrain_img_size)
            if pretrained_backbone: 
                load_checkpoint('checkpoint/upernet_swin_tiny_patch4_window7_512x512.pth', self.backbone)
            self.aspp = LRASPP(768, 128)
            decoder_params = ([96, 192, 384, 128], [128, 64, 32, 16], 'swin')
        elif variant == 'swin3d':
            if pretrained_backbone: 
                self.backbone = SwinTransformer3D(
                    pretrained='checkpoint/backbone/swin_512_backbone.swin_512_backbone.pth',
                    pretrained2d=True
                )
            else:
                self.backbone = SwinTransformer3D()
            #self.backbone.init_weights('../checkpoint/upernet_swin_tiny_patch4_window7_512x512.pth')
            self.aspp = LRASPP(768, 128)
            decoder_params = ([96, 192, 384, 128], [128, 64, 32, 16], 'swin')
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            decoder_params = ([64, 256, 512, 256], [128, 64, 32, 16], 'resnet')
        
        if decoder_variant == 'recurrent':
            self.decoder = RecurrentDecoder(*decoder_params)
        elif decoder_variant == 'attention':
            self.decoder = AttentionDecoder(*decoder_params)
        elif decoder_variant == 'attention_past':
            self.decoder = AttentionDecoder(*decoder_params, mask_future=True)
        
        output_dim = decoder_params[1][-1]    
        self.project_mat = Projection(output_dim, 4)
        self.project_seg = Projection(output_dim, 1)

        self.variant = variant
        self.decoder_variant = decoder_variant

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

        
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        # (B, T, 16, 256, 256) .. 24, 128 .. 40, 64 .. 960, 32
        # 24, 48, 96, 192
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)

        # (1, 8, 96, 128, 128) (1,8,192,64,64) 384 32, 128 16
        #f1, f2, f3, f4 = self.attention_bottleneck(f1, f2, f3, f4)

        if self.decoder_variant == 'recurrent':
            hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        else:
            hid = self.decoder(src_sm, f1, f2, f3, f4)
        # 1, 8, 16, 512, 512
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            result = [fgr, pha]
        else:
            seg = self.project_seg(hid)
            result = [seg]
        if self.decoder_variant == 'recurrent':
            return [*result, *rec] 
        else:
            return [*result]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
