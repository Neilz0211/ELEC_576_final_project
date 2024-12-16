import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class AttentionDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels, variant, mask_future=False):
        super().__init__()
        self.avgpool = AvgPool(variant)
        self.decode4 = BottleneckBlock(feature_channels[3], mask_future=mask_future)
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], mask_future=mask_future)
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1], mask_future=mask_future)
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2], mask_future=mask_future)
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3], variant)

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor):
        s1, s2, s3 = self.avgpool(s0)
        x4 = self.decode4(f4)
        x3 = self.decode3(x4, f3, s3)
        x2 = self.decode2(x3, f2, s2)
        x1 = self.decode1(x2, f1, s1)
        # x1: 1, 8, 32, 128, 128
        x0 = self.decode0(x1, s0)
        return x0
    

class AvgPool(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        self.variant = variant
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        s4 = self.avgpool(s3)
        if self.variant == 'swin':
            return s2, s3, s4
        else: return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels, mask_future=False):
        super().__init__()
        self.channels = channels
        #self.gru = ConvGRU(channels // 2)
        self.attn = SequenceAttention(channels // 2, num_heads=8, mask_future=mask_future)

    def forward(self, x):
        a, b = x.split(self.channels // 2, dim=-3)
        #b, r = self.gru(b, r)
        b = self.attn(b)
        x = torch.cat([a, b], dim=-3)
        return x

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, mask_future=False):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        #self.gru = ConvGRU(out_channels // 2)
        self.attn = SequenceAttention(dim=out_channels //2, num_heads=8, mask_future=mask_future)

    def forward_single_frame(self, x, f, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        #b, r = self.gru(b, r)
        b = self.attn(b)
        x = torch.cat([a, b], dim=1)
        return x
    
    def forward_time_series(self, x, f, s):
        B, T, _, H, W = s.shape
        # x shp 1st: 1,10,128,32,32
        x = x.flatten(0, 1)
        # f shp 1 10, 96, 64, 64
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b = self.attn(b)
        x = torch.cat([a, b], dim=2)
        return x
    
    def forward(self, x, f, s):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s)
        else:
            return self.forward_single_frame(x, f, s)

class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels, variant):
        super().__init__()
        if variant == 'swin':
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)



class SequenceAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        max_seq_length (int): Max distance the attention will look forward / backwards
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads=8, max_seq_length=64, mask_future=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #self.seq_length = max_seq_length
        self.mask_future = mask_future

        # define a parameter table of relative position bias
        # Note that this is shared across heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_seq_length - 1), num_heads))  # 2*T-1, nH

        positions = torch.arange(max_seq_length)
        #Get pairwise relative positions
        p1, p2 = torch.meshgrid(positions, positions)
        relative_position_index = p2 - p1 # T, T: [0 1...T, -1 0 ...T-1,,,-T...0]
        relative_position_index += max_seq_length -1

        # we keep the position indices as a buffer because they are fixed (don't want gradients)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (B, T, H, W, C)
            x: input features with shape of (num_windows*B, N, C)
                where B is batch_size, i.e. runs each window in each batch indep. N is the seq len, i.e. window area. C is embed dim
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, T, C, H, W = x.shape
        seq_length = T
        x = rearrange(x, 'b t c h w -> (b h w) t c')
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #3, B, nH, T, d
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q k v shape: 258, nH, T, d
        
        q = q * self.scale
        # after k.transpose(-2,-1): B nH d T
        attn = (q @ k.transpose(-2, -1))
        # attn shape: B nH T T


        ## IMPL decision: Should I add relative position indexing before or after computing attention?

        relative_indices = self.relative_position_index[:seq_length, :seq_length].contiguous().view(-1)
        relative_position_bias = self.relative_position_bias_table[relative_indices].view(
            seq_length, seq_length, -1)  # T, T,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, T, T
        
        # Add relative positional embedding
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.mask_future:
            mask_future = torch.ones((self.num_heads, seq_length, seq_length), device=x.device) * float('-inf')
            mask_future = torch.triu(mask_future, diagonal=1)
            attn = attn + mask_future.unsqueeze(0)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, ' (b h w) t c -> b t c h w', h=H, w=W)
        return x
        
    # def forward_single_frame(self, x):
    #     return NotImplementedError()

    # def forward_time_series(self, x):


    # def forward(self, x):
    #     if x.ndim == 5:
    #         return self.forward_time_series(x)
    #     else:
    #         return self.forward_single_frame(x)

class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    
