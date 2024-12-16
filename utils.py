
import os
import sys
from tabnanny import check
import torch
import torch.distributed as dist
from collections import OrderedDict
from torch import nn
#from model.swin_transformer3d import SwinTransformer3D

from mmcv.cnn import ConvModule


class SequenceConv(nn.ModuleList):
    """Sequence conv module.
    Args:
        in_channels (int): input tensor channel.
        out_channels (int): output tensor channel.
        kernel_size (int): convolution kernel size.
        sequence_num (int): sequence length.
        conv_cfg (dict): convolution config dictionary.
        norm_cfg (dict): normalization config dictionary.
        act_cfg (dict): activation config dictionary.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sequence_num, conv_cfg, norm_cfg, act_cfg):
        super(SequenceConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence_num = sequence_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sequence_num):
            self.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    padding=self.kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )

    def forward(self, sequence_imgs):
        """
        Args:
            sequence_imgs (Tensor): TxBxCxHxW
        Returns:
            sequence conv output: TxBxCxHxW
        """
        sequence_outs = []
        assert sequence_imgs.shape[0] == self.sequence_num
        for i, sequence_conv in enumerate(self):
            sequence_out = sequence_conv(sequence_imgs[i, ...])
            sequence_out = sequence_out.unsqueeze(0)
            sequence_outs.append(sequence_out)

        sequence_outs = torch.cat(sequence_outs, dim=0)  # TxBxCxHxW
        return sequence_outs

def get_backbone_dict(checkpoint):
    state_dict = OrderedDict()
    sd = checkpoint
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']

    for k, v in sd.items():
        if k[:9] == 'backbone.':
            state_dict[k[9:]] = v

    #checkpoint['state_dict'] = state_dict
    return state_dict

def load_checkpoint(checkpoint_file, model, optimizer=None, lr_scheduler=None, logger=None):

    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = get_backbone_dict(checkpoint) 
    msg = model.load_state_dict(state_dict, strict=False)
    #logger.info(msg)
    print(msg)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    '''
    
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:        # If this is a swin checkpoint file, we can use it directly
        # If this is a full model checkpoint, need to extract just the backbone

    del checkpoint
    torch.cuda.empty_cache()
    #return max_accuracy
    '''

def load_checkpoint_3d(checkpoint_file, model, optimizer=None, lr_scheduler=None, logger=None):

    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = get_backbone_dict(checkpoint) 
    msg = model.load_state_dict(state_dict, strict=False)
    #logger.info(msg)
    print(msg)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

#load_checkpoint_3d('checkpoint/swin/stage02/epoch-38.pth', SwinTransformer3D())



def convert_checkpoint(checkpoint, save_name):
    checkpoint = torch.load(checkpoint)
    state_dict = get_backbone_dict(checkpoint)

    torch.save({ 'state_dict' : state_dict}, save_name)


if __name__ == '__main__':

    convert_checkpoint(sys.argv[1], sys.argv[2])