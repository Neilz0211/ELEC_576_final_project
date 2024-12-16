"""
python3 inference.py \
    --variant mobilenetv3 \
    --checkpoint "checkpoint/stage1mv/epoch-19.pth" \
    --device cuda \
    --input-source "b5.mp4" \
    --output-type video \
    --output-composition "composition_ctrl.mp4" \
    --output-alpha "alpha_ctrl.mp4" \
    --output-foreground "foreground_ctrl.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1 \
    --input-resize 512 512
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from dataset.augmentation import ValidFrameSampler

from dataset.videomatte import VideoMatteDataset, VideoMatteTrainAugmentation

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

from train_config import DATA_PATHS

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    if input_source == 'demo':
        demo = True
        data = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=None,
                size=512,
                seq_length=80,
                seq_sampler=ValidFrameSampler(),
                transform=VideoMatteTrainAugmentation((512, 512)))

        source = None
        reader = DataLoader(data, batch_size=1, shuffle=True, num_workers=num_workers)
    
    else:
        demo=False
        # Initialize reader
        if os.path.isfile(input_source):
            source = VideoReader(input_source, transform)
        else:
            source = ImageSequenceReader(input_source, transform)
        reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 24
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=200, disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            if demo:
                video = next(iter(reader))
                src = video[0]
                #video = video[0].squeeze(dim=0)
                #reader = [x for _, x in enumerate(video)]

            
            src = src.to(device, dtype, non_blocking=True) # [B, T, C, H, W]
            fgr, pha, *rec = model(src, *rec, 1)

            if output_foreground is not None:
                writer_fgr.write(fgr[0])
            if output_alpha is not None:
                writer_pha.write(pha[0])
            if output_composition is not None:
                if output_type == 'video':
                    com = fgr * pha + bgr * (1 - pha)
                else:
                    fgr = fgr * pha.gt(0)
                    com = torch.cat([fgr, pha], dim=-3)
                writer_com.write(com[0])
            
            bar.update(1)


            

            # for src in reader:

            #     # if downsample_ratio is None:
            #     #     downsample_ratio = auto_downsample_ratio(*src.shape[2:])
            #     if demo:
            #         src = src.unsqueeze(0)

            #     src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
            #     fgr, pha, *rec = model(src, *rec, 1)

            #     if output_foreground is not None:
            #         writer_fgr.write(fgr[0])
            #     if output_alpha is not None:
            #         writer_pha.write(pha[0])
            #     if output_composition is not None:
            #         if output_type == 'video':
            #             com = fgr * pha + bgr * (1 - pha)
            #         else:
            #             fgr = fgr * pha.gt(0)
            #             com = torch.cat([fgr, pha], dim=-3)
            #         writer_com.write(com[0])
                
            #     bar.update(1)

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        #self.model = torch.jit.script(self.model)
        #self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50', 'swin', 'swin3d'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    
    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
    
    
