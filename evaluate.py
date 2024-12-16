"""
Runs a quick eval script over validation set in order to compute loss

Change lines from saved loss to compute L1 / L2 / Total loss

"""


import torch
import imp

from model.swin_transformer3d import SwinTransformer3D
from model.swin_transformer import SwinTransformer
from model.model import MattingNetwork
from train_config import DATA_PATHS, DATA_PATHS_PROJ
from train_loss import matting_loss, segmentation_loss
from dataset.videomatte import (
    VideoMatteDataset, 
    VideoMatteTrainAugmentation,
    VideoMatteValidAugmentation
)
from dataset.augmentation import (
    TrainFrameSampler,
    ValidFrameSampler
)
from torchvision.transforms.functional import center_crop
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from tqdm import tqdm



def evaluate_model(variant, checkpoint):

    model = MattingNetwork(variant)
    model.load_state_dict(torch.load(checkpoint))
    device = 'cuda'

    model.to(device)

    data = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=None,
                size=512,
                seq_length=16,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation((512, 512)))

    # data = VMImageDataset(videomatte_dir=DATA_PATHS['videomatte']['train'],
    #                 background_image_dir=DATA_PATHS['background_images']['train'],
    #                 size=512,
    #                 transform=VideoMatteStaticAugmentation((512,512)))


    dataloader_valid = DataLoader(data, batch_size=1, shuffle=True)

    model.eval()
    total_loss, alpha_loss, total_count = 0, 0, 0
    with torch.no_grad():
        with autocast():
            for true_fgr, true_pha, true_bgr in tqdm(dataloader_valid):
                true_fgr = true_fgr.to(device, non_blocking=True)
                true_pha = true_pha.to(device, non_blocking=True)
                true_bgr = true_bgr.to(device, non_blocking=True)
                true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                batch_size = true_src.size(0)
                pred_fgr, pred_pha = model(true_src)[:2]
                total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                alpha_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['pha_l1'].item() * batch_size
                total_count += batch_size
    avg_loss = total_loss / total_count
    avg_alpha_loss = alpha_loss / total_count
    print(f'Validation set average loss: {avg_alpha_loss}')

if __name__ == '__main__':
    evaluate_model('resnet50', '/projects/grail/stokesjv/RobustVideoMatting/checkpoint/hyak/resnet/stage03/epoch-38.pth')