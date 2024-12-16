import imp
import torch

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
from torch.optim import Adam

from tqdm import tqdm

def testModel(decoder_var='recurrent'):
    #model = SwinUNet()
    model = MattingNetwork('swin', decoder_variant=decoder_var, pretrained_backbone=True, pretrain_img_size=512)
    #model = MattingNetwork('mobilenetv3')

    pp=0
    for name, param in model.state_dict().items():
        nn=1
        for s in list(param.size()):
            nn = nn*s
        pp += nn
    print('Parameters', pp)
    #print(model)

    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('params:', mem_params, 'buffs:', mem_bufs)
    print('memm', mem)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    LEARNING_RATE = 1e-4

    model = model.to(device)


    data = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=None,
                size=512,
                seq_length=5,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation((512, 512)))

    # data = VMImageDataset(videomatte_dir=DATA_PATHS['videomatte']['train'],
    #                 background_image_dir=DATA_PATHS['background_images']['train'],
    #                 size=512,
    #                 transform=VideoMatteStaticAugmentation((512,512)))


    dataloader_lr_train = DataLoader(data, batch_size=1, shuffle=True)

    #scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0

    for true_fgr, true_pha, true_bgr in tqdm(dataloader_lr_train, dynamic_ncols=True):
        true_fgr = true_fgr.to(device, non_blocking=True)
        true_pha = true_pha.to(device, non_blocking=True)
        true_bgr = true_bgr.to(device, non_blocking=True)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        true_src = true_src.to(device, non_blocking=True)


        pred_fgr, pred_pha = model(true_src)[:2]


        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        loss['total'].backward()
        optimizer.step()
        
        optimizer.zero_grad()

        if step % 100 == 0:
            print(loss)
        
        step += 1

def testSingleFrame():
    model = MattingNetwork('mobilenetv3' , pretrained_backbone=True)

    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('params:', mem_params, 'buffs:', mem_bufs)
    print('memm', mem)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    LEARNING_RATE = 1e-4

    model = model.to(device)


    data = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=None,
                size=512,
                seq_length=8,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation((512, 512)))

    # data = VMImageDataset(videomatte_dir=DATA_PATHS['videomatte']['train'],
    #                 background_image_dir=DATA_PATHS['background_images']['train'],
    #                 size=512,
    #                 transform=VideoMatteStaticAugmentation((512,512)))


    dataloader_lr_train = DataLoader(data, batch_size=1, shuffle=True)

    #scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0

    for true_fgr, true_pha, true_bgr in tqdm(dataloader_lr_train, dynamic_ncols=True):
        true_fgr = true_fgr.to(device, non_blocking=True)
        true_pha = true_pha.to(device, non_blocking=True)
        true_bgr = true_bgr.to(device, non_blocking=True)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        true_src = true_src.to(device, non_blocking=True)


        pred_fgr, pred_pha = model(true_src)[:2]


        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        loss['total'].backward()
        optimizer.step()
        
        optimizer.zero_grad()

        if step % 100 == 0:
            print(loss)
        
        step += 1


def testSwin3d(data_path):
    #model = SwinUNet()
    #model = MattingNetwork('swin', pretrained_backbone=True, pretrain_img_size=512)
    #model = MattingNetwork('mobilenetv3')
    # model = SwinTransformer3D(pretrained='checkpoint/backbone/swin_512_backbone.swin_512_backbone.pth')
    # model2 = SwinTransformer()
    model = MattingNetwork('swin3d', pretrained_backbone=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    LEARNING_RATE = 1e-4

    model = model.to(device)

    data = VideoMatteDataset(
                videomatte_dir=data_path['videomatte']['train'],
                background_image_dir=data_path['background_images']['train'],
                background_video_dir=None,
                size=448,
                seq_length=4,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation((448, 448)))

    # data = VMImageDataset(videomatte_dir=DATA_PATHS['videomatte']['train'],
    #                 background_image_dir=DATA_PATHS['background_images']['train'],
    #                 size=512,
    #                 transform=VideoMatteStaticAugmentation((512,512)))


    dataloader_lr_train = DataLoader(data, batch_size=1, shuffle=True)

    #scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0

    for true_fgr, true_pha, true_bgr in tqdm(dataloader_lr_train, dynamic_ncols=True):
        true_fgr = true_fgr.to(device, non_blocking=True)
        true_pha = true_pha.to(device, non_blocking=True)
        true_bgr = true_bgr.to(device, non_blocking=True)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        true_src = true_src.to(device, non_blocking=True)
        swin_src = true_src

        # true_src shape (1, 8, 3, 512, 512)
        
        #outputs = model(true_src)
        # out shape: [1, 768, 1, 16, 16] 


        pred_fgr, pred_pha = model(true_src)[:2]
        #swin_outputs = model2(swin_src)

        # 3d:
        # [1. 4. 96. 112. 112], [1, 4, 192, 56, 56] ... [1, 4, 768, 14, 14]
        #
        # swin shape:
        # [1, 4, 96, 112, 112], [1, 4, 192, 56, 56] ... [1, 4, 768, 14, 14]

        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        loss['total'].backward()
        optimizer.step()
        
        optimizer.zero_grad()

        if step % 100 == 0:
            print(loss)
        
        step += 1

def loadSwin3d(checkpoint):
    sd = torch.load(checkpoint)

    model = MattingNetwork('swin3d')

    print(model.load_state_dict(sd))


def testAttentionHead():
    model_bb = SwinTransformer()
    model_asp = LRASPP()
    model_attn = TMAHead()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    LEARNING_RATE = 1e-4

    model_bb = model_bb.to(device)
    model_asp = model_asp.to(device)
    model_attn = model_attn.to(device)

    data = VideoMatteDataset(
                videomatte_dir=data_path['videomatte']['train'],
                background_image_dir=data_path['background_images']['train'],
                background_video_dir=None,
                size=512,
                seq_length=4,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation((512, 512)))

    dataloader_lr_train = DataLoader(data, batch_size=1, shuffle=True)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0

    for true_fgr, true_pha, true_bgr in tqdm(dataloader_lr_train, dynamic_ncols=True):
        true_fgr = true_fgr.to(device, non_blocking=True)
        true_pha = true_pha.to(device, non_blocking=True)
        true_bgr = true_bgr.to(device, non_blocking=True)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        true_src = true_src.to(device, non_blocking=True)
        swin_src = true_src

        # true_src shape (1, 8, 3, 512, 512)
        
        #outputs = model(true_src)
        # out shape: [1, 768, 1, 16, 16] 

        outputs = model_bb(true_src)
        outputs[-1] = model_asp(outputs[-1])
        

        pred_fgr, pred_pha = model(true_src)[:2]
        # swin shape:
        # [1, 4, 96, 112, 112], [1, 4, 192, 56, 56] ... [1, 4, 768, 14, 14]

        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        loss['total'].backward()
        optimizer.step()
        
        optimizer.zero_grad()

        if step % 100 == 0:
            print(loss)
        
        step += 1

if __name__ == '__main__':
    
    print("beginning tests...")
    #testVMImageDataset()
    #testSwinTransformer()
    testModel('attention_past')
    #testSwin3d(DATA_PATHS)
    #loadSwin3d('./checkpoint/hyak/swin3d/stage3/epoch-39.pth')