"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""




DATA_PATHS = {
    
    'videomatte': {
        'train': '/fast/stokesjv/BackgroundMatting3/datasets/VideoMatte240K_JPEG_SD/train',
        'valid': '/fast/stokesjv/BackgroundMatting3/datasets/VideoMatte240K_JPEG_SD/test',
    },
    'imagematte': {
        'train': '/fast/stokesjv/BackgroundMatting3/datasets/ImageMatte/train',
        'valid': '/fast/stokesjv/BackgroundMatting3/datasets/ImageMatte/valid',
    },
    'background_images': {
        'train': '/fast/stokesjv/BackgroundMatting3/datasets/Backgrounds/train',
        'valid': '/fast/stokesjv/BackgroundMatting3/datasets/Backgrounds/valid',
    },
    'background_videos': {
        'train': '/fast/stokesjv/BackgroundMatting3/datasets/BackgroundVideos/train',
        'valid': '/fast/stokesjv/BackgroundMatting3/datasets/BackgroundVideos/test',
    },
    
    
    'coco_panoptic': {
        'imgdir': '/fast/stokesjv/BackgroundMatting3/datasets/coco/train2017/',
        'anndir': '/fast/stokesjv/BackgroundMatting3/datasets/coco/panoptic_train2017/',
        'annfile': '/fast/stokesjv/BackgroundMatting3/datasets/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': '/fast/stokesjv/BackgroundMatting3/datasets/SuperviselyPersonDataset/img',
        'segdir': '/fast/stokesjv/BackgroundMatting3/datasets/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': '/fast/stokesjv/BackgroundMatting3/datasets/YouTubeVIS/train/JPEGImages',
        'annfile': '/fast/stokesjv/BackgroundMatting3/datasets/YouTubeVIS/train/instances.json',
    }
    
}

DATA_PATHS_PROJ = {
    
    'videomatte': {
        'train': '/projects/grail/stokesjv/datasets/VideoMatte240K_JPEG_SD/train',
        'valid': '/projects/grail/stokesjv/datasets/VideoMatte240K_JPEG_SD/test',
    },
    'imagematte': {
        'train': '/projects/grail/stokesjv/datasets/ImageMatte/train',
        'valid': '/projects/grail/stokesjv/datasets/ImageMatte/valid',
    },
    'background_images': {
        'train': '/projects/grail/stokesjv/datasets/Backgrounds/train',
        'valid': '/projects/grail/stokesjv/datasets/Backgrounds/valid',
    },
    'background_videos': {
        'train': '/projects/grail/stokesjv/datasets/BackgroundVideos/train',
        'valid': '/projects/grail/stokesjv/datasets/BackgroundVideos/test',
    },
    
    
    'coco_panoptic': {
        'imgdir': '/projects/grail/stokesjv/datasets/coco/train2017/',
        'anndir': '/projects/grail/stokesjv/datasets/coco/panoptic_train2017/',
        'annfile': '/projects/grail/stokesjv/datasets/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': '/projects/grail/stokesjv/datasets/SuperviselyPersonDataset/img',
        'segdir': '/projects/grail/stokesjv/datasets/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': '/projects/grail/stokesjv/datasets/YouTubeVIS/train/JPEGImages',
        'annfile': '/projects/grail/stokesjv/datasets/YouTubeVIS/train/instances.json',
    }
    
}
