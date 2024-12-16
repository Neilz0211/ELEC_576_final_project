import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import vgg19, VGG19_Weights

class GeneratorWithVGG(nn.Module):
    def __init__(self):
        super(GeneratorWithVGG, self).__init__()
        # Load pre-trained VGG19 model for feature extraction
        vgg_weights = VGG19_Weights.IMAGENET1K_V1
        vgg_features = vgg19(weights=vgg_weights).features[:10]
        self.fgr_features = nn.Sequential(*vgg_features)
        self.bgr_features = nn.Sequential(*vgg_features)

        # Freeze the VGG feature layers
        for param in self.fgr_features.parameters():
            param.requires_grad = False
        for param in self.bgr_features.parameters():
            param.requires_grad = False

        # Additional layers for processing combined features and output generation
        self.combined_features_to_image = nn.Sequential(
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, fgr, bgr):
        # Feature extraction for foreground and background
        fgr_features = self.fgr_features(fgr)
        bgr_features = self.bgr_features(bgr)

        # Combine features and generate the adapted background
        combined_features = torch.cat((fgr_features, bgr_features), dim=1)
        adapted_bgr = self.combined_features_to_image(combined_features)
        adapted_bgr = F.interpolate(adapted_bgr, size=(512, 512), mode='bilinear', align_corners=False)
        return adapted_bgr

class DiscriminatorWithEfficientNet(nn.Module):
    def __init__(self):
        super(DiscriminatorWithEfficientNet, self).__init__()
        # Initialize pre-trained EfficientNet-B0 models for foreground and background
        self.base_model_fgr = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.base_model_bgr = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        # Adjust the first convolution layer to accept 3-channel input
        self.adjust_first_conv_for_rgb()

        # Calculate output size for concatenated feature vectors
        dummy_input = torch.rand(1, 3, 512, 512)
        with torch.no_grad():
            output_size_fgr = self.base_model_fgr(dummy_input)[-1].nelement() // dummy_input.shape[0]
            output_size_bgr = self.base_model_bgr(dummy_input)[-1].nelement() // dummy_input.shape[0]
        total_output_size = output_size_fgr + output_size_bgr

        # Classifier for determining the authenticity of images
        self.classifier = nn.Sequential(
            nn.Linear(total_output_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def adjust_first_conv_for_rgb(self):
        for model in [self.base_model_fgr, self.base_model_bgr]:
            first_conv_layer = model.conv_stem
            model.conv_stem = nn.Conv2d(3, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size,
                                        stride=first_conv_layer.stride, padding=first_conv_layer.padding)

    def forward(self, fgr, bgr):
        # Feature extraction and combination
        fgr_features = self.base_model_fgr(fgr)[-1].view(fgr.shape[0], -1)
        bgr_features = self.base_model_bgr(bgr)[-1].view(bgr.shape[0], -1)
        combined_features = torch.cat([fgr_features, bgr_features], dim=1)

        # Classification of combined features
        output = self.classifier(combined_features)
        return output
