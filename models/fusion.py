import torch.nn as nn
import torch
from torch.nn import functional as F

from timm.models import create_model


# class FrozenVIT(nn.Module):
#     def __init__(self):
#         super(FrozenVIT, self).__init__()

#         # self.premodel = models.vgg16(pretrained=True)
#         # self.model = nn.Sequential(*list(self.premodel.children())[:-2]) 
#         # self.model =ViT('B_16_imagenet1k',pretrained=True,image_size =224)
#         # self.model = nn.Sequential(*list(self.premodel.children())[:-1])          ### -2
        
#         self.original_model = create_model(
#             'vit_base_patch16_224',
#             pretrained=True,
#             num_classes=1000,
#             drop_rate=0.0,
#             drop_path_rate=0.0,
#             drop_block_rate=None,
#         )
#         print ("FrozenVIT vit16 loaded")

#     def forward(self, x_input):
        
#         for p in self.original_model.parameters():
#             p.requires_grad = False
        
#         x = self.original_model.forward_features(x_input)       #####  25 512 7 7
#         x = x[:,0,:]
#         return x


class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        
        self.define_model = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
        print ("fusion_model vit16 loaded")
               
        self.out_chann = 512
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        #self.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
        
        self.acti_Tanh = nn.Tanh()
        self.acti_Sig = nn.Sigmoid()
               
#         self.input_frozen = FrozenVIT()
        
        self.pre_out = nn.Linear(768, 1)
#         self.pre_out_0 = nn.Linear(768, 196)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        
    def forward(self, RGBT, last_cont=None):
        RGB = RGBT[0]
        T_img = RGBT[1]
        # print ('T_img shape',T_img.shape)    ####  ([1, 3, 224, 224])
        B, _, H, W = RGB.shape
        # N = (H // 16) * (W // 16)
#         with torch.no_grad():
#             x_anchor = self.input_frozen(RGB)
        x_anchor = 0
#         with torch.no_grad():    
        out_fea = self.define_model.forward_features(T_img)
        token0_fea = out_fea[:, 0, :]
        out_fea1 = out_fea[:, 1:, :]
        out_pred = self.pre_out(out_fea1)
        out = out_pred.transpose(1, 2).reshape(B, 1, H // 16, W // 16)
#         out_pred = self.pre_out_0(token0_fea)
#         out = out_pred.unsqueeze(1).reshape(B, 1, H // 16, W // 16)
        out = self.up4(out)
        mu = self.relu(out)

        return  torch.abs(mu), token0_fea, x_anchor


# class FusionModel(nn.Module):
#     def __init__(self, ratio=0.6):
#         super(FusionModel, self).__init__()
#         c1 = int(64 * ratio)
#         c2 = int(128 * ratio)
#         c3 = int(256 * ratio)
#         c4 = int(512 * ratio)

#         self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
#         self.block2 = Block([c2, c2, 'M'], in_channels=c1)
#         self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
#         self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
#         self.block5 = Block([c4, c4, c4, c4], in_channels=c4)

#         self.reg_layer = nn.Sequential(
#             nn.Conv2d(c4, c3, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c3, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 1, 1)
#         )
#         self._initialize_weights()

#     def forward(self, RGBT):
#         RGB = RGBT[0]
#         T = RGBT[1]

#         RGB, T, shared = self.block1(RGB, T, None)
#         RGB, T, shared = self.block2(RGB, T, shared)
#         RGB, T, shared = self.block3(RGB, T, shared)
#         RGB, T, shared = self.block4(RGB, T, shared)
#         _, _, shared = self.block5(RGB, T, shared)
#         x = shared

#         x = F.upsample_bilinear(x, scale_factor=2)
#         x = self.reg_layer(x)
#         ##print ('x shape',x.shape)    ###  1 1 32 32
#         return torch.abs(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, RGB, T, shared):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)

        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
        rgb_fuse_gate = torch.sigmoid(rgb_s)
        t_s = self.t_fuse_1x1conv(T_m - shared_m)
        t_fuse_gate = torch.sigmoid(t_s)
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
        rgb_distribute_gate = torch.sigmoid(s_rgb)
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
        t_distribute_gate = torch.sigmoid(s_t)
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
        new_T = T + (new_shared_m - T_m) * t_distribute_gate

        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion


def fusion_model():
    model = FusionModel()
    return model


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
