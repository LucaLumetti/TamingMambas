import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He

class MC(nn.Module):
    def __init__(self, channels, reduction_ratio = 16):
        super(MC, self).__init__()
        self.channels = channels
        assert channels%reduction_ratio == 0

        self.reduction_ratio = reduction_ratio        
        self.bottleneck = channels//reduction_ratio
        self.mlp_shared = nn.Sequential(nn.Linear(channels, self.bottleneck),
                                    nn.ReLU(),
                                    nn.Linear(self.bottleneck, channels))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, f):
        f_sx = self.sigmoid(self.mlp_shared(torch.mean(f, dim=(2,3))))
        f_dx = self.mlp_shared(torch.amax(f, dim=(2,3)))
        return f_sx + f_dx

class MS(nn.Module):
    def __init__(self, channels, kernel=7):
        super(MS, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.conv = nn.Conv2d(2, 1, kernel, padding=3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, f):
        f_avg = torch.mean(f, dim=1, keepdim=True)
        f_max, _ = torch.max(f, dim=1, keepdim=True)
        f_cat = torch.cat([f_avg, f_max], dim=1)
        f_conv = self.conv(f_cat)
        out = self.sigmoid(f_conv)
        return out

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio = 16, kernel=7):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel = kernel

        self.mc = MC(channels=channels, reduction_ratio=reduction_ratio,)
        self.ms = MS(channels=channels, kernel=kernel)
    def forward(self, f):
        f1 = f*self.mc(f)
        f2 = f1*self.ms(f1)
        return f2

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class EncMSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncMSBlock, self).__init__()
        self.batchnorm_1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel=7),
                                             nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel=5),
                                             nn.BatchNorm2d(out_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel=3),
                                             nn.BatchNorm2d(out_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel=3),
                                             nn.BatchNorm2d(out_channels))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel=3),
                                             nn.BatchNorm2d(out_channels))

        self.batchnorm_2 = nn.BatchNorm2d(out_channels)     
        self.CBAM(out_channels)  

        self.linear_1 = nn.Linear(out_channels, out_channels)     
        self.gelu = nn.GeLU()
        self.GRN = GRN(out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)                                  

    def forward(self, x):
        b_1 = self.batchnorm_1(x)
        
        conv1 = self.conv1(b_1)
        res1 = conv1 + b_1
        
        conv2 = self.conv2(res1)
        res2 = conv2 + res1

        conv3 = self.conv3(res2)
        res3 = conv3 + res2

        conv4 = self.conv4(res3)
        res4 = conv4 + res3

        conv5 = self.conv5(res4)
        res5 = conv5 + res4
        
        b_2 = self.batchnorm_2(res5)
        cbam = self.CBAM(b_2)
        # permute
        p_1 = torch.einsum('nchw->nhwc', cbam)
        l_1 = self.linear_1(p_1)
        gelu = self.gelu(b_2)
        GRN = self.GRN(gelu) # check dim of GRN
        l_2 = self.linear_2(GRN)
        # permute
        p_2 = torch.einsum('nchw->nhwc', l_2)
        out = p_2 + x

        return out

class ResUDM(nn.Module):
    def __init__(self, ):
        super(ResUDM, self).__init__()
        self.EncMSBlock1 = EncMSBlock()
        self.EncMSBlock2 = EncMSBlock()
    def forward(self, x):
        f_1 = self.EncMSBlock1(x)
        f_2 = self.EncMSBlock2(f_1)
        out = x + f_2
        return out


class UDMBlock(nn.Module):
    def __init__(self, ):
        super(UDMBlock, self).__init__()
        self.ResUDM = ResUDM()
        self.MambaLayer = None
        self.MaxPooling = None
    def forward(self, x):
        f_1 = self.ResUDM(x)
        f_2 = self.MambaLayer(f_1)
        out = self.MaxPooling(f_2)
        return out        

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x)
            out += global_att
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim) # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        
    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print('x_norm.dtype', x_norm.dtype)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class Attentionlayer(nn.Module):
    def __init__(self,dim,r=16,act='relu'):
        super(Attentionlayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim//r))
        self.layer2 = nn.Linear(int(dim//r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp):
        att = self.sigmoid(self.layer2(self.relu(self.layer1(inp))))
        return att.unsqueeze(-1)


class MambaBTS(nn.Module):
    def __init__(self,
                 input_channels=1,
                 channels=32,
                 blocks=3,
                 output_channels=6,
                 deep_supervision=True
                 ):
        super(MambaBTS, self).__init__()
        self.do_ds = deep_supervision
        self.in_conv = DoubleConv(input_channels, channels, stride=2, kernel_size=3)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.att1 = Attentionlayer(channels)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2, mamba_layer=MambaLayer(channels*2))

        self.att2 = Attentionlayer(channels*2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, mamba_layer=MambaLayer(channels*4))

        self.att3 = Attentionlayer(channels*4)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, mamba_layer=MambaLayer(channels*8))

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, output_channels)

        self.ds1_cls_conv = nn.Conv3d(32, output_channels, kernel_size=1)
        self.ds2_cls_conv = nn.Conv3d(64, output_channels, kernel_size=1)
        self.ds3_cls_conv = nn.Conv3d(128, output_channels, kernel_size=1)


    def forward(self, x):
        print(f'{x.shape=}')
        if x.shape[2] == 14: # Fix for ACDC which has 14 in the first spatial dimension (and we want 16)
            x = F.pad(x, (1,1,0,0,0,0), value=0)

        c1 = self.in_conv(x)
        scale_f1 = self.att1(self.pooling(c1).reshape(c1.shape[0], c1.shape[1])).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(self.pooling(c2).reshape(c2.shape[0], c2.shape[1])).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)

        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(self.pooling(c3).reshape(c3.shape[0], c3.shape[1])).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        print(f'{c1.shape=}')
        print(f'{c2.shape=}')
        print(f'{c3.shape=}')
        print(f'{c4.shape=}')

        up_5 = self.up5(c4)
        print(f'{up_5.shape=}')
        merge5 = torch.cat([up_5, c3*scale_f3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        print(f'{c5.shape=}')
        print(f'{up_6.shape=}')
        merge6 = torch.cat([up_6, c2*scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        print(f'{c6.shape=}')
        print(f'{up_7.shape=}')
        merge7 = torch.cat([up_7, c1*scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        print(f'{c7.shape=}')
        print(f'{up_8.shape=}')
        c8 = self.conv8(up_8)

        print(f'{c8.shape=}')

        logits = []
        logits.append(c8)
        logits.append(self.ds1_cls_conv(c7))
        logits.append(self.ds2_cls_conv(c6))
        logits.append(self.ds3_cls_conv(c5))

        if self.do_ds:
            return logits
        else:
            return logits[0]

def get_mambabts_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """

    label_manager = plans_manager.get_label_manager(dataset_json)

    network_class = MambaBTS

    model = network_class(
        num_input_channels, label_manager.num_segmentation_heads, 
        "TINY", configuration_manager.patch_size, 
        0.9, 0.1
    )
    # model.apply(InitWeights_He(1e-2))

    return model          

if __name__ == "__main__":
    B, C, H, W = 2, 16, 8, 8
    x = torch.ones((B, C, H, W))
    print(f"x: {x}\n{x.shape}\n\n")
    mc = MC(C)
    ms = MS(C)
    f_mc = mc(x)
    f_ms = ms(x)
    print(f"f_mc: {f_mc}\n{f_mc.shape}\n\n")
    print(f"f_ms: {f_ms}\n{f_ms.shape}\n\n")
    cbam = CBAM(C)
    cbam_out = cbam(x)
    print(f"cbam_out: {cbam_out}\n{cbam_out.shape}\n\n")
