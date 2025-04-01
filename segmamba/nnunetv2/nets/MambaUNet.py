from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He

logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mamba_unet =  VSSM(
                                patch_size=config["PATCH_SIZE"],
                                in_chans=config["IN_CHANS"],
                                num_classes=self.num_classes,
                                embed_dim=config["EMBED_DIM"],
                                depths=config["DEPTHS"],
                                mlp_ratio=config["MLP_RATIO"],
                                drop_rate=config["DROP_RATE"],
                                drop_path_rate=config["DROP_PATH_RATE"],
                                patch_norm=config["PATCH_NORM"],
                                use_checkpoint=config["USE_CHECKPOINT"])

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

def get_mambaunet_2d_from_plans(
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
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'MambaUnet'
    network_class = MambaUnet
    kwargs = {
        'MambaUnet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # _C.MODEL.VSSM.PATCH_SIZE = 4
    # _C.MODEL.VSSM.IN_CHANS = 3
    # _C.MODEL.VSSM.EMBED_DIM = 96
    # _C.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
    # _C.MODEL.VSSM.MLP_RATIO = 4.
    # _C.MODEL.VSSM.PATCH_NORM = True
    config = {}
    config["PATCH_SIZE"] = 4
    config["IN_CHANS"] = num_input_channels
    config["EMBED_DIM"] = 96
    config["DEPTHS"] = [2, 2, 9, 2]
    config["MLP_RATIO"] = 4
    config["DROP_RATE"] = 0.0
    config["DROP_PATH_RATE"] = 0.1
    config["PATCH_NORM"] = True
    config["USE_CHECKPOINT"] = False
    config["PRETRAIN_CKPT"] = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
    model = network_class(
        config, img_size=224, num_classes=label_manager.num_segmentation_heads, zero_head=False, vis=False
    )
    # model.apply(InitWeights_He(1e-2))

    return model             