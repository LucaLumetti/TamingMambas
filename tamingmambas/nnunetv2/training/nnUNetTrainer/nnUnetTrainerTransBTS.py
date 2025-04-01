from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.TransBTS_utils.TransBTS_downsample8x_skipconnection import get_TransBTS_3d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

import numpy as np


class nnUNetTrainerTransBTS(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug: bool = False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr = 0.0002
        self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_TransBTS_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 3D models are supported")
        
        print("TransBTS: {}".format(model))

        return model
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    # def _build_loss(self):
    #     if self.label_manager.has_regions:
    #         loss = DC_and_BCE_loss({},
    #                                {'batch_dice': self.configuration_manager.batch_dice,
    #                                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
    #                                use_ignore_label=self.label_manager.ignore_label is not None,
    #                                dice_class=MemoryEfficientSoftDiceLoss, weight_ce=0.4, weight_dice=0.6)
    #     else:
    #         loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
    #                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.4, weight_dice=0.6,
    #                               ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss

    #     if self.enable_deep_supervision:
    #         deep_supervision_scales = self._get_deep_supervision_scales()
    #         weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
    #         weights[-1] = 0

    #         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    #         weights = weights / weights.sum()
    #         # now wrap the loss
    #         loss = DeepSupervisionWrapper(loss, weights)
    #     return loss    
