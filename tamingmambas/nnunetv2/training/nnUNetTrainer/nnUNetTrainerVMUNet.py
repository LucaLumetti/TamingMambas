from torch.optim.lr_scheduler import CosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.VMUNet.vmunet import get_vmunet_from_plans
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

class nnUNetTrainerVMUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False

    def configure_optimizers(self):
        # hyperparameters obtained by https://github.com/JCruan519/VM-UNet/blob/main/configs/config_setting_synapse.py
        optimizer = AdamW(
            self.network.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False)

        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=0.00001,
            last_epoch=-1
        )
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            raise NotImplementedError("Only 3D models are supported")
        elif len(configuration_manager.patch_size) == 3:
            model = get_vmunet_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model

