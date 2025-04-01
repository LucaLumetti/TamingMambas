from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.RS3Mamba import get_rs3mamba_d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainerRS3Mamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug: bool = False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr = 0.01
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.debug = debug

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_rs3mamba_d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 2D models are supported")
        
        print("RS3Mamba: {}".format(model))

        return model
    
    def configure_optimizers(self):
        params_dict = dict(self.network.named_parameters())
        params = []
        for key, value in params_dict.items():
            if '_D' in key:
                # Decoder weights are trained at the nominal learning rate
                params += [{'params':[value],'lr': self.initial_lr}]
            else:
                # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
                params += [{'params':[value],'lr': self.initial_lr / 2}]
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=self.momentum)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
        return optimizer, lr_scheduler
