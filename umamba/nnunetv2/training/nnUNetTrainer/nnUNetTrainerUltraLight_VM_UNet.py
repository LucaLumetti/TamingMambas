from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.UltraLight_VM_UNet_2d import get_ultralight_vm_unet_2d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainerUltraLight_VM_UNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr = 0.001
        self.weight_decay = 1e-2

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_ultralight_vm_unet_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 2D models are supported")
        
        print("UltraLight_VM_UNet: {}".format(model))

        return model
    
    def configure_optimizers(self):
        # 'AdamW':
        # lr = 0.001 # default: 1e-3 – learning rate
        # betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        # eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        # weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        # amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad = False, eps = 1e-8, betas = (0.9, 0.999))
        
        # sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.
        lr_scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        return optimizer, lr_scheduler
