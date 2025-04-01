from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.TMamba2D import get_tmamba_2d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

    # "optimizer_name": "AdamW",
    # "learning_rate": 0.0075, # 0.005 0.0025
    # # "weight_decay": 0.00005,
    # # "momentum": 0.8,
    # "weight_decay": 0.000001,
    # "momentum": 0.9657205586290213,
    # # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    # "lr_scheduler_name": "MultiStepLR",
    # "gamma": 0.1,
    # "step_size": 1,
    # "milestones": [24, 28,], # [20, 26,]
    # "T_max": 2,
    # "T_0": 2,
    # "T_mult": 2,
    # "mode": "max",
    # "patience": 1,
    # "factor": 0.5,
    # # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    # "metric_names": ["DSC", "IoU", "JI", "ACC"],
    # "loss_function_name": "DiceLoss",
    # "class_weight": [0.029, 1-0.029],
    # "sigmoid_normalization": False,
    # "dice_loss_mode": "extension",
    # "dice_mode": "standard",

class nnUNetTrainerTMamba2D(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr =  0.0075
        self.weight_decay =  0.000001
        self.momentum = 0.9657205586290213

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_tmamba_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 2D models are supported")
        
        print("TMamba2D: {}".format(model))

        return model
     
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(  self.network.parameters(), 
                                        self.initial_lr, 
                                        weight_decay=self.weight_decay, 
                                        amsgrad = False, 
                                        eps = 1e-8, 
                                        betas = (0.9, 0.999))    
        kwargs = {
                    'milestones':  [24, 28],
                    'gamma': 0.1,
                    }
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        return optimizer, lr_scheduler
