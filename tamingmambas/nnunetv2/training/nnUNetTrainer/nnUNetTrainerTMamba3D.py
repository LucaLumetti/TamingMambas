from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.TMamba3D import get_tmamba_3d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

    # "optimizer_name": "AdamW",
    # "learning_rate": 0.0005, # 0.0005
    # "weight_decay": 0.00005,
    # "momentum": 0.8,
    # # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    # "lr_scheduler_name": "ReduceLROnPlateau",
    # "gamma": 0.1,
    # "step_size": 9,
    # "milestones": [1, 3, 5, 7, 8, 9],
    # "T_max": 2,
    # "T_0": 2,
    # "T_mult": 2,
    # "mode": "max",
    # "patience": 1,
    # "factor": 0.5,
    # # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    # "metric_names": ["DSC"], # HD 
    # "loss_function_name": "DiceLoss",
    # "class_weight": [0.00551122, 0.99448878],
    # "sigmoid_normalization": False,
    # "dice_loss_mode": "extension",
    # "dice_mode": "standard",

class nnUNetTrainerTMamba3D(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr = 0.0005
        self.weight_decay = 0.00005
        self.momentum = 0.8

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_tmamba_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 3D models are supported")
        
        print("TMamba3D: {}".format(model))

        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad = False, eps = 1e-8, betas = (0.9, 0.999))    
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='max', 
                                                                    factor=0.5, 
                                                                    patience=1, 
                                                                    threshold=0.0001, 
                                                                    threshold_mode='rel', 
                                                                    cooldown=0, 
                                                                    min_lr=0, 
                                                                    eps=1e-08, 
                                                                    verbose='deprecated')
        return optimizer, lr_scheduler
