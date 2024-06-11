from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.MambaUNet import get_mambaunet_2d_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainerMambaUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.enable_deep_supervision = False
        self.initial_lr = 5e-4
        self.weight_decay = 0.05

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_mambaunet_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=False)
        else:
            raise NotImplementedError("Only 2D models are supported")
        
        print("MambaUNet: {}".format(model))

        return model
    

    # # -----------------------------------------------------------------------------
    # # Training settings
    # # -----------------------------------------------------------------------------
    # _C.TRAIN = CN()
    # _C.TRAIN.START_EPOCH = 0
    # _C.TRAIN.EPOCHS = 300
    # _C.TRAIN.WARMUP_EPOCHS = 20
    # _C.TRAIN.WEIGHT_DECAY = 0.05
    # _C.TRAIN.BASE_LR = 5e-4
    # _C.TRAIN.WARMUP_LR = 5e-7
    # _C.TRAIN.MIN_LR = 5e-6
    # # Clip gradient norm
    # _C.TRAIN.CLIP_GRAD = 5.0
    # # Auto resume from latest checkpoint
    # _C.TRAIN.AUTO_RESUME = True
    # # Gradient accumulation steps
    # # could be overwritten by command line argument
    # _C.TRAIN.ACCUMULATION_STEPS = 0
    # # Whether to use gradient checkpointing to save memory
    # # could be overwritten by command line argument
    # _C.TRAIN.USE_CHECKPOINT = False

    # # LR scheduler
    # _C.TRAIN.LR_SCHEDULER = CN()
    # _C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
    # # Epoch interval to decay LR, used in StepLRScheduler
    # _C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
    # # LR decay rate, used in StepLRScheduler
    # _C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

    # # Optimizer
    # _C.TRAIN.OPTIMIZER = CN()
    # _C.TRAIN.OPTIMIZER.NAME = 'adamw'
    # # Optimizer Epsilon
    # _C.TRAIN.OPTIMIZER.EPS = 1e-8
    # # Optimizer Betas
    # _C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
    # # SGD momentum
    # _C.TRAIN.OPTIMIZER.MOMENTUM = 0.9    
    def configure_optimizers(self):
        # 'AdamW':
        # lr = 0.001 # default: 1e-3 – learning rate
        # betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        # eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        # weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        # amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
        optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=0.0001)
        # optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad = False, eps = 1e-8, betas = (0.9, 0.999))
        
        kwargs = {
                    'milestones':  [10000],
                    'gamma': 1,
                    }
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        return optimizer, lr_scheduler
