from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdam import nnUNetTrainerVanillaRAdam3en4
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.SegMambaSkip import get_segmambaskip_from_plans

from calflops import calculate_flops
import time


class nnUNetTrainerSegMambaSkip(nnUNetTrainerVanillaRAdam3en4):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_segmambaskip_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)

        print("UMambaEnc: {}".format(model))

        print("CIAO SONO QUAAAAAAAA")
        input_shape = (2, 1, 147, 451, 451)
        total_params = sum(
            param.numel() for param in model.parameters()
        )
        print("SegmambaSkip Params: {}".format(total_params))
        flops, macs, params = calculate_flops(model=model,
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4)
        print("SegMambaSkip FLOPs:{}   MACs:{}   Params:{}".format(flops, macs, params))
        time.sleep(60)
        print("CIAO SONO QUAAAAAAAA")
        
        print("SegMambaSkip: {}".format(model))

        return model
    
