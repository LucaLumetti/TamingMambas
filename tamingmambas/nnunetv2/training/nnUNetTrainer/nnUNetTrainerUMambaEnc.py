from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

from calflops import calculate_flops
import time


class nnUNetTrainerUMambaEnc(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug=debug)
        self.num_epochs = 300
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        print("CIAO SONO QUAAAAAAAA")
        input_shape = (2, 1, 147, 451, 451)
        total_params = sum(
            param.numel() for param in model.parameters()
        )
        print("UMambaBot Params: {}".format(total_params))
        flops, macs, params = calculate_flops(model=model,
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4)
        print("UMbambaBot FLOPs:{}   MACs:{}   Params:{}".format(flops, macs, params))
        time.sleep(60)
        print("CIAO SONO QUAAAAAAAA")

        return model
