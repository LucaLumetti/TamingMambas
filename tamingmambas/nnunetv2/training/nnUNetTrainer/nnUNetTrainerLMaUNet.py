import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context

from nnunetv2.nets.LMaUNet import get_lmaunet_from_plans

class nnUNetTrainerLMaUNet(nnUNetTrainer):
    """
    MambaUNet Encoder + Residual Decoder + Skip Connections
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug=debug)
        self.iters_to_accumulate = 2 
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * self.iters_to_accumulate

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_lmaunet_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)
        
        print("LMaUNet: {}".format(model))

        return model

    def train_step(self, batch: dict, has_accumulated: bool = False) -> dict:
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
            l = l / self.iters_to_accumulate

        # # Accumulates scaled gradients.
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()            
        else:            
            l.backward()

        if has_accumulated is True:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:        
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)        
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {'loss': l.detach().cpu().numpy()}          
    
    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            has_accumulated = False
            for batch_id in range(self.num_iterations_per_epoch):
                if (batch_id + 1) % self.iters_to_accumulate == 0:
                    has_accumulated = True
                else:
                    has_accumulated = False
                train_outputs.append(self.train_step(next(self.dataloader_train), has_accumulated))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
