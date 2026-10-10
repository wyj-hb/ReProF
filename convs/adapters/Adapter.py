'''
Description: Adapter for all models
Author: wyj
version: 
Date: 2025-08-11 11:54:33
LastEditors: Please set LastEditors
LastEditTime: 2025-08-11 11:56:31
'''
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import ACT2FN
from utils.toolkit import AdapterConfig
import logging
logging.basicConfig(level=logging.INFO)

class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected