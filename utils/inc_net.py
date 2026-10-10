import copy
import logging
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from convs.linears import CosineLinear
from convs.simple_roberta import RobertaForSimple
from convs.simple_llama import LlamaForSimple
from convs.simple_qwen2 import Qwen2Forsimple

# ------------------------------------------------------------------ #
#  Model Registry — 5 supported models
# ------------------------------------------------------------------ #
MODEL_REGISTRY = {
    "roberta-large":           ("/home/lihr/model/roberta-large",          RobertaForSimple, 1024),
    "qwen2.5-0.5b":            ("../../model/Qwen2.5-0.5B-Instruct",       Qwen2Forsimple,   896),
    "mistral-7b-instruct-v0.3":("../../model/Mistral-7B-Instruct-v0.3",    None,             4096),
    "llama-3.2-1b":            ("../../model/Llama-3.2-1B",                LlamaForSimple,   2048),
    "llama-3-8b-instruct":     ("../../model/Llama-3-8B-Instruct",         LlamaForSimple,   4096),
}


def get_convnet(args, pretrained=False):
    if args['model_name'] == "ptm":
        config = AutoConfig.from_pretrained(
            "../../model/Llama-3.2-1B", low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = LlamaForSimple.from_pretrained(
            "../../model/Llama-3.2-1B", ignore_mismatched_sizes=True, config=config)
        model.out_dim = 2048
        return model.eval()

    name = args["convnet_type"].lower()
    if name not in MODEL_REGISTRY:
        raise NotImplementedError("Unknown convnet type: {}".format(name))

    path, model_class, out_dim = MODEL_REGISTRY[name]

    if name == "mistral-7b-instruct-v0.3":
        model = AutoModel.from_pretrained(path)
    elif name.startswith("llama"):
        config = AutoConfig.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = model_class.from_pretrained(path, config=config)
    else:
        config = AutoConfig.from_pretrained(path)
        model = model_class.from_pretrained(path, config=config)

    model.out_dim = out_dim
    return model.eval()


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self._device = args["device"][0]
        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, inputs):
        x = self.convnet(**inputs)
        out = self.fc(x["features"])
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class SimpleTwoLayer(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.k = args['k']
        self.name = args["convnet_type"].lower()

    def update_group(self, group_Num):
        self.fc_num = group_Num
        self.fc_list = [None for _ in range(self.fc_num)]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        self.fc_list = [None for _ in range(self.fc_num)]
        for i in range(self.fc_num):
            fc = self.generate_fc(self.feature_dim, nb_classes).cuda(self._device)
            self.fc_list[i] = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, inputs):
        return self.convnet(**inputs)

    def forward(self, inputs):
        x = self.convnet(**inputs)
        if self.name == "mistral-7b-instruct-v0.3":
            x = tuple(h.mean(dim=1) for h in x.hidden_states)
        x = torch.stack(x[-self.k:], dim=0).mean(dim=0)
        out = None
        for i in range(self.fc_num):
            out_fc = self.fc_list[i](x)
            if out is None:
                out = out_fc
            else:
                out['logits'] += out_fc['logits']
        return out


class SimpleMergeNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.name = args["convnet_type"].lower()

    def update_group(self, groups, group_Num):
        self.fc_num = group_Num
        self.fc_list = [None for _ in range(self.fc_num)]
        self.groups = groups

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        self.fc_list = [None for _ in range(self.fc_num)]
        for i in range(self.fc_num):
            fc = self.generate_fc(self.feature_dim, nb_classes).cuda(self._device)
            self.fc_list[i] = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, inputs):
        return self.convnet(**inputs)

    def forward(self, inputs):
        x = self.convnet(**inputs)
        if self.name == "mistral-7b-instruct-v0.3":
            x = tuple(h.mean(dim=1) for h in x.hidden_states)
        out = None
        for i in range(self.fc_num):
            indices = self.groups[i]
            selected_embeddings = [x[idx] for idx in indices]
            group_mean = torch.stack(selected_embeddings, dim=0).mean(dim=0)
            out_fc = self.fc_list[i](group_mean)
            if out is None:
                out = out_fc
            else:
                out['logits'] += out_fc['logits']
        return out


class SimpleAverageNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.k = args["k"]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        self.fc = self.generate_fc(self.feature_dim, nb_classes).cuda(self._device)

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, inputs):
        return self.convnet(**inputs)

    def forward(self, inputs):
        x = self.convnet(**inputs)
        x = torch.stack(x[-self.k:], dim=0).mean(dim=0)
        return self.fc(x)


class SimpleLayerUpNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.k = args["k"]
        self.name = args["convnet_type"]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        self.fc = self.generate_fc(self.feature_dim * self.k, nb_classes).cuda(self._device)

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, inputs):
        return self.convnet(**inputs)

    def forward(self, inputs):
        x = self.convnet(**inputs)
        if self.name == "Mistral-7B-Instruct-v0.3":
            x = tuple(h.mean(dim=1) for h in x.hidden_states)
        x = torch.cat(x[-self.k:], dim=1)
        return self.fc(x)


class SimpleRobertaNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.name = args["convnet_type"]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        self.fc = self.generate_fc(self.feature_dim, nb_classes).cuda(self._device)

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, inputs):
        return self.convnet(**inputs)

    def forward(self, inputs):
        x = self.convnet(**inputs)
        if self.name == "Mistral-7B-Instruct-v0.3":
            x = x.last_hidden_state.mean(dim=1)
        return self.fc(x)
