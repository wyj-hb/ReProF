import logging
import numpy as np
import torch
import copy
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleTwoLayer
from models.base import BaseLearner
import time
from utils.toolkit import target2onehot, tensor2numpy
num_workers = 8
batch_size=16
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.k = 16
        args['k'] = self.k
        self.args = args
        # TODO 根据模型类型判断大小
        name = args["convnet_type"].lower()
        self.name = name
        if name=="llama-3.2-1b":
            self.layers_num = 17
            self.out_dim =2048
        elif name=="llama-3.2-1b-instruct":
            self.layers_num = 17
            self.out_dim=2048
        elif name=="llama-3-8b-instruct":
            self.layers_num = 33
            self.out_dim=4096
        elif name=="qwen2.5-7b-instruct":
            self.layers_num = 29
            self.out_dim=3584
        elif name=="mistral-7b-instruct-v0.3":
            self.layers_num = 33
            self.out_dim=4096
        # # TODO 构建W_rand_list
        # self.group_Num = args['group_Num']
        self.W_rand_list = []
        for _ in range(args['group_Num']):
            self.W_rand_list.append(torch.randn(self.out_dim,args['M']).to(self._device))
        self._network = SimpleTwoLayer(args,True)
        self._network.k = self.k
        self.group_Num = args["group_Num"]
    def after_task(self):
        self._known_classes = self._total_classes
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_true = []
        for step, batch in enumerate(loader):
            batch = tuple(t.to(self._device) for t in batch)
            if len(batch) == 4:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    # "labels": batch[3],
                    "output_hidden_states": True,
                }
            elif len(batch) == 3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # "labels": batch[2],
                    "output_hidden_states": True,
                }
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(batch[2].cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    def replace_fc(self,  trainloader, dev_loader, model, args):
        model = model.eval()
        for f in range(self.group_Num):
            if self.args['use_RP']:
                self._network.fc_list[f].use_RP=True
                if self.args['M']>0:
                    self._network.fc_list[f].W_rand=self.W_rand_list[f]
        train_label_list = []
        train_groups_features = [[] for _ in range(self.group_Num)]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                batch = tuple(t.cuda(self._device) for t in batch)
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "output_hidden_states": True,
                    }
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                embedding = torch.stack(embedding[-self.k:], dim=0).mean(dim=0).cpu()
                train_groups_features[0].append(embedding)
                train_groups_features[1].append(embedding)
                # # TODO 根据group 和 group_num从embedding中提取数据
                # # TODO labels
                train_label_list.append(batch[2].cpu())
        train_label_list = torch.cat(train_label_list, dim=0)
        train_Y=target2onehot(train_label_list,self.total_classnum)
        dev_groups_features = [[] for _ in range(self.group_Num)]
        dev_label_list = []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = tuple(t.cuda(self._device) for t in batch)
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "output_hidden_states": True,
                    }
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                embedding = torch.stack(embedding[-self.k:], dim=0).mean(dim=0).cpu()
                dev_groups_features[0].append(embedding)
                dev_groups_features[1].append(embedding)
                # # TODO 深层
                dev_label_list.append(batch[2].cpu())
        dev_label_list = torch.cat(dev_label_list, dim=0)
        dev_Y=target2onehot(dev_label_list,self.total_classnum)
        # TODO 
        for i in range(self.group_Num):
            train_Features_f = torch.cat(train_groups_features[i],dim = 0)
            dev_Features_f = torch.cat(dev_groups_features[i],dim = 0)
            # * W_rand
            train_Features_h=torch.nn.functional.relu(train_Features_f @ self._network.fc_list[i].W_rand.cpu())
            dev_Features_h=torch.nn.functional.relu(dev_Features_f @ self._network.fc_list[i].W_rand.cpu())
            self.Q_list[i]=self.Q_list[i]+train_Features_h.T @ train_Y 
            self.G_list[i]=self.G_list[i]+train_Features_h.T @ train_Features_h
            # TODO lambda
            ridge=self.optimise_ridge_parameter(train_Features_h, train_Y, dev_Features_h, dev_Y)
            # TODO 计算分类器
            Wo=torch.linalg.solve(self.G_list[i]+ridge*torch.eye(self.G_list[i].size(dim=0)),self.Q_list[i]).T #better nmerical stability than .inv
            self._network.fc_list[i].weight.data=Wo[0:self._network.fc_list[i].weight.shape[0],:].to(self._device)
            if i == self.group_Num -1:
                continue
            # * 准备一下阶段train_Y和dev_Y
            train_zeros_tensor = torch.zeros(
                train_Y.size(),  # 形状
                device=train_Y.device,  # 设备
                dtype=train_Y.dtype  # 数据类型
            )
            # TODO 初始预测值
            DS_train_pred = F.linear(train_Features_h.to(self._device),self._network.fc_list[i].weight).detach().cpu()
            train_zeros_tensor[:,0:self._total_classes] = DS_train_pred
            # TODO 初始残差
            DS_train_Y = train_Y - train_zeros_tensor
            # TODO CPL,将初始残差的前 0 : self._known_classes置为0,当前stage的数据是 self._known_classes : self._total_class
            DS_train_Y[:,0:self._known_classes] = 0
            train_Y = DS_train_Y
            
            # dev
            dev_zeros_tensor = torch.zeros(
                dev_Y.size(),  # 形状
                device=dev_Y.device,  # 设备
                dtype=dev_Y.dtype  # 数据类型
            )
            DS_dev_pred = F.linear(dev_Features_h.to(self._device),self._network.fc_list[i].weight).detach().cpu()
            dev_zeros_tensor[:,0:self._total_classes] = DS_dev_pred
            Dev_Y = dev_Y - dev_zeros_tensor
            # clear
            Dev_Y[:,0:self._known_classes] = 0
            dev_Y = Dev_Y
        
    def optimise_ridge_parameter(self,train_Features, train_Y, dev_Features, dev_Y):
        ridges=10.0**np.arange(-1,6)
        losses=[]
        Q_val = train_Features.T @ train_Y 
        G_val = train_Features.T @ train_Features
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_dev_pred=dev_Features@Wo.T
            losses.append(F.mse_loss(Y_dev_pred,dev_Y))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_group(self.group_Num)
        self._network.update_fc(self._total_classes)
        if self._cur_task >= 0 :
            if self.args['use_RP']:
                for d in range(self.group_Num):
                    del self._network.fc_list[d]
                    self._network.fc_list.append(None)
            self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.data_manager=data_manager
        train_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="train",)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="dev",)
        self.dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = data_manager.get_texts_dataset(np.arange(0, self._total_classes), source="test",)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.train_loader_for_protonet = self.train_loader
        self._train(self.train_loader, self.test_loader, self.dev_loader, self.train_loader_for_protonet)
    def _train(self, train_loader, test_loader, dev_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.setup_RP()
        self.replace_fc( train_loader_for_protonet, dev_loader, self._network, None)
    def setup_RP(self):
        self.initiated_G=False
        self.Q_list = []
        self.G_list = []
        for i in range(self.group_Num):
            self._network.fc_list[i].use_RP=True
            if self.args['M']>0:
                M=self.args['M']
                self._network.fc_list[i].weight = nn.Parameter(torch.Tensor(self._network.fc_list[i].out_features, M).to(self._device)) #num classes in task x M
                self._network.fc_list[i].reset_parameters()
                self._network.fc_list[i].W_rand = self.W_rand_list[i]
            self.Q_list.append(torch.zeros(M,self.total_classnum))
            self.G_list.append(torch.zeros(M,M))
   