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
from utils.inc_net import SimpleAverageNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
num_workers = 8
batch_size=16
g = torch.Generator()
g.manual_seed(42)
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args['type'] == "all":
            self.k = 16
        elif args["dataset_name"] in ["books","poetry","recipes","reviews",]:
            self.k = 13
        elif args["dataset_name"] in ["reddit","wikipedia","news"]:
            self.k = 14
        args['k'] = self.k
        self._network = SimpleAverageNet(args, True)
        self.args = args
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
                    "labels": batch[3],
                    "output_hidden_states": True,
                }
            elif len(batch) == 3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                    "output_hidden_states": True,  # 对于 LLaMA 没有 token_type_ids
                }
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(inputs['labels'].cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,  trainloader, dev_loader, model, args):
        model = model.eval()
        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
        self._network.fc.W_rand=None
        train_Features_f = []
        train_label_list = []
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
                        "labels": batch[2],
                        "output_hidden_states": True,
                    }
                embedding=model.convnet(**inputs)
                # TODO 对embedding 取平均
                embedding = torch.stack(embedding[-self.k:], dim=0).mean(dim=0)
                train_Features_f.append(embedding.cpu())
                train_label_list.append(inputs['labels'].cpu())
        train_Features_f = torch.cat(train_Features_f, dim=0)
        train_label_list = torch.cat(train_label_list, dim=0)

        train_Y=target2onehot(train_label_list,self.total_classnum)

        dev_Features_f = []
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
                        "output_hidden_states": True
                    }
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "output_hidden_states": True,
                    }
                embedding=model.convnet(**inputs)
                embedding = torch.stack(embedding[-self.k:], dim=0).mean(dim=0)
                dev_Features_f.append(embedding.cpu())
                dev_label_list.append(inputs['labels'].cpu())
        dev_Features_f = torch.cat(dev_Features_f, dim=0)
        dev_label_list = torch.cat(dev_label_list, dim=0)

        dev_Y=target2onehot(dev_label_list,self.total_classnum)

        self.Q=self.Q+train_Features_f.T @ train_Y 
        self.G=self.G+train_Features_f.T @ train_Features_f
        ridge=self.optimise_ridge_parameter(train_Features_f, train_Y, dev_Features_f, dev_Y)
        Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
        self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device=self._device)

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
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.fc
            self._network.fc=None
        
        
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.data_manager=data_manager

        train_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="train",)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,generator=g)

        dev_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="dev",)
        self.dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True,generator=g)
        
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
        M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M ,self.total_classnum)
        self.G=torch.zeros(M ,M)

   