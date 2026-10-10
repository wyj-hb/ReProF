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
from utils.inc_net import SimpleRobertaNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size=16
g = torch.Generator()
g.manual_seed(42)
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.name=args['convnet_type']
        self._network = SimpleRobertaNet(args, True)
        self.use_act = args["use_act"]
        self.args=args
    def after_task(self):
        self._known_classes = self._total_classes
    def replace_fc(self,  trainloader, dev_loader, model, args):
        model = model.eval()
        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
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
                        "labels": batch[3]
                    }
                    train_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2]
                    }
                    train_label_list.append(batch[2].cpu())
                embedding = model.convnet(**inputs)
                if(self.name == "Mistral-7B-Instruct-v0.3"):
                    embedding = embedding.last_hidden_state
                    # 在第二个维度取平均
                    embedding = embedding.mean(dim=1)
                train_Features_f.append(embedding.cpu())
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
                        "labels": batch[3]
                    }
                    dev_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2]  # 对于 LLaMA 没有 token_type_ids
                    }
                    dev_label_list.append(batch[2].cpu())
                embedding = model.convnet(**inputs)
                if(self.name == "Mistral-7B-Instruct-v0.3"):
                    # 在第二个维度取平均
                    embedding = embedding.last_hidden_state
                    # 在第二个维度取平均
                    embedding = embedding.mean(dim=1)
                dev_Features_f.append(embedding.cpu())
        dev_Features_f = torch.cat(dev_Features_f, dim=0)
        dev_label_list = torch.cat(dev_label_list, dim=0)
        dev_Y=target2onehot(dev_label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                if self.use_act:
                    train_Features_h=torch.nn.functional.relu(train_Features_f@ self._network.fc.W_rand.cpu())
                    dev_Features_h=torch.nn.functional.relu(dev_Features_f@ self._network.fc.W_rand.cpu())
                else:
                    train_Features_h=(train_Features_f@ self._network.fc.W_rand.cpu())
                    dev_Features_h=(dev_Features_f@ self._network.fc.W_rand.cpu())
            elif self.use_act:
                train_Features_h=torch.nn.functional.relu(train_Features_f)
                dev_Features_h=torch.nn.functional.relu(dev_Features_f)
            else:
                train_Features_h=train_Features_f
                dev_Features_h=dev_Features_f
            self.Q=self.Q+train_Features_h.T @ train_Y 
            self.G=self.G+train_Features_h.T @ train_Features_h
            ridge=self.optimise_ridge_parameter(train_Features_h, train_Y, dev_Features_h, dev_Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(train_label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=train_Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(self._device) #for dil, we update all classes in all tasks
                else:
                    #original cosine similarity approach of Zhou et al (2023)
                    class_prototype=train_Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype #for cil, only new classes get updated
        num_parameters = 0
        # Gram矩阵和Q矩阵的参数量
        if self.args['use_RP']:
            num_parameters += self.G.numel()  # G矩阵的参数量
            num_parameters += self.Q.numel()  # Q矩阵的参数量
            if self.args['M'] > 0:
                num_parameters += self._network.fc.W_rand.numel()  # W_rand的参数量
        # 分类头权重的参数量
        num_parameters += self._network.fc.weight.numel()  # 分类头权重的参数量
        # 打印参数量，单位为 M (Million)
        logging.info(f"Parameter count after task {self._cur_task}: {num_parameters / (10 ** 6):.2f} M")

    def optimise_ridge_parameter(self,train_Features, train_Y, dev_Features, dev_Y):
        ridges=10.0**np.arange(-1,6)
        # num_val_samples=int(Features.shape[0]*0.8)
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
        if self._cur_task == 0 and self.args['use_RP']:
            self.setup_RP()
        self.replace_fc( train_loader_for_protonet, dev_loader, self._network, None)


    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)) #num classes in task x M
            self._network.fc.reset_parameters()
            # self._network.fc.W_rand = torch.load("W_rand.pt", map_location='cuda')
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(self._device)
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            M=self._network.fc.in_features 
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

   