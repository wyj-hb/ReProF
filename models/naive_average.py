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
from utils.inc_net import SimpleMergeNet
from models.base import BaseLearner
import time
import random
from torch.linalg import svd
from utils.toolkit import target2onehot, tensor2numpy
num_workers = 8
batch_size=16
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # TODO 根据模型类型判断大小
        name = args["convnet_type"].lower()
        self.name = name
        self.type = args.get("type", "origin")
        self.method = args.get("method", "svcca")
        if name=="llama-3.2-1b":
            self.layers_num = 17
            # self.groups = [[16,15,14,13,12,11,10],[9,8,7,6,5,4,3],[2],[1]]
            self.out_dim =2048
        elif name=="llama-3.2-1b-instruct":
            self.layers_num = 17
            # self.groups = [[16,15,14,13,12,11,10],[9,8,7,6,5,4],[3,2,1]]
            self.out_dim=2048
        elif name=="llama-3-8b-instruct":
            self.layers_num = 33
            # self.groups = [list(range(32, 15, -1)),list(range(15, 4, -1)),[4,3],[2],[1]]
            self.out_dim=4096
        elif name=="qwen2.5-7b-instruct":
            self.layers_num = 29
            self.out_dim=3584
        elif name=="mistral-7b-instruct-v0.3":
            self.layers_num = 33
            self.out_dim=4096
        elif name == "qwen2.5-0.5b":
            self.layers_num = 25
            self.out_dim=896
        elif name == "roberta-large":
            self.layers_num = 25
            self.out_dim=1024
        # # TODO 构建W_rand_list
        # self.group_Num = args['group_Num']
        self.W_rand_list = []
        for _ in range(args['group_Num']):
            self.W_rand_list.append(torch.randn(self.out_dim,args['M']).to(self._device))
        self._network = SimpleMergeNet(args,True)
        self.args=args
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
                    "labels": batch[3],
                    "output_hidden_states": True,
                }
                y_true.append(batch[3].cpu().numpy())

            elif len(batch) == 3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # "labels": batch[2],
                    "output_hidden_states": True,
                }
                y_true.append(batch[2].cpu().numpy())

            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    def compute_similarity(self, X, Y, method="svcca", k=100):
        if method == "svcca":
            Xr, Yr = X,Y
            XY = torch.cat([Xr, Yr], dim=1).T
            C = torch.cov(XY)
            sx = Xr.shape[1]
            Cxx, Cyy, Cxy = C[:sx, :sx], C[sx:, sx:], C[:sx, sx:]
            eps = 1e-5
            Cxx += eps * torch.eye(sx, device=X.device)
            Cyy += eps * torch.eye(sx, device=X.device)
            invCxx = torch.cholesky_inverse(torch.linalg.cholesky(Cxx))
            invCyy = torch.cholesky_inverse(torch.linalg.cholesky(Cyy))
            M = invCxx @ Cxy @ invCyy @ Cxy.T
            vals = torch.sqrt(torch.clamp(torch.linalg.eigvals(M).real, min=0))
            return vals.mean()
        elif method == "cka":
            # 1. 显式中心化：确保计算的是特征的变异方向，而非绝对位置
            X = X - X.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)

            # 2. 计算 Gram 矩阵 (线性核)
            K = X @ X.T
            L = Y @ Y.T
            
            # 3. 计算 HSIC (即两个中心化核矩阵的内积)
            hsic = (K * L).sum()
            
            # 4. 归一化：将结果约束在 [0, 1] 之间
            var_x = torch.linalg.norm(K) # 等价于 sqrt((K*K).sum())
            var_y = torch.linalg.norm(L)
            
            return hsic / (var_x * var_y + 1e-8)
        elif method == "kernel_cka":
            def kernel(Z):
                pairwise_dists = torch.cdist(Z, Z)
                median_dist = torch.median(pairwise_dists)
                sigma = median_dist / 2 + 1e-8
                return torch.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
            def center(K):
                n = K.size(0)
                H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones(n, n, device=K.device)
                return H @ K @ H
            K = kernel(X)
            L = kernel(Y)
            # ✅ 关键：中心化
            Kc = center(K)
            Lc = center(L)
            HSIC = (Kc * Lc).sum()
            norm = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
            return HSIC / (norm + 1e-8)
        elif method == "cosine":
            Xf = X.flatten()
            Yf = Y.flatten()
            return torch.nn.functional.cosine_similarity(Xf, Yf, dim=0)
        elif method == "pwcca":
            eps = 1e-8
            X = X - X.mean(dim=0)
            Y = Y - Y.mean(dim=0)
            N, d = X.shape
            sy = Y.shape[1]
            
            Cxx = X.T @ X / (N - 1) + eps * torch.eye(d, device=X.device)
            Cyy = Y.T @ Y / (N - 1) + eps * torch.eye(sy, device=X.device)
            Cxy = X.T @ Y / (N - 1)

            # --- 修正部分：计算平方根逆 Cxx^{-1/2} ---
            def get_inv_sqrt(mat):
                # 使用 eigh 分解对称矩阵
                vals, vecs = torch.linalg.eigh(mat)
                # 确保特征值为正
                vals = torch.clamp(vals, min=eps)
                # 计算 Q * Λ^{-1/2} * Q^T
                return vecs @ torch.diag(1.0 / torch.sqrt(vals)) @ vecs.T

            inv_sqrt_Cxx = get_inv_sqrt(Cxx)
            inv_sqrt_Cyy = get_inv_sqrt(Cyy)
            # ---------------------------------------

            T = inv_sqrt_Cxx @ Cxy @ inv_sqrt_Cyy
            # 截断 T 以免 SVD 出现数值问题
            U, S, V = torch.linalg.svd(torch.clamp(T, -1, 1))
            
            A = inv_sqrt_Cxx @ U
            Z = X @ A
            
            alpha = torch.sum(torch.abs(Z), dim=0)
            w = alpha / (alpha.sum() + eps)
            pwcca_score = (w * S).sum()
            return pwcca_score
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
    def svcca_merge_layers(self,features: torch.Tensor, k=100, threshold=0.8,method="svcca"):
        device = features.device
        L = features.shape[0]
        # ---------- PCA 降维 ----------
        def reduce_dim(x, k):
            U, S, Vh = svd(x - x.mean(0, keepdim=True))
            return (U[:, :k] * S[:k])
        if self.method == "svcca":
            reduced = [reduce_dim(features[i], k) for i in range(L)]
        else:
            reduced = features
        merged_features = []
        group_indices = []
        # 当前组的代表特征（初始为最后一层）
        current_feat = features[-1]
        current_reduced = reduced[-1]
        current_group = [L - 1]
        # -------------------------------
        # 从倒数第二层开始往前遍历
        # -------------------------------
        for layer in range(L - 2, -1, -1):
            corr = self.compute_similarity(current_reduced, reduced[layer],method=method)
            if corr > threshold:
                # 合并：平均特征
                current_feat = (current_feat  * len(current_group) + features[layer]) / (len(current_group) + 1)
                # 更新 PCA 后的 reduced 特征
                if self.method == "svcca":
                    current_reduced = reduce_dim(current_feat, k)
                else:
                    current_reduced = current_feat
                current_group.append(layer)
            else:
                # 结束当前组
                merged_features.append(current_feat)
                group_indices.append(current_group.copy())

                # 开启新组
                current_feat = features[layer]
                current_reduced = reduced[layer]
                current_group = [layer]
        # 加入最后一个组
        merged_features.append(current_feat)
        group_indices.append(current_group.copy())
        logging.info(group_indices)       
        self.groups = [[x + 1 for x in group] for group in group_indices]
        self.group_Num = min(self.args['group_Num'],len(self.groups))
        self.groups = self.groups[0:self.group_Num]
        if self.type == "second":
            self.groups = self.groups[1:]
            self.group_Num = 1
        elif self.type == "all":
            self.groups =  [list(range(1, self.layers_num))]
            self.group_Num = 1
        if self.type == "random":
            numbers = list(range(1, 17))
            group1 = random.sample(numbers, 8)
            group2 = [num for num in numbers if num not in group1]
            # 赋值给 self.groups
            self.groups = [group1, group2]
        elif self.type == "junyun":
            # TODO 随机划分到两组
            mid = (self.layers_num - 1) // 2
            self.groups = [
                list(range(self.layers_num - 1, mid, -1)),
                list(range(mid, 0, -1))
            ]
        if self.args['merge_type'] == 'M-F':
            self.groups = self.groups[::-1]
        logging.info("groups: " + str(self.groups) + " K: " + str(self.group_Num))

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
                    train_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                    train_label_list.append(batch[2].cpu())
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                # hidden_states = model.convnet(**inputs).hidden_states
                # embedding = tuple(h.mean(dim=1) for h in hidden_states)
                # TODO 根据group 和 group_num从embedding中提取数据
                for i in range(self.group_Num):
                    group = self.groups[i]
                    begin = group[0] + 1
                    end = group[-1]
                    train_groups_features[i].append(torch.stack(embedding[end:begin], dim=0).mean(dim = 0).cpu())
                # # TODO labels
               
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
                    dev_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                    dev_label_list.append(batch[2].cpu())
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                for i in range(self.group_Num):
                    group = self.groups[i]
                    begin = group[0] + 1
                    end = group[-1]
                    dev_groups_features[i].append(torch.stack(embedding[end:begin], dim=0).mean(dim = 0).cpu())
                # # TODO 深层
                
        dev_label_list = torch.cat(dev_label_list, dim=0)
        dev_Y=target2onehot(dev_label_list,self.total_classnum)
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
        if self._cur_task > 0 :
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
    def get_group_list(self, trainloader, dev_loader, model, args):
        logging.info("分组.....")
        model = model.eval()
        train_Features_f = [[] for _ in range(self.layers_num)]
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
                    train_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                    train_label_list.append(batch[2].cpu())
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                for em_list, em in zip(train_Features_f, embedding):
                    em_list.append(em.cpu())
                
        # TODO 分组
        train_Features_f = [torch.cat(em_list, dim=0) for em_list in train_Features_f]
        train_Features_f = torch.stack(train_Features_f, dim=0)
        train_Features_f = train_Features_f[1:]
        self.svcca_merge_layers(train_Features_f)
        # TODO 更新网络参数
        self._network.update_group(self.groups,self.group_Num)
        if self.args['use_RP']:
            for d in range(self.group_Num):
                del self._network.fc_list[d]
                self._network.fc_list.append(None)
        self._network.update_fc(self._total_classes)
        # TODO 设置RP
        self.setup_RP()
        train_groups_features = []
        for i in range(len(self.groups)):
            group_indices = torch.tensor(self.groups[i], device=train_Features_f.device).long()
            corrected_indices = group_indices - 1
            train_groups_features.append(train_Features_f[corrected_indices].mean(dim = 0).cpu())
        # TODO 继续训练
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
                    dev_label_list.append(batch[3].cpu())
                elif len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        # "labels": batch[2],
                        "output_hidden_states": True,
                    }
                    dev_label_list.append(batch[2].cpu())
                embedding=model.convnet(**inputs)
                if(self.name == "mistral-7b-instruct-v0.3"):
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                for i in range(self.group_Num):
                    group = self.groups[i]
                    begin = group[0] + 1
                    end = group[-1]
                    dev_groups_features[i].append(torch.stack(embedding[end:begin], dim=0).mean(dim = 0).cpu())
                # # TODO 深层
                
        dev_label_list = torch.cat(dev_label_list, dim=0)
        dev_Y=target2onehot(dev_label_list,self.total_classnum)
        for i in range(self.group_Num):
            train_Features_f = train_groups_features[i]
            dev_Features_f = torch.cat(dev_groups_features[i],dim = 0)
            # * W_rand
            train_Features_h=torch.nn.functional.relu(train_Features_f @ self._network.fc_list[i].W_rand.cpu())
            dev_Features_h=torch.nn.functional.relu(dev_Features_f @ self._network.fc_list[i].W_rand.cpu())
            self.Q_list[i]=self.Q_list[i]+train_Features_h.T @ train_Y 
            self.G_list[i]=self.G_list[i]+train_Features_h.T @ train_Features_h
            # TODO lambda
            ridge=self.optimise_ridge_parameter(train_Features_h, train_Y, dev_Features_h, dev_Y)
            # TODO 计算分类器
            Wo=torch.linalg.solve(self.G_list[i]+ridge*torch.eye(self.G_list[i].size(dim=0)),self.Q_list[i]).T
            self._network.fc_list[i].weight.data=Wo[0:self._network.fc_list[i].weight.shape[0],:].to(self._device)
    def _train(self, train_loader, test_loader, dev_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.get_group_list(train_loader_for_protonet, dev_loader, self._network, None)
        else:
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
   