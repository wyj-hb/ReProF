import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.linalg import svd
from utils.inc_net import SimpleMergeNet
from models.base import BaseLearner
from utils.toolkit import target2onehot
import random

batch_size = 16


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        name = args["convnet_type"].lower()
        self.name = name
        self.type = args['type']
        self.method = args['method']
        self.args = args

        MODEL_CONFIG = {
            "llama-3.2-1b":             (17, 2048),
            "llama-3-8b-instruct":      (33, 4096),
            "mistral-7b-instruct-v0.3": (33, 4096),
            "qwen2.5-0.5b":             (25, 896),
            "roberta-large":            (25, 1024),
        }
        if name not in MODEL_CONFIG:
            raise ValueError(f"Unsupported model: {name}")
        self.layers_num, self.out_dim = MODEL_CONFIG[name]

        self.W_rand_list = [
            torch.randn(self.out_dim, args['M']).to(self._device)
            for _ in range(args['group_Num'])
        ]
        self._network = SimpleMergeNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for step, batch in enumerate(loader):
            batch = tuple(t.to(self._device) for t in batch)
            if len(batch) == 4:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "output_hidden_states": True,
                }
                y_true.append(batch[3].cpu().numpy())
            elif len(batch) == 3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "output_hidden_states": True,
                }
                y_true.append(batch[2].cpu().numpy())
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    # ------------------------------------------------------------------ #
    #  Similarity / Layer Grouping
    # ------------------------------------------------------------------ #

    def compute_similarity(self, X, Y, method="svcca", k=100):
        if method == "svcca":
            Xr, Yr = X, Y
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
            X = X - X.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)
            K = X @ X.T
            L = Y @ Y.T
            hsic = (K * L).sum()
            return hsic / (torch.linalg.norm(K) * torch.linalg.norm(L) + 1e-8)
        elif method == "kernel_cka":
            def kernel(Z):
                dists = torch.cdist(Z, Z)
                sigma = torch.median(dists) / 2 + 1e-8
                return torch.exp(-dists ** 2 / (2 * sigma ** 2))
            n = X.size(0)
            H = torch.eye(n, device=X.device) - (1.0 / n) * torch.ones(n, n, device=X.device)
            Kc = H @ kernel(X) @ H
            Lc = H @ kernel(Y) @ H
            HSIC = (Kc * Lc).sum()
            norm = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
            return HSIC / (norm + 1e-8)
        elif method == "cosine":
            return F.cosine_similarity(X.flatten(), Y.flatten(), dim=0)
        elif method == "pwcca":
            eps = 1e-8
            X = X - X.mean(dim=0)
            Y = Y - Y.mean(dim=0)
            N, d = X.shape
            sy = Y.shape[1]

            Cxx = X.T @ X / (N - 1) + eps * torch.eye(d, device=X.device)
            Cyy = Y.T @ Y / (N - 1) + eps * torch.eye(sy, device=X.device)
            Cxy = X.T @ Y / (N - 1)

            def _inv_sqrt(mat):
                vals, vecs = torch.linalg.eigh(mat)
                vals = torch.clamp(vals, min=eps)
                return vecs @ torch.diag(1.0 / torch.sqrt(vals)) @ vecs.T

            T = _inv_sqrt(Cxx) @ Cxy @ _inv_sqrt(Cyy)
            U, S, V = torch.linalg.svd(torch.clamp(T, -1, 1))
            A = _inv_sqrt(Cxx) @ U
            Z = X @ A
            alpha = torch.sum(torch.abs(Z), dim=0)
            w = alpha / (alpha.sum() + eps)
            return (w * S).sum()
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def svcca_merge_layers(self, features: torch.Tensor, k=100, threshold=0.8, method="svcca"):
        device = features.device
        L = features.shape[0]

        def reduce_dim(x, k):
            U, S, Vh = svd(x - x.mean(0, keepdim=True))
            return (U[:, :k] * S[:k])

        if self.method == "svcca":
            reduced = [reduce_dim(features[i], k) for i in range(L)]
        else:
            reduced = features

        merged_features = []
        group_indices = []
        current_feat = features[-1]
        current_reduced = reduced[-1]
        current_group = [L - 1]

        for layer in range(L - 2, -1, -1):
            corr = self.compute_similarity(current_reduced, reduced[layer], method=method)
            if corr > threshold:
                current_feat = (current_feat * len(current_group) + features[layer]) / (len(current_group) + 1)
                if self.method == "svcca":
                    current_reduced = reduce_dim(current_feat, k)
                else:
                    current_reduced = current_feat
                current_group.append(layer)
            else:
                merged_features.append(current_feat)
                group_indices.append(current_group.copy())
                current_feat = features[layer]
                current_reduced = reduced[layer]
                current_group = [layer]
        merged_features.append(current_feat)
        group_indices.append(current_group.copy())

        self.groups = [[x + 1 for x in group] for group in group_indices]
        self.group_Num = min(self.args['group_Num'], len(self.groups))
        self.groups = self.groups[0:self.group_Num]

        if self.type == "second":
            # A tiny validation split can merge every layer into one block.
            # In that degenerate case there is no second block to select, so
            # retain the only (and deepest) block instead of producing none.
            self.groups = self.groups[1:] if len(self.groups) > 1 else self.groups[:1]
            self.group_Num = len(self.groups)
        elif self.type == "all":
            self.groups = [list(range(1, self.layers_num))]
            self.group_Num = 1
        if self.type == "random":
            numbers = list(range(1, 17))
            group1 = random.sample(numbers, 8)
            group2 = [num for num in numbers if num not in group1]
            self.groups = [group1, group2]
        elif self.type == "junyun":
            mid = (self.layers_num - 1) // 2
            self.groups = [
                list(range(self.layers_num - 1, mid, -1)),
                list(range(mid, 0, -1)),
            ]
        if self.args['merge_type'] == 'M-F':
            self.groups = self.groups[::-1]
        logging.info("groups: " + str(self.groups) + " K: " + str(self.group_Num))

    # ------------------------------------------------------------------ #
    #  Feature Extraction Helpers
    # ------------------------------------------------------------------ #

    def _collect_group_features(self, loader):
        """Forward pass: extract mean-pooled hidden states for each group.

        Returns:
            group_features: list of [N, D] tensors, one per group
            labels: [N] class labels
        """
        group_features = [[] for _ in range(self.group_Num)]
        label_list = []

        self._network.eval()
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.cuda(self._device) for t in batch)
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "output_hidden_states": True,
                    }
                    label_list.append(batch[3].cpu())
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "output_hidden_states": True,
                    }
                    label_list.append(batch[2].cpu())

                embedding = self._network.convnet(**inputs)
                if self.name == "mistral-7b-instruct-v0.3":
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)

                for i in range(self.group_Num):
                    selected = [embedding[idx] for idx in self.groups[i]]
                    group_mean = torch.stack(selected, dim=0).mean(dim=0).cpu()
                    group_features[i].append(group_mean)

        labels = torch.cat(label_list, dim=0)
        return [torch.cat(f, dim=0) for f in group_features], labels

    def _fit_ridge_classifiers(self, train_features, train_Y, dev_features, dev_Y):
        """Fit ridge regression for all groups with residual stacking."""
        for i in range(self.group_Num):
            train_h = F.relu(train_features[i] @ self._network.fc_list[i].W_rand.cpu())
            dev_h = F.relu(dev_features[i] @ self._network.fc_list[i].W_rand.cpu())

            self.Q_list[i] = self.Q_list[i] + train_h.T @ train_Y
            self.G_list[i] = self.G_list[i] + train_h.T @ train_h

            ridge = self._optimise_ridge_parameter(train_h, train_Y, dev_h, dev_Y)
            Wo = torch.linalg.solve(
                self.G_list[i] + ridge * torch.eye(self.G_list[i].size(dim=0)),
                self.Q_list[i]
            ).T
            self._network.fc_list[i].weight.data = Wo[:self._network.fc_list[i].weight.shape[0], :].to(self._device)

            if i == self.group_Num - 1:
                continue

            # Residual prediction for the next group
            train_pred = F.linear(train_h.to(self._device), self._network.fc_list[i].weight).detach().cpu()
            train_Y = train_Y.clone()
            train_Y[:, :self._total_classes] -= train_pred
            train_Y[:, :self._known_classes] = 0

            dev_pred = F.linear(dev_h.to(self._device), self._network.fc_list[i].weight).detach().cpu()
            dev_Y = dev_Y.clone()
            dev_Y[:, :self._total_classes] -= dev_pred
            dev_Y[:, :self._known_classes] = 0

    def _optimise_ridge_parameter(self, train_Features, train_Y, dev_Features, dev_Y):
        ridges = 10.0 ** np.arange(-1, 6)
        losses = []
        Q_val = train_Features.T @ train_Y
        G_val = train_Features.T @ train_Features
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T
            losses.append(F.mse_loss(dev_Features @ Wo.T, dev_Y))
        ridge = ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: " + str(ridge))
        return ridge

    # ------------------------------------------------------------------ #
    #  Main Training Loop
    # ------------------------------------------------------------------ #

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self._cur_task > 0 and self.args['use_RP']:
            for d in range(self.group_Num):
                del self._network.fc_list[d]
                self._network.fc_list.append(None)
            self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.data_manager = data_manager

        train_dataset = data_manager.get_texts_dataset(
            np.arange(self._known_classes, self._total_classes), source="train")
        dev_dataset = data_manager.get_texts_dataset(
            np.arange(self._known_classes, self._total_classes), source="dev")
        test_dataset = data_manager.get_texts_dataset(
            np.arange(0, self._total_classes), source="test")

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.train_loader_for_protonet = self.train_loader

        self._train(self.train_loader, self.test_loader, self.dev_loader, self.train_loader_for_protonet)

    def get_group_list(self, trainloader, dev_loader, model, args):
        logging.info(f"Stage {self._cur_task + 1} 分组.....")
        model = model.eval()

        # Step 1: Collect per-layer hidden states for similarity-based grouping
        all_layer_features = [[] for _ in range(self.layers_num)]
        with torch.no_grad():
            for batch in trainloader:
                batch = tuple(t.cuda(self._device) for t in batch)
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0], "attention_mask": batch[1],
                        "token_type_ids": batch[2], "output_hidden_states": True,
                    }
                else:
                    inputs = {
                        "input_ids": batch[0], "attention_mask": batch[1],
                        "output_hidden_states": True,
                    }
                embedding = model.convnet(**inputs)
                if self.name == "mistral-7b-instruct-v0.3":
                    embedding = tuple(h.mean(dim=1) for h in embedding.hidden_states)
                for em_list, em in zip(all_layer_features, embedding):
                    em_list.append(em.cpu())

        # Step 2: Run SVCCA / similarity-based grouping
        all_layer_features = [torch.cat(em_list, dim=0) for em_list in all_layer_features]
        all_layer_features = torch.stack(all_layer_features, dim=0)[1:]  # drop input embedding
        self.svcca_merge_layers(all_layer_features, method=self.method)

        # Step 3: Update network and setup random projection
        self._network.update_group(self.groups, self.group_Num)
        if self.args['use_RP']:
            for d in range(self.group_Num):
                del self._network.fc_list[d]
                self._network.fc_list.append(None)
        self._network.update_fc(self._total_classes)
        self.setup_RP()

        # Step 4: Collect group features and fit classifiers
        train_features, train_labels = self._collect_group_features(trainloader)
        dev_features, dev_labels = self._collect_group_features(dev_loader)
        train_Y = target2onehot(train_labels, self.total_classnum)
        dev_Y = target2onehot(dev_labels, self.total_classnum)
        self._fit_ridge_classifiers(train_features, train_Y, dev_features, dev_Y)

    def replace_fc(self, trainloader, dev_loader, model, args):
        model = model.eval()
        for f in range(self.group_Num):
            if self.args['use_RP']:
                self._network.fc_list[f].use_RP = True
                if self.args['M'] > 0:
                    self._network.fc_list[f].W_rand = self.W_rand_list[f]

        train_features, train_labels = self._collect_group_features(trainloader)
        dev_features, dev_labels = self._collect_group_features(dev_loader)
        train_Y = target2onehot(train_labels, self.total_classnum)
        dev_Y = target2onehot(dev_labels, self.total_classnum)
        self._fit_ridge_classifiers(train_features, train_Y, dev_features, dev_Y)

    def _train(self, train_loader, test_loader, dev_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self._cur_task == 0 or self.args['recalculate']:
            self.get_group_list(train_loader_for_protonet, dev_loader, self._network, None)
        else:
            self.replace_fc(train_loader_for_protonet, dev_loader, self._network, None)

    def setup_RP(self):
        self.Q_list = []
        self.G_list = []
        for i in range(self.group_Num):
            self._network.fc_list[i].use_RP = True
            if self.args['M'] > 0:
                M = self.args['M']
                fc = self._network.fc_list[i]
                fc.weight = nn.Parameter(torch.Tensor(fc.out_features, M).to(self._device))
                fc.reset_parameters()
                fc.W_rand = self.W_rand_list[i]
            self.Q_list.append(torch.zeros(M, self.total_classnum))
            self.G_list.append(torch.zeros(M, M))
