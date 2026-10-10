import logging
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from utils.data import COT_Bench


class DataManager(object):
    def __init__(self, dataset_name, convnet_type, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self.convnet_type = convnet_type
        self._setup_data_text(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_texts_dataset(self, indices, source, appendent=None, ret_data=False, m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "dev":
            x, y = self._dev_data, self._dev_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)

        # Select tokenizer and dataset class based on convnet type
        if self.convnet_type == 'Roberta-large':
            tokenizer = AutoTokenizer.from_pretrained("/home/lihr/model/roberta-large")
            text_Dataset = textsDataset(tokenizer, data, targets)
        elif self.convnet_type == 'Llama-3.2-1B':
            tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-3.2-1B")
            text_Dataset = LlamaDataset(tokenizer, data, targets)
        elif self.convnet_type == 'Llama-3-8B-Instruct':
            tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-3-8B-Instruct")
            text_Dataset = LlamaDataset(tokenizer, data, targets)
        elif self.convnet_type == 'Mistral-7B-Instruct-v0.3':
            tokenizer = AutoTokenizer.from_pretrained("../../model/Mistral-7B-Instruct-v0.3")
            text_Dataset = MistralDataset(tokenizer, data, targets)
        elif self.convnet_type == "qwen2.5-0.5B":
            tokenizer = AutoTokenizer.from_pretrained("/home/wangyujie/model/Qwen2.5-0.5B-Instruct")
            text_Dataset = QwenDataset(tokenizer, data, targets)
        else:
            raise NotImplementedError("Unsupported convnet type: {}".format(self.convnet_type))

        if ret_data:
            return data, targets, text_Dataset
        else:
            return text_Dataset

    def get_texts_old_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        data, targets = [], []
        appendent_data, appendent_targets = appendent
        data.append(appendent_data)
        targets.append(appendent_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)
        tokenizer = AutoTokenizer.from_pretrained("../../model/roberta-base")
        if ret_data:
            return data, targets, textsDataset(tokenizer, data, targets)
        else:
            return textsDataset(tokenizer, data, targets)

    def _setup_data_text(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data(self.dataset_name)
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._dev_data, self._dev_targets = idata.dev_data, idata.dev_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._class_to_idx, self._classes = idata.class_to_idx, idata.classes

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        for key, value in self._class_to_idx.items():
            self._class_to_idx[key] = self._class_order.index(value)
        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(0, len(idxes), size=int((1 - m_rate) * len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class textsDataset(Dataset):
    """For BERT-style models with token_type_ids (e.g., RoBERTa-large)."""

    def __init__(self, tokenizer, texts, labels, max_seq_length=512):
        super(textsDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]

    def get_input(self, texts, labels):
        tokens_text = list(map(self.tokenizer.tokenize, texts))
        result = list(map(self.truncate_and_pad, tokens_text))
        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        tokens_type_ids = [i[2] for i in result]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(tokens_type_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def truncate_and_pad(self, tokens_text):
        tokens_text = ['[cls]'] + tokens_text
        if len(tokens_text) > self.max_seq_length:
            tokens_text = tokens_text[0: self.max_seq_length]
        padding = [0] * (self.max_seq_length - len(tokens_text))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_text)
        input_ids += padding
        attention_mask = [1] * len(tokens_text) + padding
        token_type_ids = [0] * (self.max_seq_length)
        assert len(input_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(token_type_ids) == self.max_seq_length
        return input_ids, attention_mask, token_type_ids


class LlamaDataset(Dataset):
    """For LLaMA-series models (no token_type_ids)."""

    def __init__(self, tokenizer, texts, labels, max_seq_length=512):
        super(LlamaDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.tokenizer.pad_token = tokenizer.eos_token
        self.input_ids, self.attention_mask, self.labels = self.get_input(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def get_input(self, texts, labels):
        texts = texts.tolist()
        encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return (
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )


class QwenDataset(Dataset):
    """For Qwen-series models (no token_type_ids)."""

    def __init__(self, tokenizer, texts, labels, max_seq_length=512):
        super(QwenDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_ids, self.attention_mask, self.labels = self.get_input(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def get_input(self, texts, labels):
        texts = texts.tolist()
        encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return (
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )


class MistralDataset(Dataset):
    """For Mistral-series models (no token_type_ids)."""

    def __init__(self, tokenizer, texts, labels, max_seq_length=512):
        super(MistralDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_ids, self.attention_mask, self.labels = self.get_input(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def get_input(self, texts, labels):
        texts = texts.tolist()
        encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return (
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if "cot-bench" in name or "cot" in name:
        return COT_Bench()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
