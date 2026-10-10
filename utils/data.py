import numpy as np
import pandas as pd


class COT_Bench():
    use_path = True

    class_order = np.arange(20).tolist()

    def download_data(self, dataset_name):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "{}/train.jsonl".format(dataset_name)
        dev_dir = "{}/dev.jsonl".format(dataset_name)
        test_dir = "{}/test.jsonl".format(dataset_name)

        train_dset = pd.read_json(train_dir, lines=True)
        dev_dset = pd.read_json(dev_dir, lines=True)
        test_dset = pd.read_json(test_dir, lines=True)

        self.train_data = train_dset['text'].values
        self.train_targets = train_dset['label'].values

        self.dev_data = dev_dset['text'].values
        self.dev_targets = dev_dset['label'].values

        self.test_data = test_dset['text'].values
        self.test_targets = test_dset['label'].values

        self.class_to_idx = train_dset.set_index('model')['label'].to_dict()
        self.classes = train_dset['model'].unique()