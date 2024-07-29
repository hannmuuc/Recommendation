import torch.utils.data as data
import numpy as np

class BPRData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()

        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j