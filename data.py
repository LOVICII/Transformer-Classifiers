from torch.utils.data import Dataset
import pandas as pd



# 创建一个dataset类，这个类的属性是数据集中的US和label，label这里的列名后续需要修改
class MyDataset(Dataset):
    def __init__(self, csv_file):
        if csv_file is None:
            self.US = None
            self.label = None
        else:
            self.data = pd.read_csv(csv_file)
            self.US = self.data['user_story']
            self.label= self.data['1_score_label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.US.iloc[index], self.label.iloc[index]
    

class TextDataset(Dataset):
    def __init__(self, features: dict, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        feature = {k:self.features[k][idx] for k in self.features.keys()}
        label = self.labels[idx]
        return feature, label
