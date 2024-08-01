import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import multiclass_f1_score
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd

from types import SimpleNamespace
from model import TransformerClassifier
from datetime import datetime
from data import MyDataset, TextDataset
from utils import logprint, create_unique_folder, plot_and_save

import argparse

parser = argparse.ArgumentParser()
    
# Transformer configurations
parser.add_argument('--model_type', type=str, choices=['bert', 'roberta', 'xlnet'], default='bert',
                    help='Type of the transformer model.')
parser.add_argument('--freeze_layer', type=int, default=8,
                    help='The number of initial transformer layers to freeze.')
parser.add_argument('--freeze_embedding', type=bool, default=True,
                    help='Whether to freeze the word embedding layer in the transformer.')
parser.add_argument('--freeze_pool', type=bool, default=True,
                    help='Whether to freeze the pooling layer (BERT/Roberta only).')
parser.add_argument('--freeze_summary', type=bool, default=True,
                    help='Whether to freeze the summary layer (XLNet only).')

# Classifier configurations
parser.add_argument('--input_size', type=int, default=768,
                    help='Input dimension to the classifier.')
parser.add_argument('--hidden_layers', nargs='*', type=int, default=[],
                    help='List of integers specifying the sizes of hidden layers in the classifier.')
parser.add_argument('--num_classes', type=int, default=8,
                    help='Number of classes for classification.')

# Data configurations
parser.add_argument('--dataset_folder', type=str, default='USs/user_stories_score_full.csv',
                    help='Path to the dataset folder.')
parser.add_argument('--train_batch_size', type=int, default=32,
                    help='Training batch size.')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='Test batch size.')

# Training configurations
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs for training.')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay for optimizer.')
parser.add_argument('--log_freq', type=int, default=10,
                    help='Frequency of logging during training iterations.')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to run the model on.')

# Output configurations
parser.add_argument('--output_folder', type=str, default='output/roberta',
                    help='Folder for outputting the results.')

config = parser.parse_args()

config.output_folder = create_unique_folder(config.output_folder)

#######################################################################################
# Data Preparation

# 1.  raw text data

# 创建数据集和数据加载器
dataset = MyDataset(config.dataset_folder)

# 分出训练集和测试集
US_train, US_test, label_train, label_test = train_test_split(
    dataset.US, dataset.label, test_size=0.2, random_state=42
)

# 创建训练集和测试集的 Dataset 实例
train_dataset = MyDataset(None)
train_dataset.US = US_train
train_dataset.label = label_train

test_dataset = MyDataset(None)
test_dataset.US = US_test
test_dataset.label = label_test

# 将数据集的US处理成列表 然后进行tokenizer
train_text = train_dataset.US.tolist()
test_text = test_dataset.US.tolist()



# 2. tokenization and vectorization

# init tokenizer
if config.model_type == 'bert':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # out: ['input_ids', 'token_type_ids', 'attention_mask']
elif config.model_type == 'roberta':
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')    # out: ['input_ids', 'attention_mask']
elif config.model_type == 'xlnet':
    from transformers import XLNetTokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet/xlnet-base-cased')  # out: ['input_ids', 'token_type_ids', 'attention_mask']
else:
    raise NotImplementedError


# tokenize features
train_features = tokenizer(text=train_text,
                           add_special_tokens=True,
                           padding='max_length',
                           truncation=True,
                           max_length=128,
                           return_tensors='pt')

test_features = tokenizer(text=test_text,
                          add_special_tokens=True,
                          padding='max_length',
                          truncation=True,
                          max_length=128,
                          return_tensors='pt')


# vectorize labels
# 把训练和测试的label格式转换成tensor，因为tensor()不能接受str类型的list，所以使用OneHot转换成数字类型
labelEncoder= LabelEncoder()
train_labels = torch.tensor(labelEncoder.fit_transform(train_dataset.label.values))
test_labels = torch.tensor(labelEncoder.fit_transform(test_dataset.label.values))



# 3. prepare dataset and dataloader

train_data = TextDataset(train_features, train_labels)
test_data = TextDataset(test_features, test_labels)

train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False)




####################################################################################################
# Init Models

# init transformer backbone and freeze weights

if config.model_type == 'bert':
    from transformers import BertModel
    from model import freeze_bert_weights
    backbone = BertModel.from_pretrained('bert-base-uncased')
    print(backbone)
    freeze_bert_weights(backbone, config.freeze_layer, config.freeze_embedding, config.freeze_pool)
elif config.model_type == 'roberta':
    from transformers import BertModel
    from model import freeze_bert_weights
    backbone = BertModel.from_pretrained('roberta-base')
    print(backbone)
    freeze_bert_weights(backbone, config.freeze_layer, config.freeze_embedding, config.freeze_pool)
elif config.model_type == 'xlnet':
    from transformers import XLNetForSequenceClassification
    from model import freeze_xlnet_weights
    backbone = XLNetForSequenceClassification.from_pretrained('xlnet/xlnet-base-cased', num_labels=config.num_classes) 
    print(backbone)
    freeze_xlnet_weights(backbone, config.freeze_layer, config.freeze_embedding, config.freeze_summary)
else:
    raise NotImplementedError


# init classifier
if config.model_type == 'xlnet':
    model = backbone
else:
    model = TransformerClassifier(backbone=backbone,
                                input_size=config.input_size,
                                hidden_layers=config.hidden_layers,
                                num_classes=config.num_classes)
model = model.to(config.device)


# init optimizer
optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# loss function
loss_fn = nn.CrossEntropyLoss()




###################################################################################################################
# Train

# reset handler
for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

logging.basicConfig(filename=os.path.join(config.output_folder, 'train_log.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.info(f'configurations: {config}')
logging.info(model)



best_f1 = 0
train_loss_lst, train_f1_lst, train_acc_lst, test_loss_lst, test_f1_lst, test_acc_lst = [], [], [], [], [], []
print(f"开始训练: {datetime.now()} ")
for i in range(config.epochs):
    total_labels_train = []
    total_logits_train = []
    gradnorm_list = []
    for j, (X, y) in enumerate(train_loader):
        for k in X.keys():
            X[k] = X[k].to(config.device)
        labels = y.to(config.device).to(torch.int64)

        if config.model_type == 'xlnet':
            X['labels'] = labels

        # forward
        output = model(**X)

        # obtain loss and stats
        if config.model_type == 'xlnet':
            loss = output.loss
            logits = output.logits
        else:
            # for bert/roberta
            logits = output
            loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        gradnorm_list.append(gradnorm)
        
        optimizer.step()

        with torch.no_grad():
            total_labels_train.append(labels)
            total_logits_train.append(logits)

        if j % config.log_freq == 0:
            logprint(f"epoch {i}, batch {j}/{len(train_loader)}, train loss: {loss.item()}")

    # 打印梯度范数的均值
    print(f'梯度范数', sum(gradnorm_list) / len(gradnorm_list))
    # 评估训练结果
    with torch.no_grad():
        train_logits_all = torch.concat(total_logits_train)
        train_labels_all = torch.concat(total_labels_train).to(torch.int64)
        train_loss = loss_fn(train_logits_all, train_labels_all)
        train_f1 = multiclass_f1_score(train_logits_all, train_labels_all, num_classes=8, average='macro')
        train_acc = torch.sum(train_logits_all.argmax(-1) == train_labels_all) / len(train_labels_all)
        train_loss_lst.append(train_loss.item())
        train_f1_lst.append(train_f1.item())
        train_acc_lst.append(train_acc.item())
    print(f"开始测试: {datetime.now()} ")
    with torch.no_grad():
        total_logits_test = []
        for X, y in test_loader:
            for k in X.keys():
                X[k] = X[k].to(config.device)
            labels = y.to(config.device).to(torch.int64)

            if config.model_type == 'xlnet':
                X['labels'] = labels

            output = model(**X)

            if config.model_type == 'xlnet':
                logits = output.logits
            else:
                logits = output
            total_logits_test.append(logits)

        test_logits_all = torch.concat(total_logits_test)
        test_labels_all = test_data.labels.to(config.device).to(torch.int64)

        test_loss = loss_fn(test_logits_all, test_labels_all)
        test_f1 = multiclass_f1_score(test_logits_all, test_labels_all, num_classes=8, average='macro')
        test_acc = torch.sum(test_logits_all.argmax(-1) == test_labels_all) / len(test_labels_all)
        test_loss_lst.append(test_loss.item())
        test_f1_lst.append(test_f1.item())
        test_acc_lst.append(test_acc.item())
    
    print(f"结束测试: {datetime.now()} ")
    if test_f1 > best_f1:
        best_f1 = test_f1.item()
        torch.save(model.state_dict(), os.path.join(config.output_folder, 'best_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(config.output_folder, 'optimizer.pt'))

        logprint('Saving models ...')

    logprint(
        f"epoch {i} train loss: {train_loss}, train_f1:{train_f1}, train acc: {train_acc}\n  test loss: {test_loss}, test f1: {test_f1}, test acc: {test_acc}")
    logprint(f'epoch {i} best f1: {best_f1}')
    logprint(' ')  # empty line to separate epochs

    with open(os.path.join(config.output_folder, 'results.json'), 'w') as f:
        json.dump({
            'train_loss': train_loss_lst,
            'train_f1': train_f1_lst,
            'train_acc': train_acc_lst,
            'test_loss': test_loss_lst,
            'test_f1': test_f1_lst,
            'test_acc': test_acc_lst,
        }, f)


# 绘制 accuracy 曲线
plot_and_save(test_f1_lst, 'Test Accuracy', config.output_folder, 'test_accuracy.png')

# 绘制 test_f1_lst 曲线
plot_and_save(test_f1_lst, 'Test F1 Score', config.output_folder, 'test_f1.png')

# 绘制 test_loss_lst 曲线
plot_and_save(test_loss_lst, 'Test Loss', config.output_folder, 'test_loss.png')