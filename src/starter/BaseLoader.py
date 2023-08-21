#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BaseLoader.py
@Author     ：Heywecome
@Date       ：2023/8/21 09:44 
@Description：todo
"""
import pandas as pd
import torch

from utils import utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config as config


class BaseLoader(object):
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='weibo',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of txt file.')
        parser.add_argument('--num_words', type=int, default=50000,
                            help='Maximum number of words, '
                                 'selecting the words with the highest usage rate before using new_words')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.path = args.path
        self.dataset = args.dataset
        self.num_words = args.num_words
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        # self._get_dataloader()

    def get_dataloader(self):
        weibo = pd.read_csv(self.path + self.dataset + '/all_data.txt',
                            sep='\t',
                            names=['label', 'content'],
                            encoding='utf-8')
        weibo = weibo.dropna()  # Remove missing values
        labels, contents = weibo['label'].values.tolist(), weibo['content'].values.tolist()
        contents = utils.key_to_index(contents, utils.get_word2vec(), self.num_words)
        maxlen = utils.get_maxlength(contents)
        contents = utils.padding_truncating(contents, maxlen)

        # Splitting the data into train, val, and test sets
        x_train, x_test, y_train, y_test = train_test_split(contents,
                                                            labels,
                                                            test_size=0.2,  # 20% for testing
                                                            shuffle=True,
                                                            random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.5,  # 50% of remaining data for validation
                                                          shuffle=True,
                                                          random_state=0)

        train_dataset = RummorDataset(model="train", contents=x_train, labels=y_train)
        val_dataset = RummorDataset(model="val", contents=x_val, labels=y_val)
        test_dataset = RummorDataset(model="test", contents=x_test, labels=y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


class RummorDataset(Dataset):
    def __init__(self, model, contents, labels):
        super(RummorDataset, self).__init__()
        self.contents = contents
        self.labels = labels

    def __getitem__(self, item):
        return torch.tensor(self.contents[item]), torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.contents)


def get_dataloader(model="train"):
    labels, contents = utils.get_df()
    contents = utils.key_to_index(contents, utils.word2vec, config.num_words)
    maxlen = utils.get_maxlength(contents)
    contents = utils.padding_truncating(contents, maxlen)

    # Splitting the data into train, val, and test sets
    x_train, x_test, y_train, y_test = train_test_split(contents,
                                                        labels,
                                                        test_size=0.2,  # 20% for testing
                                                        shuffle=True,
                                                        random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.5,  # 50% of remaining data for validation
                                                      shuffle=True,
                                                      random_state=0)

    train_dataset = RummorDataset(model="train", contents=x_train, labels=y_train)
    val_dataset = RummorDataset(model="val", contents=x_val, labels=y_val)
    test_dataset = RummorDataset(model="test", contents=x_test, labels=y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    if model == "train":
        return train_loader
    elif model == "val":
        return val_loader
    elif model == "test":
        return test_loader
    else:
        raise Exception("Please choose train, val and test.")


if __name__ == '__main__':
    train_loader = get_dataloader("train")
    train_batch = next(iter(train_loader))
    print("Training batch:")
    print(train_batch)
