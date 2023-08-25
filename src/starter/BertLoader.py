#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BertLoader.py
@Author     ：Heywecome
@Date       ：2023/8/24 08:43 
@Description：todo
"""
import logging
import os
import pandas as pd
import torch

from utils import utils, data_helpers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class BertLoader(object):
    def parse_data_args(parser):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parser.add_argument('--path', type=str, default=project_dir + '/data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='weibo',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of txt file.')
        parser.add_argument('--max_sen_len', default=None,
                            help='None, same, 50')
        parser.add_argument('--max_position_embeddings', type=int, default=512,
                            help='Specify the maximum sample length above which the sample will be intercepted.')
        parser.add_argument('--pad_index', type=int, default=0,
                            help='pad index')
        parser.add_argument('--is_sample_shuffle', type=bool, default=True,
                            help='Whether to disrupt the training set samples (only for the training set)')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--vocab_path', type=str, default=project_dir + '/pretrain/bert_base_chinese/vocab.txt',
                            help='the path of vocab')
        parser.add_argument('--pretrained_model_dir', type=str, default=project_dir + '/pretrain/bert_base_chinese',
                            help='the path of vocab')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.path = args.path
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_sen_len = args.max_sen_len
        self.max_position_embeddings = args.max_position_embeddings
        self.pad_index = args.pad_index
        self.is_sample_shuffle = args.is_sample_shuffle
        self.regenerate = args.regenerate
        self.vocab_path = args.vocab_path
        self.pretrained_model_dir = args.pretrained_model_dir

    def get_dataloader(self):
        # Define the filenames
        file_path = self.path + self.dataset
        input_filename = file_path + '/all_data.txt'
        train_filename = file_path + '/train.txt'
        val_filename = file_path + '/val.txt'
        test_filename = file_path + '/test.txt'

        # Check if the output files already exist
        if os.path.exists(train_filename) \
                and os.path.exists(val_filename) \
                and os.path.exists(test_filename) \
                and self.regenerate != 1:
            logging.info("Split files already exist. No need to regenerate. Loading from disk···")
            with open(train_filename, 'r', encoding='utf-8') as file:
                x_train = file.readlines()

            with open(val_filename, 'r', encoding='utf-8') as file:
                x_val = file.readlines()

            with open(test_filename, 'r', encoding='utf-8') as file:
                x_test = file.readlines()
        else:
            # Read the data from all_data.txt
            with open(input_filename, 'r', encoding='utf-8') as file:
                contents = file.readlines()

            # Splitting the data into train, val, and test sets
            x_train, x_test = train_test_split(contents,
                                               test_size=0.2,  # 20% for testing
                                               shuffle=True,
                                               random_state=0)
            x_test, x_val = train_test_split(x_test,
                                             test_size=0.5,  # 50% of remaining data for validation
                                             shuffle=True,
                                             random_state=0)

            utils.save_to_file(train_filename, x_train)
            utils.save_to_file(test_filename, x_test)
            utils.save_to_file(val_filename, x_val)
            logging.info("Data split and saved successfully.")

        bert_tokenize = BertTokenizer.from_pretrained(self.pretrained_model_dir).tokenize
        data_loader = data_helpers.LoadSingleSentenceDataset(vocab_path=self.vocab_path,
                                                             tokenizer=bert_tokenize,
                                                             batch_size=self.batch_size,
                                                             max_sen_len=self.max_sen_len,
                                                             split_sep=self.sep,
                                                             max_position_embeddings=self.max_position_embeddings,
                                                             pad_index=self.pad_index,
                                                             is_sample_shuffle=self.is_sample_shuffle)

        train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(train_filename,
                                                                               val_filename,
                                                                               test_filename)

        return train_iter, test_iter, val_iter


class RummorDataset(Dataset):
    def __init__(self, model, contents, labels):
        super(RummorDataset, self).__init__()
        self.contents = contents
        self.labels = labels

    def __getitem__(self, item):
        return torch.tensor(self.contents[item]), torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.contents)


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(project_dir)
