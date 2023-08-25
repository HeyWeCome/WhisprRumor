#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：data_helpers.py
@Author     ：Heywecome
@Date       ：2023/8/22 17:01 
@Description：Some useful tools for dataloader
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six
from transformers import BertTokenizer


class Vocab:
    """
    Construct a word list from a local vocab file
    vocab = Vocab()
    print(vocab.itos)           # Get a list, return every word in the vocab;
    print(vocab.itos[2])        # Return the corresponding word in the vocab by index;
    print(vocab.stoi)           # Get a dictionary, return the index of each word in the vocab;
    print(vocab.stoi['我'])     # Return the corresponding index in the vocab by word
    print(len(vocab))           # Return the length of the vocab
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)           # Get a list, return every word in the vocab;  ；
    print(vocab.itos[2])        # Return the corresponding word in the vocab by index;
    print(vocab.stoi)           # Get a dictionary, return the index of each word in the vocab;
    print(vocab.stoi['我'])     # Return the corresponding index in the vocab by word
    """
    return Vocab(vocab_path)


def cache(func):
    """
    This decorator's purpose is to cache the results processed by the data_process() method,
    so the next time it is used the results can be loaded directly.
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"Cache file {data_path} does not exist, Reprocess and cache···")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"Cache file {data_path} exist, load the cache file directly!")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class LoadSingleSentenceDataset:
    def __init__(self,
                 vocab_path='../../pretrain/bert_base_chinese/vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\t',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):
        """
        :param vocab_path: the path of local vocab.txt
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: Configuration when processing each batch;
            When max_sen_len = None, i.e.,
            padding the others by the length of the longest sample in each batch
            When max_sen_len = 'same',
            the longest sample in the entire dataset is used as the criterion for padding the rest.
            When max_sen_len = 50,
            it means padding with some fixed length samples, and cut off the extra ones;
        :param split_sep:   Separator before text and labels, default is '\t'
        :param max_position_embeddings:
                            Specify the maximum sample length above which the sample will be intercepted.
        :param pad_index:
        :param is_sample_shuffle:Whether to disrupt the training set samples (only for the training set)
            In the subsequent construction of the DataLoader,
            both the validation set and the test set are specified to be in a fixed order (i.e., no shuffling),
            so please do not shuffle them when modifying the program.
            This is because when shuffle is True,
            the order of the samples is different each time the data_iter is traversed through a for loop.
            This can cause the order of labels returned during model prediction to be different from the original order,
            which is not easy to deal with.
        """
        self.vocab = build_vocab(vocab_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.is_sample_shuffle = is_sample_shuffle

    @cache
    def data_process(self, filepath, postfix='cache'):
        """
        Convert every word in every sentence to index form based on the dictionary,
        while also returning the length of the longest sample among all samples.
        :param filepath: Path to the dataset
        :return: Processed data and maximum sequence length
        """
        raw_lines = open(filepath, encoding="utf8").readlines()
        processed_data = []
        max_sequence_length = 0

        for raw_line in tqdm(raw_lines, ncols=80):
            fields = raw_line.rstrip("\n").split(self.split_sep)
            sentence, label = fields[1], fields[0]

            indexed_tokens = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence)]
            if len(indexed_tokens) > self.max_position_embeddings - 1:
                # BERT pre-trained model only takes the first 512 characters
                indexed_tokens = indexed_tokens[:self.max_position_embeddings - 1]
            indexed_tokens += [self.SEP_IDX]

            token_tensor = torch.tensor(indexed_tokens, dtype=torch.long)
            label_tensor = torch.tensor(int(label), dtype=torch.long)

            max_sequence_length = max(max_sequence_length, token_tensor.size(0))
            processed_data.append((token_tensor, label_tensor))

        return processed_data, max_sequence_length

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process(filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(filepath=train_file_path,
                                                    postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(filepath=val_file_path,
                                        postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


if __name__ == '__main__':
    bert_tokenize = BertTokenizer.from_pretrained("../../pretrain/bert_base_chinese/").tokenize
    loader = LoadSingleSentenceDataset(tokenizer=bert_tokenize)

    train_data, max_sen_len = loader.data_process(filepath="../../data/weibo/all_data.txt", postfix=None)  # 得到处理好的所有样本

    print(train_data[0][1].item())

    # raw_lines = open("../../data/weibo/all_data.txt", encoding="utf8").readlines()
    # field = raw_lines[0].rstrip("\n").split("\t")
    # sentence, label = field[1], field[0]
    # print(bert_tokenize(sentence))
    # print(label)
