#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：utils.py
@Author     ：Heywecome
@Date       ：2023/8/21 08:57 
@Description：tools
"""
import bz2
import os
import gensim
import jieba
import re
import numpy as np
import pandas as pd
import pkuseg
import torch

from datetime import datetime
from typing import NoReturn
from gensim.models import KeyedVectors


def init_random_seed(seed=2023):
    """
    Fix random seed of numpy and pytorch
    :param seed: the number of random seed
    :return: None
    """
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
    """
    Format the args
    :param args:
    :param exclude_lst:
    :param max_len:
    :return:
    """
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len - 3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def get_time():
    """
    Get current time.
    :return:
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_dir(file_name: str) -> NoReturn:
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)


# pkuseg分词，可以选择微博领域
def pkuseg_cut(contents, model_name="web"):
    seg = pkuseg.pkuseg(model_name='web')  # 程序会自动下载所对应的细领域模型
    contents_S = []
    for line in contents:
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
        current_segment = jieba.lcut(line)  # 列表，元素为分割出来的词
        contents_S.append(current_segment)
    return contents_S


def get_stopwords():
    stopwords = pd.read_csv("../stopwords/stopwords.txt",
                            index_col=False,
                            sep="\t",
                            quoting=3,
                            names=['stopword'],
                            encoding='utf-8')
    return set(stopwords['stopword'].values.tolist())


def drop_stopwords(contents, stopwords):
    """
    Drop stopwords
    :param contents:
    :param stopwords:
    :return:
    """
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))  # str()转换为字符串##记录所有line_clean中的词
        contents_clean.append(line_clean)
    return contents_clean, all_words

def bz2Decompress():
    if os.path.exists("../../embeddings/sgns.weibo.bigram") == False:
        with open("./embeddings/sgns.weibo.bigram", 'wb') as new_file, open("./embeddings/sgns.weibo.bigram.bz2",
                                                                            'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(decompressor.decompress(data))

def get_word2vec():
    word2vec = KeyedVectors.load_word2vec_format('../embeddings/sgns.weibo.bigram',
                                                 binary=False,
                                                 unicode_errors="ignore")
    return word2vec

def get_embedding(word2vec, num_words=50000, embedding_dim=300):
    """
    :param num_words: 只选择使用前50k个使用频率最高的词
    :param embedding_dim: 词向量维度
    :param word2vec: 预训练好的词向量模型
    :return:
    """
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(num_words):
        # embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]#前50000个index对应的词的词向量
        embedding_matrix[i, :] = word2vec[i]  # 前50000个index对应的词的词向量
    embedding_matrix = embedding_matrix.astype('float32')
    return torch.from_numpy(embedding_matrix)

def key_to_index(contents, word2vec, num_words):
    """
    :param contents:
    :param word2vec:预训练好的词向量模型，词向量根据使用频率降序排列
    :param num_words: 最大词汇数，选择使用前new_words个使用评率最高的词
    :return:
    """
    train_tokens = []
    contents_S = pkuseg_cut(contents)
    stopword = get_stopwords()
    contents_clean, all_words = drop_stopwords(contents_S, stopword)
    for line_clean in contents_clean:
        for i, key in enumerate(line_clean):
            try:
                index = word2vec.key_to_index[key]
                if index < num_words:
                    line_clean[i] = word2vec.key_to_index[key]
                else:
                    line_clean[i] = 0  # 超出前num_words个词用0代替
            except KeyError:  # 如果词不在字典中，则输出0
                line_clean[i] = 0
        train_tokens.append(line_clean)
    return train_tokens


def get_maxlength(train_tokens):
    num_tokens = [len(tokens) for tokens in train_tokens]
    num_tokens = np.array(num_tokens)
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    return max_tokens


def padding_truncating(train_tokens, maxlen):
    for i, token in enumerate(train_tokens):
        if len(token) > maxlen:
            train_tokens[i] = token[len(token) - maxlen:]
        elif len(token) < maxlen:
            train_tokens[i] = [0] * (maxlen - len(token)) + token
    return train_tokens
