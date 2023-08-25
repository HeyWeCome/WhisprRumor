#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BertBase.py
@Author     ：Heywecome
@Date       ：2023/8/24 09:59 
@Description：todo
"""
import os

import torch
import torch.nn as nn

from model.BasicBert import BertModel
from model.BasicBert import BertConfig

class BertBase(nn.Module):
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        parser.add_argument('--num_label', type=int, default=2,
                            help='The numer of labels')
        parser.add_argument('--bert_pretrained_model_dir', type=str,
                            default=project_dir+'/pretrain/bert_base_chinese',
                            help='Path of pretrain model')
        parser.add_argument('--dropout', type=float, default=0, help='Prob of dropout')
        parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')

        return parser

    def __init__(self, args):
        super(BertBase, self).__init__()
        self.num_label = args.num_label
        self.bert_pretrained_model_dir = args.bert_pretrained_model_dir
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_size, args.num_label)
        self.loss = nn.CrossEntropyLoss()

        self.bert_config_path = self.bert_pretrained_model_dir+"/config.json"
        bert_config = BertConfig.from_json_file(self.bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        self.bert = BertModel.from_pretrained(bert_config, self.bert_pretrained_model_dir)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        """

        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        pooled_output, _ = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids)  # [batch_size,hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_label), labels.view(-1))
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    # bert = BertBase()
    print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))