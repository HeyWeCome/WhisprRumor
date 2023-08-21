#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：LSTM.py
@Author     ：Heywecome
@Date       ：2023/8/21 09:19 
@Description：todo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
import config


class LSTM(nn.Module):
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--embedding_size', type=int, default=300,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=128,
                            help='LSTM hidden layer size')
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of layers of LSTM')
        parser.add_argument('--batch_first', type=bool, default=True,
                            help='Whether batch first')
        parser.add_argument('--bidirectional', type=bool, default=True,
                            help='Whether the LSTM is bidirectional')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='The ratio of dropout')
        return parser

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.word2vec = utils.get_word2vec()
        self.embedding = nn.Embedding.from_pretrained(utils.get_embedding(self.word2vec,
                                             num_words=args.num_words,
                                             embedding_dim=args.embedding_size), freeze=True)
        self.lstm = nn.LSTM(input_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=args.batch_first,
                            bidirectional=args.bidirectional,
                            dropout=args.dropout)
        self.fc1 = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, 2)

    def forward(self, input):
        x = self.embedding(input)
        out, (h_n, _) = self.lstm(x)
        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]
        out_put = torch.cat([output_fw, output_bw], dim=-1)

        out_fc1 = F.relu(self.fc1(out_put))
        out_put = self.fc2(out_fc1)
        return out_put

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters


if __name__ == '__main__':
    model = LSTM()
    print(model)
