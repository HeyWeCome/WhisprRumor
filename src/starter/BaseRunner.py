#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BaseRunner.py
@Author     ：Heywecome
@Date       ：2023/8/21 10:11 
@Description：BaseRunner
"""
import logging
import os

import numpy as np
import torch
import torch.optim as optim

from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import CrossEntropyLoss
from utils import utils
from tqdm import tqdm


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=3,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=5e-5,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--metric', type=str, default='ACC,f1',
                            help='metrics: ACC')
        parser.add_argument('--model_val_per_epoch', type=int, default=2,
                            help='val per epoch')
        return parser

    def __init__(self, args):
        self.model_name_suffix = args.model_name
        self.epoch = args.epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.lr = args.lr
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}'.format(self.metrics[0])  # early stop based on main_metric
        self.device = args.device
        self.time = None  # will store [start_time, last_step_time]
        self.pad_idx = args.pad_index
        self.model_val_per_epoch = args.model_val_per_epoch

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def save_model(self, model, acc, model_name_suffix=None):
        if model_name_suffix is None:
            model_name_suffix = f"acc_{acc:.5f}"

        model_dir = '../../model/' + model_name_suffix
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"model_{model_name_suffix}.pth"
        model_path = os.path.join(model_dir, model_name)

        torch.save(model.state_dict(), model_path)
        print(f"Saved model with accuracy: {acc:.5f} to {model_path}")

    def evaluate_metrics(self, dataloader, model, device):
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for sample, label in dataloader:
                sample = sample.to(device)
                label = label.to(device)
                logits = model(input_ids=sample)
                preds = logits.argmax(1)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        # precision = precision_score(all_labels, all_preds, average='weighted')
        # recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return acc, f1

    def train(self, model, optimizer, train_dataloader, val_dataloader):
        max_acc = 0
        stop = 0
        model.train()
        for epoch in range(self.epoch):
            losses = 0
            # bar = tqdm(train_dataloader, total=len(train_dataloader))  # 配置进度条
            start_time = time()
            for idx, (sample, label) in enumerate(train_dataloader):
                sample = sample.to(self.device)
                label = label.to(self.device)
                loss, logits = model(
                    input_ids=sample,
                    labels=label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_dataloader)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
            end_time = time()
            train_loss = losses / len(train_dataloader)

            # Evaluate performance
            acc, f1 = self.evaluate(val_dataloader, model, self.device, PAD_IDX=self.pad_idx)
            if acc > max_acc:
                max_acc = acc
                logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                             f"Epoch time = {(end_time - start_time):.3f}s, "
                             f"F1: {f1:.3f}, "
                             f"Accuracy on val {acc:.3f}*")
                stop = 0
            else:
                logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                             f"Epoch time = {(end_time - start_time):.3f}s, "
                             f"F1: {f1:.3f}, "
                             f"Accuracy on val {acc:.3f}")
                stop += 1
            if stop >= self.early_stop:
                logging.info("Early stop.")
                break

    def inference(self, model, test_iter):
        model.eval()
        acc = self.evaluate(test_iter, model, device=self.device, PAD_IDX=self.pad_idx)
        logging.info(f"Acc on test: {acc:.3f}")

    def evaluate(self, dataloader, model, device, PAD_IDX):
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sample, label in dataloader:
                sample = sample.to(device)
                label = label.to(device)
                padding_mask = (sample == PAD_IDX).transpose(0, 1)
                logits = model(input_ids=sample, attention_mask=padding_mask)
                preds = logits.argmax(1)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        model.train()
        return acc, f1

    def fit(self, model, train_loader, val_loader, test_loader):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        train_dataloader = train_loader
        val_dataloader = val_loader  # Adding validation dataloader
        test_dataloader = test_loader

        self.train(model=model,
                   optimizer=optimizer,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader)
        self.inference(model=model, test_iter=test_dataloader)

