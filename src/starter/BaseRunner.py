#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BaseRunner.py
@Author     ：Heywecome
@Date       ：2023/8/21 10:11 
@Description：todo
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
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=3,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--metric', type=str, default='ACC',
                            help='metrics: ACC')
        parser.add_argument('--model_val_per_epoch', type=int, default=2,
                            help='val per epoch')
        return parser

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.lr = args.lr
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
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

    # def save_model(self, model, acc):
    #     model_path = '../../model/best_model.pkl'
    #     torch.save(model.state_dict(), model_path)
    #     print(f"Saved model with accuracy: {acc:.5f} to {model_path}")

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
            acc, f1 = self.evaluate_metrics(val_dataloader, model, self.device)
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

            # if (epoch + 1) % self.model_val_per_epoch == 0:
            #     acc = self.evaluate(val_dataloader, model, self.device, self.pad_idx)
            #     logging.info(f"Accuracy on val {acc:.3f}")
            #     if acc > max_acc:
            #         max_acc = acc
                    # torch.save(model.state_dict(), model_save_path)

    def inference(self, model, test_iter):
        model.eval()
        acc = self.evaluate(test_iter, model, device=self.device, PAD_IDX=self.pad_idx)
        logging.info(f"Acc on test: {acc:.3f}")

    def evaluate(self, data_iter, model, device, PAD_IDX):
        model.eval()

        with torch.no_grad():
            acc_sum, n = 0.0, 0

            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)

                logits = model(x, attention_mask=padding_mask)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

        model.train()
        return acc_sum / n

    # def test(self, model, loss_fn, data_loader, stage):
    #     model.eval()
    #     loss_list = []
    #     acc = 0
    #     total = 0
    #     loss_fn.to(self.device)
    #     with torch.no_grad():
    #         for input, target in data_loader:
    #             input = input.to(self.device)
    #             target = target.to(self.device)
    #             output = model(input)
    #             loss = loss_fn(output, target)
    #             loss_list.append(loss.item())
    #             # Accuracy
    #             output_max = output.max(dim=-1)
    #             pred = output_max[-1]
    #             acc += pred.eq(target).cpu().float().sum().item()
    #             total += target.shape[0]
    #
    #     avg_loss = np.mean(loss_list)
    #     avg_acc = acc / total
    #     logging.info("{} loss:{:.6f}, acc:{:.6f}".format(stage, avg_loss, avg_acc))
    #     return avg_acc, avg_loss

    # def fit(self, model, train_loader, val_loader, test_loader):
    #     optimizer = optim.Adam(model.parameters(), lr=self.lr)
    #     train_dataloader = train_loader
    #     val_dataloader = val_loader  # Adding validation dataloader
    #     test_dataloader = test_loader
    #     loss_fn = CrossEntropyLoss()
    #
    #     best_acc = 0
    #     early_stop_cnt = 0
    #     train_loss_list = []
    #     val_loss_list = []  # Adding validation loss list
    #
    #     for epoch in range(self.epoch):
    #         train_acc, train_loss = self.train(epoch,
    #                                            model,
    #                                            loss_fn,
    #                                            optimizer,
    #                                            train_dataloader)
    #         val_acc, val_loss = self.test(model,
    #                                       loss_fn,
    #                                       val_dataloader,
    #                                       'val')  # Using val_dataloader for validation
    #         train_loss_list.append(train_loss)
    #         val_loss_list.append(val_loss)  # Adding validation loss
    #
    #         # Using validation accuracy for early stopping and saving the best model
    #         if val_acc > best_acc:
    #             best_acc = val_acc
    #             # self.save_model(model, best_acc)
    #             early_stop_cnt = 0
    #         else:
    #             early_stop_cnt += 1
    #
    #         if early_stop_cnt > self.early_stop:
    #             logging.info("Early stopping triggered.")
    #             break
    #
    #     # plot_learning_curve(train_loss_list, val_loss_list, test_loss_list)  # Plotting validation loss as well
    #
    #     # Print the best validation accuracy
    #     logging.info("Best Validation Accuracy: {:.5f}".format(best_acc))
    #
    #     # After training, get the test results
    #     final_test_acc, final_test_loss = self.test(model,
    #                                                 loss_fn,
    #                                                 test_dataloader,
    #                                                 'test')
    #     logging.info(os.linesep + "Final Test Accuracy: {:.5f}. Test Loss: {:.5f}"
    #                  .format(final_test_acc, final_test_loss))

    def fit(self, model, train_loader, val_loader, test_loader):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        train_dataloader = train_loader
        val_dataloader = val_loader  # Adding validation dataloader
        test_dataloader = test_loader
        loss_fn = CrossEntropyLoss()

        best_acc = 0
        early_stop_cnt = 0
        train_loss_list = []
        val_loss_list = []  # Adding validation loss list

        self.train(model=model, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        self.inference(model=model, test_iter=test_dataloader)

