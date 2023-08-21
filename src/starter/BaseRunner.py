#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BaseRunner.py
@Author     ：Heywecome
@Date       ：2023/8/21 10:11 
@Description：todo
"""
import numpy as np
import torch
import torch.optim as optim

from time import time

from torch.nn import CrossEntropyLoss
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
        parser.add_argument('--early_stop', type=int, default=10,
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

    def train(self, epoch, model, loss_fn, optimizer, train_dataloader):
        model.train()
        loss_list = []
        train_acc = 0
        train_total = 0
        loss_fn.to(self.device)
        bar = tqdm(train_dataloader, total=len(train_dataloader))  # 配置进度条
        for idx, (input, target) in enumerate(bar):
            input = input.to(self.device)
            target = target.to(self.device)
            output = model(input)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.cpu().item())
            optimizer.step()
            # 准确率
            output_max = output.max(dim=-1)  # 返回最大值和对应的index
            pred = output_max[-1]  # 最大值的index
            train_acc += pred.eq(target).cpu().float().sum().item()
            train_total += target.shape[0]
        acc = train_acc / train_total
        print("train epoch:{}  loss:{:.6f} acc:{:.5f}".format(epoch, np.mean(loss_list), acc))
        return acc, np.mean(loss_list)

    def test(self, model, loss_fn, data_loader, stage):
        model.eval()
        loss_list = []
        acc = 0
        total = 0
        loss_fn.to(self.device)
        with torch.no_grad():
            for input, target in data_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                output = model(input)
                loss = loss_fn(output, target)
                loss_list.append(loss.item())
                # Accuracy
                output_max = output.max(dim=-1)
                pred = output_max[-1]
                acc += pred.eq(target).cpu().float().sum().item()
                total += target.shape[0]

        avg_loss = np.mean(loss_list)
        avg_acc = acc / total
        print("{} loss:{:.6f}, acc:{}".format(stage, avg_loss, avg_acc))
        return avg_acc, avg_loss

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
        test_loss_list = []

        for epoch in range(self.epoch):
            train_acc, train_loss = self.train(epoch, model, loss_fn, optimizer, train_dataloader)
            val_acc, val_loss = self.test(model, loss_fn, val_dataloader, 'val')  # Using val_dataloader for validation
            # test_acc, test_loss = test(model, loss_fn, test_dataloader, 'test')  # Test
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)  # Adding validation loss
            # test_loss_list.append(test_loss)

            if val_acc > best_acc:  # Using validation accuracy for early stopping and saving the best model
                best_acc = val_acc
                # self.save_model(model, best_acc)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt > self.early_stop:
                print("Early stopping triggered.")
                break

        # plot_learning_curve(train_loss_list, val_loss_list, test_loss_list)  # Plotting validation loss as well

        # Print the best validation accuracy
        print("Best Validation Accuracy: {:.5f}".format(best_acc))

        # After training, get the test results
        final_test_acc, final_test_loss = test(model, loss_fn, test_dataloader, 'test')
        print("Final Test Accuracy: {:.5f}. Test Loss: {:.6f}".format(final_test_acc, final_test_loss))