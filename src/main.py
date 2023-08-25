#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：main.py
@Author     ：Heywecome
@Date       ：2023/8/21 08:54 
@Description：Startup class for projects
"""
import argparse
import logging
import os
import sys
import torch

from starter import *
from utils.utils import init_random_seed, get_time, format_arg_str, check_dir
from model.general import *


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES, default for CUDA')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--model_path', type=str, default='',
                        help='Address where the model is stored')
    parser.add_argument('--random_seed', type=int, default=2023,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
               'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
    logging.info(format_arg_str(args, exclude_lst=exclude))

    # Init the random seed
    init_random_seed()

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    # Define model
    model = model_class(args).to(args.device)
    # logging.info('#params: {}'.format(model.count_variables()))
    logging.info(model)

    # Get train, val, test dataset
    loader = loader_class(args)
    train_iter, test_iter, val_iter = loader.get_dataloader()

    # train
    runner = runner_class(args)
    runner.fit(model, train_iter, val_iter, test_iter)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model', type=str, default='BertBase', help='Choose a model to run.')
    init_parser.add_argument('--data_loader', type=str, default='BertLoader', help='Choose a dataloader object.')
    init_parser.add_argument('--data_runner', type=str, default='BaseRunner', help='Choose a runner object.')
    init_args, init_extras = init_parser.parse_known_args()
    model_class = eval('{0}.{0}'.format(init_args.model))
    loader_name = init_args.data_loader
    loader_class = eval('{0}.{0}'.format(loader_name))
    runner_name = init_args.data_runner
    runner_class = eval('{0}.{0}'.format(runner_name))

    # Args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default=init_args.model)
    parser = parse_global_args(parser)
    parser = model_class.parse_model_args(parser)
    parser = loader_class.parse_data_args(parser)
    parser = runner_class.parse_runner_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model, args.dataset, str(args.random_seed)]
    for arg in ['lr', 'l2'] + model_class.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model, log_file_name)

    check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()