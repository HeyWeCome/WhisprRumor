#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：__init__.py.py
@Author     ：Heywecome
@Date       ：2023/8/21 09:22 
@Description：todo
"""
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]
