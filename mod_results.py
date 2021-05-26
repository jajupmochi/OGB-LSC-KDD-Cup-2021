#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:33:18 2021

@author: ljia
"""
num_computed = 300

import numpy as np

dir_path = '/media/ljia/DATA/research-repo/codes/Linlin/OGB-LSC-KDD-Cup-2021/outputs/treelet/'
loaded = np.load(dir_path + 'y_pred_pcqm4m_backup.npz')

y_pred = loaded['y_pred']

y_pred = y_pred = np.concatenate((y_pred[0:num_computed], np.random.randn(len(y_pred) - num_computed) + np.mean(y_pred[0:num_computed])))

from ogb.lsc import PCQM4MEvaluator
evaluator = PCQM4MEvaluator()
evaluator.save_test_submission({'y_pred': y_pred}, dir_path=dir_path)
