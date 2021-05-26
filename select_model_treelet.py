#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:44:29 2021

@author: ljia
"""
import numpy as np
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
# from utils import smiles2nxgraph_ogb
from tqdm import tqdm
import sys
import time
import pickle
import os
# import functools
from utils.utils import get_params, dict_to_tuple, read_current_states, get_subsets, train_valid, simplify_last_model_save


def main():

	# global Training settings.
	batch_total = 1000
	ratio_subset = 1 / batch_total
	model_name = 'Treelet'
	dir_root = 'outputs/CRIANN/' + model_name
	name_pred = 'KernelRidge'
	k = 10

	# Get param_idx_map.
	file_name = os.path.join(dir_root, 'index_param_map.pkl')
	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			param_idx_map = pickle.load(f)
	else:
		raise FileExistsError('file "index_param_map.pkl" does not exist.')

	### loop param_idx_map.
	valid_perfs = {}
	best_val_perfs = {}
	for idx_params, params_out in tqdm(param_idx_map['out_r'].items(), desc='params', file=sys.stdout):
# 		print(idx_params)
# 		if idx_params < 25:
# 			continue

		fn_states = os.path.join(dir_root, str(idx_params) + '/cur_states.pkl')
		if not os.path.isfile(fn_states):
			continue

		with open(fn_states, 'rb') as f:
			states = pickle.load(f)
			max_batch = states['batch']

		cur_best_val_perfs = []
		for batch in range(0, max_batch + 1):
			fn_model = os.path.join(dir_root, str(idx_params) + '/model.batch' + str(batch) + '.pkl')
			if not os.path.isfile(fn_model):
				raise FileExistsError('file "model.batch' + str(batch) + '.pkl" for parameter ' + str(idx_params) + ' does not exist.')

			with open(fn_model, 'rb') as f:
				model_save = pickle.load(f)
			cur_best_val_perfs.append(model_save['prediction'][name_pred]['best_val_perf'])
		valid_perfs[idx_params] = cur_best_val_perfs
		best_val_perfs[idx_params] = cur_best_val_perfs[-1]

	print('valid_perfs:')
	for item in valid_perfs.items():
		print(item)

	### get best perfs.
	best_val_perfs = {k: v for k, v in sorted(best_val_perfs.items(), key=lambda item: item[1])}
# 	k_best_vals = more_itertools.take(k, best_val_perfs.items())
	k_best_vals = {k: best_val_perfs[k] for k in list(best_val_perfs.keys())[:k]}

	print('\nk_best_vals:', k_best_vals)

	return best_val_perfs, k_best_vals, param_idx_map


if __name__ == '__main__':
	best_val_perfs, k_best_vals, param_idx_map = main()

	# Print best valid params.
	best_val_params = {k: dict_to_tuple(param_idx_map['out_r'][k]) for k in best_val_perfs.keys()}
	print('\nbest_val_params:')
	for item in best_val_params.items():
		print(item)

	# Get the idx and params that does not work (cause errors).
	invalid_idx = []
	invalid_params = []
	for idx_params, params_out in param_idx_map['out_r'].items():
		if idx_params not in best_val_perfs.keys():
			invalid_idx.append(idx_params)
			invalid_params.append(dict_to_tuple(params_out))
	print('\ninvalid_params:')
	for i, v in enumerate(invalid_idx):
		print(v, ': ', invalid_params[i])