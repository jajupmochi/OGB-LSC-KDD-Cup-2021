#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:34:54 2021

@author: ljia
"""
### importing OGB-LSC
import numpy as np
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
# from utils import smiles2nxgraph_ogb
# from tqdm import tqdm
import sys
import time
import pickle
import os
# import functools
from utils.utils import get_params, dict_to_tuple, read_current_states, get_subsets, train_valid, simplify_last_model_save
# import json


def get_kernel_matrix(graphs_train, graphs_valid, kernel_options, cur_batch, dir_params, model_save_last=None):
	# Initailize parameters for graph kernel computation.
	node_labels = list(graphs_train[0].nodes[0].keys())
	edge_labels = list(graphs_train[0].edges[(0, 1)].keys())

	from gklearn.kernels import Treelet

	start_time = time.time()

	if cur_batch == 0:
		# Initialize graph kernel.
		graph_kernel = Treelet(parallel=None,#'imap_unordered',
						 n_jobs=None,
						 chunksize=None,
						 normalize=True,
						 verbose=2,
						 save_canonkeys=True,
						 node_labels=node_labels, edge_labels=edge_labels,
						 ds_infos={'directed': False}, **kernel_options)
		# Compute kernel matrix.
		try:
			gram_matrix = graph_kernel.fit_transform(graphs_train)
			Y_matrix = graph_kernel.transform(graphs_valid)
		except:
			raise

		kernel_matrix = np.concatenate((gram_matrix, Y_matrix))
		run_time = time.time() - start_time

	else:
		### Compute new matrix based on the previous batch.

		# previous matrices.
		last_train_size = model_save_last['sim_matrix'].shape[1]
		mat_t1_t1 = model_save_last['sim_matrix'][:last_train_size]
		mat_v1_t1 = model_save_last['sim_matrix'][last_train_size:]
		# previous model.
		model = model_save_last['model']
		attrs_last = {'_X_diag': model._X_diag, '_Y_diag': model._Y_diag,
# 			  '_graphs': model._graphs, '_Y': model._Y,
			  '_canonkeys': model._canonkeys, '_Y_canonkeys': model._Y_canonkeys}

		# Compute kernel matrices between train1 and current data.
		try:
			mat_t2_t1 = model.transform(graphs_train)
			mat_v2_t1 = model.transform(graphs_valid)
		except:
			raise

		# Compute kernel matrx between valid1 and train2.
		model._X_diag = attrs_last['_Y_diag']
		model._canonkeys = attrs_last['_Y_canonkeys']
		try:
			mat_t2_v1 = model.transform(graphs_train) # @todo: here, dummy labels have already been added into train2. This may cause errors when v/e don't have labels. # @fixme
		except:
			raise

		# Compute kernel matrices between current data.
		graph_kernel = Treelet(parallel=None,#'imap_unordered',
						 n_jobs=None,
						 chunksize=None,
						 normalize=True,
						 verbose=2,
						 save_canonkeys=True,
						 node_labels=node_labels, edge_labels=edge_labels,
						 ds_infos={'directed': False}, **kernel_options)
		try:
			mat_t2_t2 = graph_kernel.fit_transform(graphs_train) # @todo: same as previous.
			mat_v2_t2 = graph_kernel.transform(graphs_valid)
		except:
			raise

		# Concatenate matrices.
		col1 = np.concatenate((mat_t1_t1, mat_t2_t1, mat_v1_t1, mat_v2_t1), axis=0)
		col2 = np.concatenate((mat_t2_t1.T, mat_t2_t2, mat_t2_v1.T, mat_v2_t2), axis=0)
		kernel_matrix = np.concatenate((col1, col2), axis=1)

		# Update model.
		graph_kernel._X_diag = np.concatenate((attrs_last['_X_diag'], graph_kernel._X_diag))
		graph_kernel._Y_diag = np.concatenate((attrs_last['_Y_diag'], graph_kernel._Y_diag))
		graph_kernel._canonkeys = attrs_last['_canonkeys'] + graph_kernel._canonkeys
		graph_kernel._Y_canonkeys = attrs_last['_Y_canonkeys'] + graph_kernel._Y_canonkeys
# 		graph_kernel._graphs = attrs_last['_graphs'] + graph_kernel._graphs
# 		graph_kernel._Y = attrs_last['_Y'] + graph_kernel._Y

		run_time = time.time() - start_time + model_save_last['mat_runtime']


	# Remove graphs in model to save space.
	graph_kernel._graphs = []
	graph_kernel._Y = []

	return kernel_matrix, run_time, graph_kernel


def param_lists():
	import functools
	from sklearn.model_selection import ParameterGrid
	from gklearn.utils.kernels import gaussiankernel, polynomialkernel
	from gklearn.model_learning import dichotomous_permutation

	gkernels = [functools.partial(gaussiankernel, gamma=1 / ga)
#            for ga in np.linspace(1, 10, 10)]
            for ga in dichotomous_permutation(np.logspace(0, 10, num=11, base=10))]
            # for ga in np.logspace(0, 6, num=3, base=10)]
	pkernels = [functools.partial(polynomialkernel, d=d, c=c)
			 for d in dichotomous_permutation(range(1, 5))
# 			 for d in range(1, 8, 3)
             for c in dichotomous_permutation(np.logspace(0, 10, num=11, base=10))]
# 			 for c in np.logspace(0, 6, num=3, base=10)]
	param_grid_precomputed = {'sub_kernel': pkernels + gkernels}
# 	param_grid = {'alpha': np.logspace(-10, 10, num=21, base=10)}
	param_grid = {'alpha': dichotomous_permutation(np.logspace(-10, 10, num=21, base=10))}

	return list(ParameterGrid(param_grid_precomputed)), list(ParameterGrid(param_grid))


def loop_batch(batch_total, states, param_idx_map, dir_params, params_out, param_list, idx_params, name_pred, fn_states):
	model_save_last = None
	for cur_batch in range(0, batch_total):

		# Check the params for the current batch.
		if cur_batch in states['param_idx_map']:
			if states['param_idx_map'][cur_batch] == param_idx_map['in']:
				continue

		print('\n# of batch:', str(cur_batch))

		# automatic dataloading and splitting
		dataset = PCQM4MDataset(root='dataset/', only_smiles=True)
		graphs_train, graphs_valid, y_train, y_valid = get_subsets(dataset, batch_total=batch_total, batch=cur_batch, stratified=True)

		### automatic evaluator. takes dataset name as input
		evaluator = PCQM4MEvaluator()

		# Read previous batch data.
		if cur_batch > 0 and model_save_last is None:
			fn_model_last = os.path.join(dir_params, 'model.batch' + str(cur_batch - 1) + '.pkl')
			with open(fn_model_last, 'rb') as f:
				model_save_last = pickle.load(f)

		### Start evaluation.
		cur_state, model_save = train_valid(graphs_train, y_train, graphs_valid, y_valid, evaluator, params_out, param_list, param_idx_map, dir_params, idx_params, cur_batch, get_kernel_matrix, name_pred, states, model_save_last)

		### Save states.
		states['batch'] = cur_batch
		states['state'] = cur_state
		states['param_idx_map'][cur_batch] = param_idx_map['in']
		with open(fn_states, 'wb') as f:
			pickle.dump(states, f)


		### simplify the previous model data, save and overwrite.
		if model_save_last is not None:
			simplify_last_model_save(model_save_last, name_pred, dir_params, cur_batch)

		model_save_last = model_save


def main():

	# global Training settings.
	batch_total = 1000
	ratio_subset = 1 / batch_total
	model_name = 'Treelet'
	dir_root = 'outputs/' + model_name + '.stratifiled'
	name_pred = 'KernelRidge'

	# Get parameter grids.
	param_list_precomputed, param_list, param_idx_map = get_params(dir_root, param_lists)


	### main loop.
	for params_out in param_list_precomputed[:]:
		idx_params = param_idx_map['out'][dict_to_tuple(params_out)]

		print('\n# of outer parameters:', str(idx_params))

		# Create folder for this parameter setting.
		dir_params = os.path.join(dir_root, str(idx_params))
		os.makedirs(dir_params, exist_ok=True)

		# load current states.
		fn_states = os.path.join(dir_params, 'cur_states.pkl')
		states = read_current_states(dir_params)
		if states['state'] == 'mat_error' or param_idx_map['in'] == states['param_idx_map']['all']:
			continue


		### batch loop.
		loop_batch(batch_total, states, param_idx_map, dir_params, params_out, param_list, idx_params, name_pred, fn_states)


		# Update inner params if all batches are finished.
		states['param_idx_map']['all'] = param_idx_map['in']
		with open(fn_states, 'wb') as f:
			pickle.dump(states, f)


def run_task(model_name, idx_params):
	# global Training settings.
	batch_total = 1000
	ratio_subset = 1 / batch_total
	dir_root = 'outputs/' + model_name + '.stratifiled'
	name_pred = 'KernelRidge'

	# Get parameter grids.
	param_list_precomputed, param_list, param_idx_map = get_params(dir_root, param_lists)


	### main.
	print('\n# of outer parameters:', str(idx_params))

	params_out = param_idx_map['out_r'][idx_params]

	# Create folder for this parameter setting.
	dir_params = os.path.join(dir_root, str(idx_params))
	os.makedirs(dir_params, exist_ok=True)

	# load current states.
	fn_states = os.path.join(dir_params, 'cur_states.pkl')
	states = read_current_states(dir_params)


	### batch loop.
	loop_batch(batch_total, states, param_idx_map, dir_params, params_out, param_list, idx_params, name_pred, fn_states)


	# Update inner params if all batches are finished.
	states['param_idx_map']['all'] = param_idx_map['in']
	with open(fn_states, 'wb') as f:
		pickle.dump(states, f)


if __name__ == '__main__':
	test = True
	if test:
		run_task('Treelet', 0)

	if len(sys.argv) > 1:
		run_task(sys.argv[1], int(sys.argv[2]))
	else:
		main()