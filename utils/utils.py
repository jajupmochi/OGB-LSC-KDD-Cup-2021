#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:32:24 2021

@author: ljia
"""
import os
import sys
# from tqdm import tqdm
import pickle
import time
import functools
import numpy as np
from utils.mol import smiles2nxgraph_ogb
from gklearn.utils import get_iters
from gklearn.model_learning import dichotomous_permutation


def simplify_last_model_save(model_save_last, name_pred, dir_params, cur_batch):
	del model_save_last['model']
	del model_save_last['sim_matrix']
	del model_save_last['y_train']
	del model_save_last['y_valid']
	del model_save_last['prediction'][name_pred]['best_y_pred']
	del model_save_last['prediction'][name_pred]['runtimes']
	del model_save_last['prediction'][name_pred]['train_perfs']
	del model_save_last['prediction'][name_pred]['valid_perfs']
	del model_save_last['prediction'][name_pred]['y_preds_train']
	del model_save_last['prediction'][name_pred]['y_preds_valid']

	fn_model_last = os.path.join(dir_params, 'model.batch' + str(cur_batch - 1) + '.pkl')
	with open(fn_model_last, 'wb') as f:
		pickle.dump(model_save_last, f)


def split_by_values(lst, values):
    indices = [i for i, x in enumerate(lst) if x >= values]
    for start, end in zip([0, *indices], [*indices, len(lst)]):
        yield lst[start:end+1]


def get_subindex_stratified(y, batch_total=1000, batch=0, n_split_base=100):
	# Get training subset.
	y_sort = np.sort(y, kind='stable')
	total_idx = np.argsort(y, kind='stable')

	# Split stratifiedly.
	if  len(y_sort) < n_split_base:
		n_split_base = len(y_sort)
	max_, min_ = y_sort[-1], y_sort[0]
	interval = (max_ - min_) / n_split_base
	sprt_vals = [min_ + interval * i for i in range(1, n_split_base)]
	sprt_idx = np.searchsorted(y_sort, sprt_vals, side='right', sorter=None)

	# Get splitted indices.
	total_idx_sub = []
	batch_size = int(len(y_sort) / batch_total)
	for start, end in zip([0, *sprt_idx], [*sprt_idx, len(y_sort)]):
		sub_size = end - start
		if sub_size < 1:
			continue

		subbatch_size = max(int(sub_size / len(y_sort) * batch_size), 1)
		n_subbatch = min(int(sub_size / subbatch_size), batch_total)
		if batch >= n_subbatch:
			continue
		offset = dichotomous_permutation(range(0, n_subbatch))[batch]
		total_idx_sub += total_idx[start:end][offset::n_subbatch].tolist()

	return total_idx_sub


def get_subsets_stratified(dataset, batch_total=1000, batch=0, n_split_base=100):
	split_idx = dataset.get_idx_split()

	# Get training subset.
	y_train = [dataset[i][1] for i in split_idx['train']]
	train_idx_sub = get_subindex_stratified(y_train, batch_total=batch_total, batch=batch, n_split_base=n_split_base)
	y_train_sub = [y_train[i] for i in train_idx_sub]
	x_train_sub = [dataset[i][0] for i in train_idx_sub]
	x_train_sub = [smiles2nxgraph_ogb(smiles) for smiles in get_iters(x_train_sub, desc='get training graphs', file=sys.stdout)]

	# Get valid subset.
	y_valid = [dataset[i][1] for i in split_idx['valid']]
	valid_idx_sub = get_subindex_stratified(y_valid, batch_total=batch_total, batch=batch, n_split_base=n_split_base)
	y_valid_sub = [y_valid[i] for i in valid_idx_sub]
	x_valid_sub = [dataset[i][0] for i in valid_idx_sub]
	x_valid_sub = [smiles2nxgraph_ogb(smiles) for smiles in get_iters(x_valid_sub, desc='get valid graphs', file=sys.stdout)]

	return x_train_sub, x_valid_sub, np.array(y_train_sub), np.array(y_valid_sub)


def get_subsets(dataset, batch_total=1000, batch=0, stratified=False, n_split_base=100):
	if stratified:
		return get_subsets_stratified(dataset, batch_total=1000, batch=0)

	else:
		offset = dichotomous_permutation(range(0, batch_total))[batch]

		split_idx = dataset.get_idx_split()

		# Get training subset.
		y_train = [dataset[i][1] for i in split_idx['train']]
		y_train_sort = np.sort(y_train, kind='stable')
		train_idx = np.argsort(y_train, kind='stable')
	# 	y_train_sort = y_train[train_idx]
		y_train_sub = y_train_sort[offset::batch_total]
		train_idx_sub = train_idx[offset::batch_total]
		x_train_sub = [dataset[i][0] for i in train_idx_sub]
	# 	print(train_idx_sub)
		x_train_sub = [smiles2nxgraph_ogb(smiles) for smiles in get_iters(x_train_sub, desc='get training graphs', file=sys.stdout)]

		# Get valid subset.
		y_valid = [dataset[i][1] for i in split_idx['valid']]
		y_valid_sort = np.sort(y_valid, kind='stable')
		valid_idx = np.argsort(y_valid, kind='stable')
	# 	y_valid_sort = y_valid[valid_idx]
		y_valid_sub = y_valid_sort[offset::batch_total]
		valid_idx_sub = valid_idx[offset::batch_total]
		x_valid_sub = [dataset[i][0] for i in valid_idx_sub]
		x_valid_sub = [smiles2nxgraph_ogb(smiles) for smiles in get_iters(x_valid_sub, desc='get valid graphs', file=sys.stdout)]

		return x_train_sub, x_valid_sub, np.array(y_train_sub), np.array(y_valid_sub)


def read_current_states(root):
	file_name = os.path.join(root, 'cur_states.pkl')

	if os.path.isfile(file_name):
		with open(file_name, 'rb') as f:
			states = pickle.load(f)
	else:
		states = {'batch': -1, 'ratio': 0.001, 'param_idx_map': {'all': {}}, 'state': None}

	return states


def read_param_index_map(root):
	file_name = os.path.join(root, 'index_param_map.pkl')

	if os.path.isfile(file_name) and os.path.getsize(file_name) != 0:
		with open(file_name, 'rb') as f:
			return pickle.load(f)
	else:
		return {'out': {}, 'out_r': {}, 'in': {}}


def dict_to_tuple(dict_):
	# @todo: to update (for sp kernel).
	tuple_ = []
	for key, val in dict_.items():
		if isinstance(val, functools.partial):
			tuple_.append(tuple((key, val.func.__name__, tuple(val.keywords.items()))))
		else:
			tuple_.append(tuple((key, val)))

	return tuple(tuple_)


def get_params(root, param_lists):
	param_list_precomputed, param_list = param_lists()

	pp_idx_map = read_param_index_map(root)

	# Parameters precomputed.
	old_idx1, cur_idx1 = len(pp_idx_map['out']), len(pp_idx_map['out'])
	keys = pp_idx_map['out'].keys()
	for param in param_list_precomputed:
		param_t = dict_to_tuple(param)
		if param_t not in keys:
			pp_idx_map['out'][param_t] = cur_idx1
			pp_idx_map['out_r'][cur_idx1] = param
			cur_idx1 += 1

	# Parameters inner loop.
	old_idx2, cur_idx2 = len(pp_idx_map['in']), len(pp_idx_map['in'])
	keys = pp_idx_map['in'].keys()
	for param in param_list:
		param_t = dict_to_tuple(param)
		if param_t not in keys:
			pp_idx_map['in'][param_t] = cur_idx2
			cur_idx2 += 1

	if cur_idx1 > old_idx1 or cur_idx2 > old_idx2:
		# Save new params to file.
		os.makedirs(root, exist_ok=True)
		file_name = os.path.join(root, 'index_param_map.pkl')
		with open(file_name, 'wb') as f:
			pickle.dump(pp_idx_map, f)

	return param_list_precomputed, param_list, pp_idx_map


def inner_model_selection(param_list, param_idx_map, sim_matrix, y_train, y_valid, size_train, evaluator, model_save, fn_model, cur_batch, dir_params, name_pred, states, model_save_last=None):
	from sklearn.kernel_ridge import KernelRidge

	if cur_batch == 0:
		model_save['y_train'] = y_train
		model_save['y_valid'] = y_valid
	else:
		model_save['y_train'] = np.concatenate((model_save_last['y_train'], y_train))
		model_save['y_valid'] = np.concatenate((model_save_last['y_valid'], y_valid))


	prediction = {'train_perfs': [None] * len(param_list),
			'valid_perfs': [None] * len(param_list),
# 			'models': [None] * len(param_list),
			'y_preds_train': [None] * len(param_list),
			'y_preds_valid': [None] * len(param_list),
			'runtimes': [None] * len(param_list),
			'total_runtime': time.time()}
	idx_list = []
	for params_in in param_list: # @todo: skip previous computed params.
		index_in = param_idx_map['in'][dict_to_tuple(params_in)]
		prediction['runtimes'][index_in] = time.time()
		idx_list.append(index_in)

		# Do prediction.
		# @todo: skip previous batches.
		kr = KernelRidge(kernel='precomputed', **params_in)
		kr.fit(sim_matrix[:size_train][:], model_save['y_train'])

		# predict on the train, validation and test set
		y_pred_train = kr.predict(sim_matrix[:size_train][:])
		y_pred_valid = kr.predict(sim_matrix[size_train:][:])

		# Evaluate.
		input_dict_train = {"y_true": model_save['y_train'], "y_pred": y_pred_train}
		input_dict_valid = {"y_true": model_save['y_valid'], "y_pred": y_pred_valid}

		prediction['train_perfs'][index_in] = evaluator.eval(input_dict_train)['mae']
		prediction['valid_perfs'][index_in] = evaluator.eval(input_dict_valid)['mae']
# 		prediction['models'][index_in] = kr
		prediction['y_preds_train'][index_in] = y_pred_train
		prediction['y_preds_valid'][index_in] = y_pred_valid

		prediction['runtimes'][index_in] = time.time() - prediction['runtimes'][index_in]

	model_save['prediction'] = {name_pred: prediction}


	### Find the best model.
	best_val_perf = np.amin(prediction['valid_perfs'])
	best_params_idx = np.where(prediction['valid_perfs'] == best_val_perf)
	best_params_idx = [i[0] for i in best_params_idx]
	best_params_in = [param_list[idx_list.index(i)] for i in best_params_idx]
	best_train_perf = [prediction['train_perfs'][i] for i in best_params_idx]
	best_y_pred = [prediction['y_preds_valid'][i] for i in best_params_idx]

	prediction['total_runtime'] = time.time() - prediction['total_runtime']


	### Save and print.
	# Update states.
	best_results = {'best_val_perf': best_val_perf,
			  'best_train_perf': best_train_perf,
			 'best_params_idx': best_params_idx, 'best_params_in': best_params_in,
			 'best_y_pred': best_y_pred}
	results = {'mat_runtime': model_save['mat_runtime'], 'prediction': {name_pred: best_results}}
	if 'results' in states:
		if cur_batch in states['results']:
			states['results'][cur_batch]['prediction'][name_pred] = best_results
		else:
			states['results'][cur_batch] = results
	else:
		states['results'] = {cur_batch: results}

	# Save model.
	model_save['prediction'][name_pred].update(best_results)
	with open(fn_model, 'wb') as f:
		pickle.dump(model_save, f) # @todo: add instead of overwrite.
	print('kernel matrix computation runtime:', model_save['mat_runtime'])
	print('prediction runtime:', prediction['total_runtime'])
	print('total runtime:', model_save['mat_runtime'] + prediction['total_runtime'])
	print('best valid perf:', best_val_perf)


def train_valid(graphs_train, y_train, graphs_valid, y_valid, evaluator, params_out, param_list, param_idx_map, dir_params, index_out, cur_batch, get_kernel_matrix, name_pred, states, model_save_last=None):

	size_train = len(graphs_train)
	if cur_batch > 0:
		size_train += model_save_last['sim_matrix'].shape[1]
# 	size_valid = len(graphs_valid)
	fn_model = os.path.join(dir_params, 'model.batch' + str(cur_batch) + '.pkl')


	### Compute Gram matrix.
	if os.path.isfile(fn_model) and os.path.getsize(fn_model) != 0:
		# Kernel matrix has already been computed.
		with open(fn_model, 'rb') as f:
			model_save = pickle.load(f)

		if 'sim_matrix' in model_save:
			sim_matrix = model_save['sim_matrix']
			inner_model_selection(param_list, param_idx_map, sim_matrix, y_train, y_valid, size_train, evaluator, model_save, fn_model, cur_batch, dir_params, name_pred, states, model_save_last)
			return None, model_save

		else:
			raise Exception('sim_matrix not found while calling train_valid().')

	else:
		# Start from scratch.
		try:
			sim_matrix, run_time, model = get_kernel_matrix(graphs_train, graphs_valid, params_out, cur_batch, dir_params, model_save_last)
		except Exception as e:
			print('Skipped due to the following error:', e) # @todo
			return 'mat_error', None
		else:
			# Save matrix to file.
			model_save = {'sim_matrix': sim_matrix, 'mat_runtime': run_time, 'model': model}
			with open(fn_model, 'wb') as f:
				pickle.dump(model_save, f) # @todo: already loaded in `get_kernel_matrix()`.

			### Predict.
			inner_model_selection(param_list, param_idx_map, sim_matrix, y_train, y_valid, size_train, evaluator, model_save, fn_model, cur_batch, dir_params, name_pred, states, model_save_last)
			return None, model_save