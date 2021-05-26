#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:05:52 2021

@author: ljia
"""
import os
import re


def param_lists():
	import functools
	import numpy as np
	from sklearn.model_selection import ParameterGrid
	from gklearn.utils.kernels import gaussiankernel, polynomialkernel
	from gklearn.model_learning import dichotomous_permutation

	gkernels = [functools.partial(gaussiankernel, gamma=1 / ga)
#            for ga in np.linspace(1, 10, 10)]
            for ga in dichotomous_permutation(np.logspace(0, 10, num=11, base=10))]
            # for ga in np.logspace(0, 6, num=3, base=10)]
	pkernels = [functools.partial(polynomialkernel, d=d, c=c)
			 for d in dichotomous_permutation(range(1, 11))
# 			 for d in range(1, 8, 3)
             for c in dichotomous_permutation(np.logspace(0, 10, num=11, base=10))]
# 			 for c in np.logspace(0, 6, num=3, base=10)]
	param_grid_precomputed = {'sub_kernel': pkernels + gkernels}
# 	param_grid = {'alpha': np.logspace(-10, 10, num=21, base=10)}
	param_grid = {'alpha': dichotomous_permutation(np.logspace(-10, 10, num=21, base=10))}

	return list(ParameterGrid(param_grid_precomputed)), list(ParameterGrid(param_grid))


def get_job_script(model_name, idx_params):
	script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name="ogb.""" + model_name + r"." + idx_params + r""""
#SBATCH --partition=court
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_ogb.""" + model_name + r"." + idx_params + r""".txt"
#SBATCH --error="errors/error_ogb.""" + model_name + r"." + idx_params + r""".txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
##SBATCH --mem-per-cpu=4000
#SBATCH --mem=40000

srun hostname
srun cd /home/2019015/ljia02/codes/OGB-LSC-KDD-Cup-2021
srun python3 main_treelet.py """ + model_name + r" " + idx_params
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return script


def check_task_status(save_dir, param_idx_map, *params):
	str_task_id = '.' + '.'.join(params)

# 	# Check if the task is in out of memeory or out of space lists or missing labels.
# 	if params in OUT_MEM_LIST or params in OUT_TIME_LIST or params in MISS_LABEL_LIST:
# 		return True

	# Check if the task is running or in queue of slurm.
	command = 'squeue --user $USER --name "ogb' + str_task_id + '" --format "%.2t" --noheader'
	stream = os.popen(command)
	output = stream.readlines()
	if len(output) > 0:
		return True

# 	# Check if there are more than 10 tlong tasks running.
# 	command = 'squeue --user $USER --partition tlong --noheader'
# 	stream = os.popen(command)
# 	output = stream.readlines()
# 	if len(output) >= 10:
# 		return True


	# Check if the results are already computed.
	# load current states.
	states = read_current_states(dir_params)
	if states['state'] == 'mat_error' or param_idx_map['in'] == states['param_idx_map']['all']:
		return True

	return False


if __name__ == '__main__':
	from utils.utils import get_params, dict_to_tuple, read_current_states

	# global Training settings.
	model_name = 'Treelet'
	dir_root = 'outputs/' + model_name

	os.makedirs(dir_root, exist_ok=True)
	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	# Get parameter grids.
	param_list_precomputed, param_list, param_idx_map = get_params(dir_root, param_lists)

	### main loop.
	for params_out in param_list_precomputed[:]:
		idx_params = str(param_idx_map['out'][dict_to_tuple(params_out)])

		if int(idx_params) not in [8,  45,   2,   9,   5,  67,   6, 111,  76,  10,   1, 119, 112, 71,  72, 118, 120, 116, 115,  54,  75,   4, 114]:
			continue

		print('\n# of outer parameters:', idx_params)

		# Create folder for this parameter setting.
		dir_params = os.path.join(dir_root, idx_params)
		os.makedirs(dir_params, exist_ok=True)

		if False == check_task_status(dir_root, param_idx_map, model_name, idx_params):
			job_script = get_job_script(model_name, idx_params)
			command = 'sbatch <<EOF\n' + job_script + '\nEOF'
	# 			print(command)
			os.system(command)
	# 		os.popen(command)
	# 		output = stream.readlines()