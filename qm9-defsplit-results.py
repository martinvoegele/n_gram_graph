import os
import numpy as np

from n_gram_graph.dataset_specification import *


qm9_task_list = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 
    'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom',
    'u0', 'u298', 'h298', 'g298']
qm9_conversion = np.array([
    1,1,27.2114,27.2114,27.2114,1,27211.4,1,
    0.04336414,0.04336414,0.04336414,0.04336414,
    27.2114,27.2114,27.2114,27.2114])

def read_results(fname):
    with open(fname) as file:
        res_lines = (file.readlines() [-5:-1])
    results = []
    for rl in res_lines:
        results.append(rl.split(' ')[-1].rstrip())
    return results

def print_results(transform=True):
    for t, task in enumerate(qm9_task_list):
        mae_list = []
        for i in range(5):
            result_file = 'output/n_gram_xgb/defsplit-seed'+str(i)+'/'+task+'.out'
            pearson, r2, rmse, mae = read_results(result_file)
            new_mae = float(mae)
            if transform:
                new_mae *= qm9_conversion[t]
            mae_list.append(new_mae)
        print('%9s:  %6.3f +/- %5.3f'%(task, np.mean(mae_list), np.std(mae_list)))

print('Transformed Results')
print_results(transform=True)

print('Un-Transformed Results')
print_results(transform=False)

