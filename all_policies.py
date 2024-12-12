# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: risk_certificates
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import argparse
import secrets
from run_simulations import run_experiments, generate_arm_means
from utils import delete_duplicate_results
import json 
import sys

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed   = 43
    trials = 100
    n_arms = 10
    max_pulls_per_arm = 50
    first_stage_pulls_per_arm = 25
    arm_distribution = 'uniform'
    out_folder = "baseline"
    arm_parameters=  {'uniform_low': 0, 'uniform_high': 1}
    delta = 0.1
    run_all_k = True
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--trials', help='Trials', type=int, default=25)
    parser.add_argument('--n_arms',         '-N', help='Number of arms', type=int, default=10)
    parser.add_argument('--max_pulls_per_arm',        help='Maximum pulls per arm', type=int, default=10)
    parser.add_argument('--first_stage_pulls_per_arm',          help='Number of first stage pulls ', type=int, default=4)
    parser.add_argument('--arm_distribution',          help='Distribution of arms', type=str, default='uniform')
    parser.add_argument('--run_all_k',        help='Maximum pulls per arm', action='store_true')
    parser.add_argument('--delta',        help='Maximum pulls per arm', type=float, default=0.1)
    parser.add_argument('--alpha',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--beta',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_mean_1',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_std_1',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_mean_2',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_std_2',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--uniform_low',        help='Uniform Low', type=float, default=0)
    parser.add_argument('--uniform_high',        help='Uniform High', type=float, default=1)
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='policy_comparison')

    args = parser.parse_args()

    seed = args.seed
    n_arms = args.n_arms
    max_pulls_per_arm = args.max_pulls_per_arm 
    first_stage_pulls_per_arm = args.first_stage_pulls_per_arm
    arm_distribution = args.arm_distribution
    out_folder = args.out_folder
    delta = args.delta 
    alpha = args.alpha 
    beta = args.beta 
    trials = args.trials 
    diff_mean_1 = args.diff_mean_1 
    diff_std_1 = args.diff_std_1 
    diff_mean_2 = args.diff_mean_2 
    diff_std_2 = args.diff_std_2
    uniform_low = args.uniform_low 
    uniform_high = args.uniform_high 
    arm_parameters = {'alpha': alpha, 'beta': beta, 'diff_mean_1': diff_mean_1, 'diff_mean_2': diff_mean_2, 'diff_std_1': diff_std_1, 'diff_std_2': diff_std_2, 'uniform_low': uniform_low, 'uniform_high': uniform_high}
    run_all_k = args.run_all_k

save_name = secrets.token_hex(4)  
# -

random.seed(seed)
np.random.seed(seed)

## Run Policies

# if arm_distribution == 'effect_size':
#     a = list(csv.DictReader(open('../../data/meta_analyses.csv')))
#     arm_parameters['all_effect_sizes'] = [float(i['effect']) for i in a if i['ma.doi'] == '10.1093/gerona/glp082']

arm_means = generate_arm_means(arm_distribution,arm_parameters,n_arms)

experiment_config = {
    'number_arms': n_arms, 
    'sample_size': max_pulls_per_arm*n_arms, 
    'first_stage_size': first_stage_pulls_per_arm*n_arms, 
    'distribution': arm_means, 
    'arm_distribution': arm_distribution, 
    'random_seed': seed+1, 
    'delta': delta,
    'run_all_k': run_all_k, 
    'reward_parameters': arm_parameters, 
    'true_value': np.max(arm_means)
}

# +
all_results = []

for i in range(trials):
    experiment_config['random_seed'] = seed+i
    results = run_experiments(experiment_config)
    all_results.append(results)

# +
aggregate_results = {}
aggregate_results['parameters'] = experiment_config
aggregate_results['parameters']['seed'] = seed 
# import pdb; pdb.set_trace()
for method in all_results[0]:
    aggregate_results[method] = {}
    # just store the best certifictae found
    aggregate_results[method]['certificate'] = [max(i[method]['certificate']) for i in all_results]
    if method in ["ucb", "successive_elimination", 'sample_split_3', 'sample_split_4', 'sample_split_5']:
        aggregate_results[method]['historic_ratio'] = [i[method]['historic_ratio'] for i in all_results]
        aggregate_results[method]['historic_linear'] = [i[method]['historic_linear'] for i in all_results]
    # aggregate_results[method]['certificate_width'] = [i[method]['certificate_width'].tolist() for i in all_results]

# -

for i in aggregate_results:
    if 'certificate' in aggregate_results[i]:
        print(i,np.mean(aggregate_results[i]['certificate']))

# ## Write Data

save_path = "{}/{}.json".format(out_folder,save_name)

delete_duplicate_results(out_folder,"",aggregate_results)

json.dump(aggregate_results,open('results/'+save_path,'w'))
print("Success! Wrote results to {}".format(save_path))
