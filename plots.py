import sys
sys.path.append('/Users/scortesg/Documents/weakly_adaptivity')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
import scipy.stats as stats 
from utils import get_results_matching_parameters, aggregate_data, aggregate_normalize_data
import seaborn as sns
from scipy.stats import beta as beta_function
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use('default')


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.rcParams['savefig.bbox'] = 'tight'

colors_by_method = {}
non_adaptive_methods = ['','prior','k_1','one_stage','fixed_k']
# our_methods = ['sample_split','sample_split_total','prior']
our_methods = ['sample_split_3', 'sample_split_4', 'sample_split_5']
adaptive_methods = ['two_stage_thompson','two_stage_successive_elimination','successive_elimination','ucb']

greys = plt.cm.Greys(np.linspace(0.2, 1, len(non_adaptive_methods)+1))
vidris = plt.cm.coolwarm(np.linspace(0, 0.4, len(our_methods)))
wistia = plt.cm.coolwarm(np.linspace(0.6, 0.9, len(adaptive_methods)))

for i,m in enumerate(non_adaptive_methods):
    colors_by_method[m] = greys[i]

for i,m in enumerate(our_methods):
    colors_by_method[m] = vidris[i]

for i,m in enumerate(adaptive_methods):
    colors_by_method[m] = wistia[i]

colors_by_method['omniscient'] = np.array([0.9,0.05,0.05,1.0])
colors_by_method['median_effect'] = np.array([0.2, 0.6, 0.2, 1.0])

shapes_by_method = {}
all_shapes = ['P','o','v','D','s','x','^','<','>']
for i,m in enumerate(non_adaptive_methods+our_methods+['omniscient']+adaptive_methods):
    shapes_by_method[m] = all_shapes[i%len(all_shapes)]


fig, axs = plt.subplots(1, 2, figsize=(12, 2))  # Two subplots for the metrics
method_names = ['ucb', 'successive_elimination']
nice_names = ["UCB", "Successive Elimination"]
width = 0.1

baseline_params = {'sample_size': 500, 'number_arms': 10}
dataset = "baseline"
first_stage_sizes = [150]
first_stage_percent = ["30%"]
metrics = ["historic_ratio", "historic_linear"]
x_locations = [] 
shift = 0
results = get_results_matching_parameters(dataset, "", baseline_params)
aggregated_results = aggregate_data(results)

for i, metric in enumerate(metrics):
    
    for k, method in enumerate(method_names):
        historic_data = aggregated_results[method][metric]  # Get data for the current method and metric
        shape = shapes_by_method[method]  # Get shape for current method
        # import pdb; pdb.set_trace()
        # Plot the data
        axs[i].plot(
            historic_data[0],
            label=nice_names[k],
            marker=shape,
            linestyle='-',
            color=colors_by_method[method],
        )
        axs[i].axhline(y=0.005116293838899756, color=colors_by_method["prior"], linestyle='--', linewidth=2, label='Bayesian')

    # Set axis titles, ticks, and limits
    axs[i].set_title(metric.replace("_", " ").capitalize(), fontsize=14)  # Use the metric as the title
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    # axs[i].set_ylim([0.75, 0.95])  # Set y-axis limits (adjust as needed)
    # axs[i].set_xlabel(first_stage_percent[0], fontsize=14)
    # axs[i].set_xticks([])  # Remove x-axis ticks
    # axs[i].set_yticks([0.8, 0.9])  # Add specific y-axis ticks

# Add y-axis label to the first subplot
axs[0].set_ylabel("Normalized Cert.", fontsize=14)

# Add legend across both subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=14)

# Add a shared xlabel below the subplots
fig.supxlabel("Number of stages", fontsize=14, x=0.5, y=-0.13)

# Save the figure
fig.savefig("results/figures/adaptive.png", dpi=300, bbox_inches='tight')

# =============================================================================

fig, axs = plt.subplots(1, 2, figsize=(12, 2))  # Two subplots for the metrics
method_names = ['sample_split_3', 'sample_split_4', 'sample_split_5']

nice_names = ["3 Splits", "4 Splits", "5 Splits"]
width = 0.1

baseline_params = {'sample_size': 500, 'number_arms': 10}
dataset = "baseline"
first_stage_sizes = [150]
first_stage_percent = ["30%"]
metrics = ["historic_ratio", "historic_linear"]
x_locations = [] 
shift = 0
results = get_results_matching_parameters(dataset, "", baseline_params)
aggregated_results = aggregate_data(results)

for i, metric in enumerate(metrics):
    
    for k, method in enumerate(method_names):
        historic_data = aggregated_results[method][metric]  # Get data for the current method and metric
        shape = shapes_by_method[method]  # Get shape for current method
        # import pdb; pdb.set_trace()
        # Plot the data
        axs[i].plot(
            historic_data[0],
            label=nice_names[k],
            marker=shape,
            linestyle='-',
            color=colors_by_method[method],
        )

    # Set axis titles, ticks, and limits
    axs[i].set_title(metric.replace("_", " ").capitalize(), fontsize=14)  # Use the metric as the title
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    # axs[i].set_ylim([0.75, 0.95])  # Set y-axis limits (adjust as needed)
    # axs[i].set_xlabel(first_stage_percent[0], fontsize=14)
    # axs[i].set_xticks([])  # Remove x-axis ticks
    # axs[i].set_yticks([0.8, 0.9])  # Add specific y-axis ticks

# Add y-axis label to the first subplot
axs[0].set_ylabel("Normalized Cert.", fontsize=14)

# Add legend across both subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=14)

# Add a shared xlabel below the subplots
fig.supxlabel("Number of stages", fontsize=14, x=0.5, y=-0.13)

# Save the figure
fig.savefig("results/figures/non_adaptive.png", dpi=300, bbox_inches='tight')

#=============================================================================

fig, axs = plt.subplots(1,3, figsize=(12, 2))
method_names = ['one_stage', 'prior','two_stage_successive_elimination','successive_elimination','ucb']
nice_names = ["Single-Stage","Prior-Based","Two-Stage Successive Elimination","Successive Elimination","UCB"]
width = 0.1

baseline_params = {'arm_distribution': 'beta', 'alpha': 1, 'beta': 1}
dataset = "prior_data"

x_locations = [] 
beta = [1,2,4]
shift = 0
for i in range(len(method_names)):
    if method_names[i] == "sample_split" or method_names[i] == 'two_stage_successive_elimination' or method_names[i] == 'successive_elimination':
        shift += 0.5 
    x_locations.append((i+shift)*width)

for i in range(len(beta)):
    baseline_params['beta'] = beta[i]
    max_val = 0
    results = get_results_matching_parameters(dataset,"",baseline_params)
    num_data = 0
    if len(results)>0:
        num_data = len(results)*len(results[0]['random']['certificate'])
    results = aggregate_normalize_data(results,normalized='max')
    print("one_Stage", results["one_stage"])
    print("prior", results["prior"])