import numpy as np 
from copy import deepcopy 
from utils import sample_bernoulli, sample_normal, compute_hoeffding_bound, compute_subgaussian_bound
from scipy.stats import norm

def beta_prior_policy(first_stage,n,reward_parameters,s_1,T,seed):
    """Compute the best certificate with a beta prior policy
    
    Arguments:
        first_stage: The arm pulls from the first stage
            A matrix of arm pulls for each arm with 0-1 
            Of Whether it is a success
        n: Number of total arms 
        reward_parameters: Dictionary, with 
            Info on the alpha, beta parameter (for the prior)
        s_1: Number of total pulls in stage 1
        T: Total pulls across stages
        seed: Integer, random seed
    
    Returns: Certificate, float, and the width, float"""

    np.random.seed(seed)
    new_alphas = []
    new_betas = []

    for i in range(len(first_stage)):
        success = np.sum(first_stage[i])
        total = len(first_stage[i])
        new_alphas.append(reward_parameters['alpha']+success)
        new_betas.append(reward_parameters['beta']-success+total)

    current_set = []
    curr_value = 0

    for _ in range(n):
        next_best_value = (0,0)
        for next_elem in range(n):
            if next_elem in current_set:
                continue 
        
            new_set = deepcopy(current_set)
            new_set.append(next_elem)

            num_trials = 100
            min_means = []

            for i in range(num_trials):
                means = np.array([np.random.beta(new_alphas[new_set[j]],new_betas[new_set[j]]) for j in range(len(new_set))])
                sampled_values = sample_bernoulli(T-s_1, len(new_set), means)
                predicted_means = np.mean(sampled_values,axis=1)
                # print("predicted_means",predicted_means)
                # print("new_set",new_set)
                min_means.append(np.max(predicted_means))
            min_means = np.mean(min_means)

        #     means = np.array([np.random.beta(new_alphas[new_set[j]],new_betas[new_set[j]]) for j in range(len(new_set))])
        #     sampled_values = sample_bernoulli(T-s_1, len(new_set), means)
        #     predicted_means = np.mean(sampled_values,axis=1)
        #     min_certificate_delta = compute_hoeffding_bound((T-s_1)//len(new_set), delta)
        #     min_certificate.append(np.max(predicted_means) - min_certificate_delta)
        # min_certificate = np.mean(min_certificate)

            if next_best_value[0] < min_means:
                next_best_value = (min_means,next_elem)
        if next_best_value[0] == 0:
            break 
        elif next_best_value[0] <= curr_value:
            break 
        else:
            curr_value = next_best_value[0]
            current_set.append(next_best_value[1])
    
    certificate = curr_value

    return [certificate/2], current_set