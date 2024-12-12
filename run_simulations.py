import numpy as np
from copy import deepcopy 
import csv
from utils import compute_hoeffding_bound, compute_subgaussian_bound, sample_bernoulli, sample_normal
from adaptive_policies import UCB, successive_elimination
from non_adaptive_policies import compute_top_k, fixed_k_policies, two_stage_thompson_sampling_bernoulli, two_stage_successive_elimination
from prior_policies import beta_prior_policy

def generate_arm_means(arm_distribution,arm_parameters,n_arms):
    """Generate the underlying parameters for all arms 
    
    Arguments:
        arm_distribution: String, either uniform, beta, 
            or beta_misspecified
        arm_parameters: Dictionary, with information on the following: 
            alpha, beta for beta distributions
            diff_mean_1, diff_std_1, for misspecified
        n_arms: Integer, number of arms we're generating 
    
    Returns: List of floats, the means for all arms"""

    arm_means = []

    for _ in range(n_arms):
        if arm_distribution == 'uniform':
            arm_means.append(np.random.uniform(arm_parameters['uniform_low'],arm_parameters['uniform_high']))
        elif arm_distribution == 'beta':
            arm_means.append(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']))
        elif arm_distribution == 'beta_misspecified':
            arm_means.append(np.clip(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']) + np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']),0,1))
    if arm_distribution == 'unimodal_diff':
        arm_means.append(np.random.random())    
        for _ in range(1,n_arms):
            diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
            arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))
    if arm_distribution == 'bimodal_diff':
        arm_means.append(np.random.random())    
        for _ in range(1,n_arms):
            if np.random.random() < 0.5:
                diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
            else:
                diff = np.random.normal(arm_parameters['diff_mean_2'],arm_parameters['diff_std_2']) 
            arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))
    return arm_means 


def run_experiments(config):
    """Given a set of parameters, run experiments comparing various
        policies for selecting arms
        
    Arguments:
        config: Dictionary with information on number of arms, 
            sample_size, delta, arm distributions, and s_1, T
    
    Returns: Dictionary of results; each key is a type of algorithm
        which maps to a dictionary with a (list of) certificates and a 
        certificate width"""

    n = config['number_arms']
    T = config['sample_size']
    mu = config['distribution']
    arm_distribution = config['arm_distribution']
    s_1 = config['first_stage_size']
    # delta = config['delta']
    reward_parameters = config['reward_parameters']
    seed = config['random_seed']
    s_2 = T - s_1    
    mu = np.array(mu)

    np.random.seed(seed)

    # if arm_distribution == 'effect_size': 
    #     first_stage = sample_normal(s_1, n, mu)
    first_stage = sample_bernoulli(s_1, n, mu)

    # Two Stage Algorithms
    top_k_split, omniscient_k, dominant_k, train_means_indices = compute_top_k(first_stage, n, s_1, s_2, mu)
    means_omniscient = fixed_k_policies(omniscient_k, s_2, mu, train_means_indices,seed,arm_distribution)
    means_dominant = fixed_k_policies(dominant_k, s_2, mu, train_means_indices,seed,arm_distribution)
    means_split = fixed_k_policies(top_k_split, s_2, mu, train_means_indices,seed,arm_distribution)

    # if arm_distribution == 'effect_size':
    #     width_split_total = compute_subgaussian_bound(s_2/top_k_split + s_1/n, delta)
    
    # width_split_total = compute_hoeffding_bound(s_2/top_k_split + s_1/n, delta)
    certificate_split_total = means_split # + width_split - width_split_total 

    # Other Two Stage Algorithms
    # if arm_distribution == 'effect_size':
    #     certificate_thompson, width_thompson = two_stage_thompson_sampling_normal(first_stage,n,delta,s_1,T,mu,seed,arm_distribution)
    # else:
    # certificate_thompson, width_thompson = two_stage_thompson_sampling_bernoulli(first_stage,n,s_1,T,mu,seed,arm_distribution)
    # certificate_two_stage_se, width_two_stage_se = two_stage_successive_elimination(first_stage,n,mu, s_1, T,seed,arm_distribution)

    # Adaptive Algorithms
    # import pdb; pdb.set_trace()
    # TODO how to incorporate loss fucnitons
    certificate_ucb, historic_ratio, historic_linear = UCB(mu, n, T, seed,arm_distribution)
    certificate_se, historic_ratio, historic_linear = successive_elimination(mu, n, T, seed,arm_distribution)

    results = {} 
    results["omniscient"] = {"certificate":means_omniscient}
    results["dominant"] = {"certificate":means_dominant}
    results["sample_split"]  = {"certificate":means_split}
    #TODO change
    results["sample_split_total"] = {"certificate":certificate_split_total}
    best_arm = np.argmax(mu)
    for other_splits in [3,4,5]:
        np.random.seed(seed)

        train_mean_indices_start = deepcopy(train_means_indices)
        current_top_k = top_k_split 
        current_mu = deepcopy(mu)

        top_k_values = []
        empirical_means = np.mean(first_stage, axis=1)
        historic_ratio = [(mu[best_arm] - empirical_means[best_arm])]
        historic_linear = [(mu[best_arm] - empirical_means[best_arm])]

        for step in range(1,other_splits):
            selected_indices = train_mean_indices_start[:current_top_k]
            if arm_distribution == 'effect_size': 
                second_stage = sample_normal(s_2//(other_splits-1), current_top_k, current_mu[selected_indices])
            else:
                second_stage = sample_bernoulli(s_2//(other_splits-1), current_top_k, current_mu[selected_indices])
        
            # TODO Why s_2//2?
            current_top_k, _,_,train_mean_indices_start = compute_top_k(second_stage, top_k_split, s_2//2, s_2//2, current_mu[selected_indices])
            current_mu = current_mu[selected_indices]
            empirical_means = np.mean(second_stage, axis=1)
            best_empirical_arm = np.argmax(empirical_means)
            top_k_values.append(current_top_k)
            historic_ratio.append((mu[best_arm] - empirical_means[best_empirical_arm])/(step + 1))
            historic_linear.append((mu[best_arm] - empirical_means[best_empirical_arm]) - (step + 1))


        means_split_n = fixed_k_policies(current_top_k, s_2//(other_splits-1), current_mu, train_mean_indices_start,seed,arm_distribution)

        
        results["sample_split_{}".format(other_splits)]  = {"certificate":means_split_n, "historic_ratio": historic_ratio, "historic_linear": historic_linear}

#===================================================================================================
    if 'beta' in config['arm_distribution']:
        
        certificate_prior, width_prior = beta_prior_policy(first_stage,n,reward_parameters,s_1,T,seed)
        results["prior"] = {'certificate': certificate_prior}
    
    

        # total_arm_pulls = s_1/n
        # for i in range(len(top_k_values)):
        #     total_arm_pulls += (s_2//(other_splits-1))/(top_k_values[i])
        # if arm_distribution == 'effect_size':
        #     width_split_total_n = compute_subgaussian_bound(total_arm_pulls)
        # else:
        #     width_split_total_n = compute_hoeffding_bound(total_arm_pulls)
        # certificate_split_total_n = certificate_split_n + width_split_n - width_split_total_n
        # results["sample_split_total_{}".format(other_splits)]  = {"certificate":certificate_split_total_n, 
        #     "certificate_width": width_split_total_n}



    # results['two_stage_thompson'] = {"certificate": certificate_thompson, 
    #                     "certificate_width": width_thompson}
    # results['two_stage_successive_elimination'] = {'certificate': certificate_two_stage_se, 
    #                     "certificate_width": width_two_stage_se}

    results["ucb"] = {'certificate': certificate_ucb, "historic_ratio": historic_ratio, "historic_linear": historic_linear}
    results["successive_elimination"] = {'certificate': certificate_se, "historic_ratio": historic_ratio, "historic_linear": historic_linear}
    
    if config['run_all_k']:
        for new_k in range(1,n+1):
            means_k = fixed_k_policies(new_k, s_2, mu, train_means_indices,seed,arm_distribution)
            results["k_{}".format(new_k)] = {'certificate': means_k}

        results["random"] = results["k_{}".format(np.random.randint(1,n))]
        results["one_stage"] = deepcopy(results["k_{}".format(n)])

        # if arm_distribution == 'effect_size':
        #     results["one_stage"]["certificate_width"] = compute_subgaussian_bound(T//n)
        # else:
        #     results["one_stage"]["certificate_width"] = compute_hoeffding_bound(T//n)
        # results["one_stage"]["certificate"] += (results["k_{}".format(n)]["certificate_width"]-results["one_stage"]["certificate_width"])

    # if 'beta' in config['arm_distribution']:
    #     certificate_prior, width_prior = beta_prior_policy(first_stage,n,reward_parameters,delta,s_1,T,seed)
    #     results["prior"] = {'certificate': certificate_prior, 'certificate_width': width_prior}
    # elif arm_distribution == 'effect_size':
    #     certificate_prior, width_prior = fixed_prior_policy(first_stage,n,reward_parameters,delta,s_1,T,seed)
    #     results["prior"] = {'certificate': certificate_prior, 'certificate_width': width_prior}

    return results

