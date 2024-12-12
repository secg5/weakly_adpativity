from utils import compute_hoeffding_bound, compute_hoeffding_bound_one_way, compute_subgaussian_bound_one_way, compute_subgaussian_bound, sample_bernoulli, sample_normal
import numpy as np

def compute_top_k(first_stage, n, s_1, s_2, mu):
    """
    Arguments:
        first_stage: Numpy matrix (n x s_{1}//n) of rewards for each
            arm for each timestep in the first stage
        n: Number of total arms 
        s_1: Total arm pulls in the first stage 
        s_2: Total arm pulls in the second stage
        mu: True arm means

    Returns: 4 things
        top_k_split: Value of k from sample splitting
        omniscient_k: Value of k with knowledge of mu
        dominant_k: Value of k where ??
        train_means_indices: Sorted list of the top arms 
    """

    sample_number = s_1 // n
    train_data = first_stage[:,:(sample_number // 2)]
    test_data = first_stage[:,(sample_number // 2):sample_number]
    train_means = train_data.mean(axis=1)
    train_means_indices = train_means.argsort()[::-1]

    test_means = test_data.mean(axis=1)

    srt_test_means = np.take_along_axis(test_means, train_means_indices, axis=0)
    # delta = np.sqrt((np.arange(n)+1)/(2*s_2))

    values = []
    for i in range(n):
        value = np.max(srt_test_means[:(i+1)])
        values.append(value)
    values = np.array(values)
    
    best_estimate = np.argmax(values)    
    top_k_split = best_estimate + 1

    if len(srt_test_means) == len(values):
        aux = values - np.roll(srt_test_means, -1, axis=0)
        non_negative_mask = aux >= 0
        non_negative_indices = np.where(non_negative_mask)[0]
        dominant_k = non_negative_indices[0] + 1 if non_negative_indices.size > 0 else 1
        str_mu = np.take_along_axis(mu, train_means_indices, axis=0)
        true_values = str_mu
        omniscient_k = np.argmax(true_values) + 1
    
        return top_k_split, omniscient_k, dominant_k, train_means_indices
    else:
        return top_k_split, -1, -1, train_means_indices


def fixed_k_policies(k, s_2, arm_means, selected_arms,seed, arm_distribution):
    np.random.seed(seed)
    B = selected_arms[:k]

    if arm_distribution == 'effect_size':
        second_stage = sample_normal(s_2, k, arm_means, B)
    else:
        second_stage = sample_bernoulli(s_2, k, arm_means, B)
    means = second_stage.mean(axis=1)
    return means

def two_stage_thompson_sampling_bernoulli(first_stage,n,delta,s_1,T,mu,seed,arm_distribution):
    """Run a two-stage Thompson Sampling Method
    Essentially, given a uniform prior, compute a posterior 
    Then, using this posterior, compute the relative probability for 
    each arm being the top, and sample proportional to this
    
    Arguments:
        first_stage: 0-1 matrix of size n x s_{1}, with rewards for
            arm x time stage
        n: Integer, number of arms
        delta: Float, confidence we're aiming for
        s_1: Number of arms pulled in the first stage
        T: total number of arms pulled
        mu: True means 
        seed: Integer, random seed"""

    np.random.seed(seed)
    new_alphas = []
    new_betas = []

    initial_alpha = initial_beta = 1

    for i in range(len(first_stage)):
        success = np.sum(first_stage[i])
        total = len(first_stage[i])
        new_alphas.append(initial_alpha+success)
        new_betas.append(initial_beta-success+total)
    
    trials = 1000
    max_arm_rates = np.array([0.0 for i in range(n)])

    for _ in range(trials):
        p_samples = [np.random.beta(new_alphas[j],new_betas[j]) for j in range(n)]
        max_arm_rates[np.argmax(p_samples)] += 1
    max_arm_rates /= trials 

    rounded_arm_pulls = np.zeros(n)
    s_2 = T-s_1
    for i in range(s_2):
        rounded_arm_pulls[np.random.choice(n,p=max_arm_rates)] += 1
 
    empirical_means = np.zeros(n)
    for i in range(n):
        if rounded_arm_pulls[i] > 0:
            if arm_distribution == 'effect_size':
                empirical_means[i] = np.mean(sample_normal(int(rounded_arm_pulls[i]), 1, np.array([mu[i]])))
            else:
                empirical_means[i] = np.mean(sample_bernoulli(int(rounded_arm_pulls[i]), 1, np.array([mu[i]])))

    if arm_distribution == 'effect_size':
        widths = np.array([compute_subgaussian_bound(rounded_arm_pulls[i],delta) for i in range(n)])
    else:
        widths = np.array([compute_hoeffding_bound(rounded_arm_pulls[i],delta) for i in range(n)])
    certificates = empirical_means-widths
    best_certificate = np.argmax(certificates)

    return [certificates[best_certificate]], widths[best_certificate]

def two_stage_thompson_sampling_normal(first_stage,n,delta,s_1,T,mu,seed,arm_distribution):
    """Run a two-stage Thompson Sampling Method
    Essentially, given a uniform prior, compute a posterior 
    Then, using this posterior, compute the relative probability for 
    each arm being the top, and sample proportional to this
    
    Arguments:
        first_stage: 0-1 matrix of size n x s_{1}, with rewards for
            arm x time stage
        n: Integer, number of arms
        delta: Float, confidence we're aiming for
        s_1: Number of arms pulled in the first stage
        T: total number of arms pulled
        mu: True means 
        seed: Integer, random seed"""

    np.random.seed(seed)
    new_means = []
    new_stds = []

    initial_mean = 0
    initial_std = 1

    for i in range(len(first_stage)):
        success = np.sum(first_stage[i])
        total = len(first_stage[i])
        std = np.std(first_stage[i])
        new_means.append((1/(1/initial_std**2)+total/(std**2))*(initial_mean/initial_std**2 + success/std**2))
        new_stds.append((1/(1/initial_std**2)+total/(std**2)))
    
    trials = 1000
    max_arm_rates = np.array([0.0 for i in range(n)])

    for _ in range(trials):
        p_samples = [np.random.normal(new_means[j],new_stds[j]) for j in range(n)]
        max_arm_rates[np.argmax(p_samples)] += 1
    max_arm_rates /= trials 

    rounded_arm_pulls = np.zeros(n)
    s_2 = T-s_1
    for i in range(s_2):
        rounded_arm_pulls[np.random.choice(n,p=max_arm_rates)] += 1
 
    empirical_means = np.zeros(n)
    for i in range(n):
        if rounded_arm_pulls[i] > 0:
            if arm_distribution == 'effect_size':
                empirical_means[i] = np.mean(sample_normal(int(rounded_arm_pulls[i]), 1, np.array([mu[i]])))
            else:
                empirical_means[i] = np.mean(sample_bernoulli(int(rounded_arm_pulls[i]), 1, np.array([mu[i]])))

    if arm_distribution == 'effect_size':
        widths = np.array([compute_subgaussian_bound(rounded_arm_pulls[i],delta) for i in range(n)])
    else:
        widths = np.array([compute_hoeffding_bound(rounded_arm_pulls[i],delta) for i in range(n)])
    certificates = empirical_means-widths
    best_certificate = np.argmax(certificates)

    return [certificates[best_certificate]], widths[best_certificate]


def two_stage_successive_elimination(first_stage,n,arm_means, s_1, T,seed, arm_distribution):
    """Successive Elimination algorithm for best arm identification.
        Do this in two-stages; first uniform
        Compute Successive Elimination
        Then do uniform again

    Parameters:
        first_stage: 0-1 matrix of size n x s_{1}, with rewards for
            arm x time stage
        n: Integer, number of arms 
        arm_means (list or np.array): True means of the arms (for simulation).
        s_1: Integer, total arm pulls in the first stage
        T: Integer, total arm pulls
        delta (float): Confidence parameter.
        seed: Integer, random seed

    Returns:
        best_arm (int): The index of the identified best arm.
        confidence_bound (list): Number of pulls for each arm.
    """
    np.random.seed(seed)
    s_2 = T-s_1
    arm_pulls = np.zeros(n)
    empirical_means = np.zeros(n)
    remaining_arms = list(range(n))
    delta = 0.1

    arm_pulls += s_1//n 
    empirical_means = np.mean(first_stage,axis=1)
    
    if arm_distribution == 'effect_size':
        confidence_bound = compute_subgaussian_bound_one_way(arm_pulls,delta)
    else:
        confidence_bound = compute_hoeffding_bound_one_way(arm_pulls,delta)
    
    upper_bounds = empirical_means[remaining_arms] + confidence_bound
    lower_bounds = empirical_means[remaining_arms] - confidence_bound
    
    best_arm_index = np.argmax(empirical_means[remaining_arms])
    
    remaining_arms = [
        arm for i, arm in enumerate(remaining_arms)
        if upper_bounds[i] >= lower_bounds[best_arm_index]
    ]
    k = len(remaining_arms)

    if arm_distribution == 'effect_size':
        second_stage = sample_normal(s_2, k, arm_means, remaining_arms)
        lower_bound = compute_subgaussian_bound(s_2//k, delta)
    else:
        second_stage = sample_bernoulli(s_2, k, arm_means, remaining_arms)
        lower_bound = compute_hoeffding_bound(s_2//k, delta)
    certificate = second_stage.mean(axis=1) - lower_bound
    return certificate, lower_bound

    