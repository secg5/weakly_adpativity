import numpy as np
from utils import compute_hoeffding_bound, compute_hoeffding_bound_one_way, compute_subgaussian_bound, compute_subgaussian_bound_one_way

def UCB(arm_means, num_arms, total_arm_pulls,seed,arm_distribution):
    """Run the UCB algorithm to compute the certificate
    
    Arguments:
        arm_means: List of floats, what the true means are 
        num_arms: N, the total number of arms
        total_arm_pulls: s_1+s_2, the total arms pulled in both stages
        delta: How confident we want to be in our results, float
    
    Returns: Float, certificate, lower bound on the best arm mean
        And Delta, the distance from the empirical mean"""
    delta = 0.1
    # TODO: Create random generator
    np.random.seed(seed)
    max_arm_value = 100
    ucb = max_arm_value * np.ones(num_arms)
    emp_means = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    historic_ratio = []
    historic_linear = []
    best_arm = np.argmax(arm_means)
    for t in range(total_arm_pulls):
        greedy_arm = np.argmax(ucb)
        if arm_distribution == 'effect_size':
            reward = np.random.normal(arm_means[greedy_arm],1)
        else:
            reward = np.random.binomial(1, arm_means[greedy_arm])
        num_pulls[greedy_arm] += 1
        emp_means[greedy_arm] = (emp_means[greedy_arm]*(num_pulls[greedy_arm]-1) + reward)/(num_pulls[greedy_arm])
        ucb[greedy_arm] = emp_means[greedy_arm]

        # if arm_distribution == 'effect_size':
        #     ucb[greedy_arm] += compute_subgaussian_bound(num_pulls[greedy_arm],delta)
        # else:
        ucb[greedy_arm] += compute_hoeffding_bound(num_pulls[greedy_arm],delta)
        # TODO loss function??
        historic_ratio.append((arm_means[best_arm] - emp_means[best_arm])/(t+1))
        historic_linear.append((arm_means[best_arm] - emp_means[best_arm]) - (t+1))

    greedy_arm = np.argmax(ucb)

    # if arm_distribution == 'effect_size':
    #     lower_bound = compute_subgaussian_bound(num_pulls[greedy_arm],delta)
    # else:
    #     lower_bound = compute_hoeffding_bound(num_pulls[greedy_arm],delta)
    
    return emp_means, historic_ratio, historic_linear

def successive_elimination(arm_means, num_arms, total_steps, seed,arm_distribution):
    """Successive Elimination algorithm for best arm identification.

    Parameters:
        arm_means (list or np.array): True means of the arms (for simulation).
        num_arms (int): Number of arms.
        total_steps (int): Total number of time steps for pulling arms.
        delta (float): Confidence parameter.

    Returns:
        best_arm (int): The index of the identified best arm.
        confidence_bound (list): Number of pulls for each arm.
    """
    np.random.seed(seed)
    arm_pulls = np.zeros(num_arms)
    empirical_means = np.zeros(num_arms)
    remaining_arms = list(range(num_arms))
    delta = 0.1
    step = 0
    historic_ratio = []
    historic_linear = []
    best_arm = np.argmax(arm_means)
    while len(remaining_arms) > 1 and step < total_steps:
        for arm in remaining_arms:
            arm_pulls[arm] += 1

            if arm_distribution == 'effect_size':
                reward =  np.random.normal(arm_means[arm],1)
            else:
                reward =  np.random.binomial(1, arm_means[arm])
            empirical_means[arm] = ((empirical_means[arm] * (arm_pulls[arm] - 1)) + reward) / arm_pulls[arm]
            step += 1
            if step >= total_steps:
                break
        
        # Update confidence bounds
        if arm_distribution == 'effect_size':
            confidence_bound = compute_subgaussian_bound_one_way(arm_pulls[remaining_arms],delta)
        else:
            confidence_bound = compute_hoeffding_bound_one_way(arm_pulls[remaining_arms],delta)
        
        # Calculate upper and lower bounds
        # Here we do need the 2/delta
        upper_bounds = empirical_means[remaining_arms] + confidence_bound
        lower_bounds = empirical_means[remaining_arms] - confidence_bound
        
        # Find the best arm based on current estimates
        best_arm_index = np.argmax(empirical_means[remaining_arms])
        best_arm = remaining_arms[best_arm_index]
        
        # Eliminate arms whose upper bound is worse than the best arm's lower bound
        remaining_arms = [
            arm for i, arm in enumerate(remaining_arms)
            if upper_bounds[i] >= lower_bounds[best_arm_index]
        ]
        # TODO: encapsulate as a function
        historic_ratio.append((arm_means[best_arm] - empirical_means[best_arm])/(step + 1))
        historic_linear.append((arm_means[best_arm] - empirical_means[best_arm]) - (step +1))

    return empirical_means, historic_ratio, historic_linear


