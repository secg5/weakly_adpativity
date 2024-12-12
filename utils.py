
import glob 
import json 
import numpy as np
import os
import copy

def get_results_matching_parameters(folder_name,result_name,parameters):
    """Get a list of dictionaries, with data, which match some set of parameters
    
    Arguments:
        folder_name: String, which folder the data is located in
        result_name: String, what the suffix is for the dataset
        parameters: Dictionary with key,values representing some known parameters
        
    Returns: List of Dictionaries"""
   
    all_results = glob.glob("results/{}/{}*.json".format(folder_name,result_name))
    ret_results = []
    for file_name in all_results:
        load_file = json.load(open(file_name,"r"))

        for p in parameters:
            if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                if p in load_file['parameters']['reward_parameters']:
                    if load_file['parameters']['reward_parameters'][p] != parameters[p]:
                        break 
                else:
                    break 
        else:
            ret_results.append(load_file)
    return ret_results

def aggregate_data(results):
    """Get the average and standard deviation for each key across 
        multiple trials
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""
    

    # Filters only the desired keys
    r = [] 
    for row in results:
        temp = {}
        for key in row:
            if key != 'parameters':
                temp[key] = {}
                temp[key]['certificate'] = row[key]['certificate']
                if key in ["ucb", "successive_elimination", 'sample_split_3', 'sample_split_4', 'sample_split_5']:
                    temp[key]["historic_ratio"] = row[key]["historic_ratio"]
                    temp[key]["historic_linear"] = row[key]["historic_linear"]
        
        n_arms = len(row['parameters']['distribution'])
        temp["median_effect"] = sorted(row['parameters']['distribution'])[n_arms//2]
        r.append(temp)

    #Build the return dict, aggregated by experiments
    results = r
    ret_dict = {}
    for experiment in results:
        for key in experiment:
            if type(experiment[key]) == int or type(experiment[key]) == float:
                if key not in ret_dict:
                    ret_dict[key] = []
                ret_dict[key].append(experiment[key])
            elif type(experiment[key]) == dict:
                
                if key not in ret_dict:
                    ret_dict[key] = {}
                    ret_dict[key]["certificate"] = []
                    if key in ["ucb", "successive_elimination", 'sample_split_3', 'sample_split_4', 'sample_split_5']:
                        ret_dict[key]["historic_ratio"] = []
                        ret_dict[key]["historic_linear"] = []
                
                ret_dict[key]["certificate"].append(experiment[key]["certificate"])
                if key in ["ucb", "successive_elimination", 'sample_split_3', 'sample_split_4', 'sample_split_5']:
                    
                    ret_dict[key]["historic_ratio"] += experiment[key]["historic_ratio"]
                    ret_dict[key]["historic_linear"] += experiment[key]["historic_linear"]
    #Aggregate the data per runs
    for key in ret_dict:
        if type(ret_dict[key]) != dict:
            ret_dict[key] = (np.mean(ret_dict[key]),np.std(ret_dict[key]))
        else:
            for param in ret_dict[key]:
                if param == "certificate":
                    ret_dict[key][param] = (np.mean(ret_dict[key][param]),np.std(ret_dict[key][param]))
                elif param in ["historic_ratio", "historic_linear"]:

                    lists = ret_dict[key][param]
                    min_length = min(len(lst) for lst in lists)
                    trimmed_lists = [lst[:min_length] for lst in lists]
                    historic = np.stack(trimmed_lists, axis=0)
                    ret_dict[key][param] = (np.mean(historic, axis=0), np.std(historic, axis=0))
                

    # import pdb; pdb.set_trace()
    return ret_dict 

def aggregate_normalize_data(results,baseline=None, normalized="max"):
    """Get the average and standard deviation for each key across 
        multiple trials; with each reward/etc. being averaged
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""
    
    results_copy = copy.deepcopy(results)

    for data_point in results_copy:
        
        # if normalized == "max":
        #     normalized_factor = max(data_point["parameters"]["distribution"])
        # elif normalized == "median":
        #     n_arms = len(data_point["parameters"]["distribution"])
        #     normalized_factor = sorted(data_point["parameters"]["distribution"])[n_arms // 2]

        if normalized == "max":
            normalized_factor = max(data_point["parameters"]["distribution"])
        elif normalized == "median":
            normalized_factor = data_point["random"]["certificate"]
        for key in data_point:
            if key != "parameters":
                data_point[key]["certificate"] = [x/normalized_factor for i, x in enumerate(data_point[key]["certificate"])]

    return aggregate_data(results_copy)


def delete_duplicate_results(folder_name,result_name,data):
    """Delete all results with the same parameters, so it's updated
    
    Arguments:
        folder_name: Name of the results folder to look in
        results_name: What experiment are we running (hyperparameter for e.g.)
        data: Dictionary, with the key parameters
        
    Returns: Nothing
    
    Side Effects: Deletes .json files from folder_name/result_name..."""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))

    for file_name in all_results:
        try:
            load_file = json.load(open(file_name,"r"))
        except:
            continue 

        if 'parameters' in load_file and load_file['parameters'] == data['parameters']:
            try:
                os.remove(file_name)
            except OSError as e:
                print(f"Error deleting {file_name}: {e}")

def compute_hoeffding_bound(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a Hoeffding Bound
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt((1/(2*n_data))*np.log(1/delta))

def compute_hoeffding_bound_one_way(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a Hoeffding Bound
        Here, we do 2/delta, because we just want a 
        bound for whether it is larger
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt(np.log(2 / delta) / (2* n_data))



def compute_subgaussian_bound(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a subgaussian Bound with \sigma=1
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt((2/(n_data))*np.log(1/delta))

def compute_subgaussian_bound_one_way(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a subgaussian Bound with \sigma=1
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt((2/(n_data))*np.log(2/delta))



def sample_bernoulli(N, K, mu, indices=None):
    """
    Generate a sample of N from K different Bernoulli distributions with given probabilities.
    
    Arguments:
        N (int): Number of samples to draw total.
        K (int): Number of different Bernoulli distributions.
        p (list or np.array): Probabilities for each of the K distributions.
    
    Returns:
        np.array: A K x N array where each row contains N samples from the corresponding Bernoulli distribution.
    """
    if indices is None:
        indices = range(K)
    samples = np.random.binomial(1, mu[indices, np.newaxis], (K, N//K))
    return samples

def sample_normal(N, K, mu, indices=None):
    """
    Generate a sample of N from K different Bernoulli distributions with given probabilities.
    
    Arguments:
        N (int): Number of samples to draw total.
        K (int): Number of different Bernoulli distributions.
        p (list or np.array): Probabilities for each of the K distributions.
    
    Returns:
        np.array: A K x N array where each row contains N samples from the corresponding Bernoulli distribution.
    """
    if indices is None:
        indices = range(K)
    samples = np.random.normal(mu[indices, np.newaxis], 1,(K, N//K))
    return samples



