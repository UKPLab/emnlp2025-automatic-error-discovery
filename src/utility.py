'''
    Utility functions for evaluation, ...
'''
import re
import math
import json
import argparse
import numpy as np
from tabulate import tabulate
from scipy.optimize import linear_sum_assignment
from transformers.modeling_utils import PreTrainedModel


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        _data = json.load(f)
    return _data


def dump_json(path:str, data:dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def pp_args(args: argparse.Namespace) -> None:
    """Pretty print the passed script arguments

    Args:
        args (argparse.Namespace): argparse Namespace
    """

    print("\n\nPassed Arguments")    
    _args = [[arg, getattr(args, arg)] for arg in vars(args)]
    _headers = ["Argument", "Value"]

    # if it is a path and the path has more than 4 sections, just keep the last 
    # 4 sections
    for arg in _args:        
        if "/" in str(arg[1]):
            splitted = re.split(r"/", arg[1])
            if len(splitted) > 3:
                arg[1] = "../" + '/'.join(splitted[-3:])

    print(tabulate(_args, _headers, tablefmt="rst") + "\n")


def print_model_footprint(model:PreTrainedModel) -> None:
    """Print the size (loaded) of the pretrained model.

    Args:
        model (PreTrainedModel): The pretrained model
    """
    model_name = model.config_class.model_type
    model_size = model.get_memory_footprint() / math.pow(1000, 3)
    print(f"{model_name} size (loaded): {round(model_size, 2)} GB.")


def accuracy_metrics(preds:np.array, target:list, mapping:dict=None, 
    n_labels:list[int]=None) -> tuple[float, float, float, dict]:
    """This method calculates the harmonic mean (h-mean), accuracy for known classes and accuracy for novel classes as used in the LOOP paper (https://arxiv.org/abs/2312.10897). For the h-mean, we are using the formula from OVANet (https://openaccess.thecvf.com/content/ICCV2021/papers/Saito_OVANet_One-vs-All_Network_for_Universal_Domain_Adaptation_ICCV_2021_paper.pdf)

    Args:
        preds: 1D np.array of clustering predictions
        mapping: mapping of predicted classes to target classes (e.g., from the 
            hungarian algorithm)
        target: list of target values
        novel: list of novel classes

    Returns:
        h_mean: The harmonic mean
        acc_k: Accuracy for known classes
        acc_n: Accuracy for novel classes
        class_wise_acc: Dictionary with per-class accuracies
    """

    unique_target = list(set(target))

    # correct_preds_class collects the correct predictions per class; 
    # all_preds_class collects all predictions per class.
    correct_preds_class = {i: 0 for i in unique_target}
    all_preds_class = {i: 0 for i in unique_target}
    
    # calculate class-wise accuracy
    for i in range(len(preds)):      
        all_preds_class[target[i]] = all_preds_class[target[i]] + 1     
        if mapping:   
            if mapping[preds[i]] == target[i]:
                correct_preds_class[target[i]] =\
                    correct_preds_class[target[i]] + 1
        else:
            if preds[i] == target[i]:
                correct_preds_class[target[i]] =\
                    correct_preds_class[target[i]] + 1
    class_wise_acc = {k: round(correct_preds_class[k] / v, 3) if k in 
        correct_preds_class else 0.0 for k, v in all_preds_class.items()}
        
    # calculate accuracy for known classes
    # if len(unique_target) - len(n_labels) == 0, the novel classes are not 
    # part of the batch
    denominator = len(unique_target) - len(n_labels) if len(unique_target) - len(n_labels) > 0 else len(unique_target)    
    acc_k = round(sum([v for k, v in class_wise_acc.items() if k not in 
        n_labels]) / (denominator), 3) if n_labels else\
        round(sum([v for k, v in class_wise_acc.items()]) / len(unique_target), 3)
    
    # calculate accuracy for novel classes classes
    acc_n = 0.0 if not n_labels else round(sum([v for k, v in 
        class_wise_acc.items() if k in n_labels]) / len(n_labels), 3)
    
    # calculate harmonic mean
    h_mean = 0.0 if acc_k == 0.0 and acc_n == 0.0 else\
        round((2 * acc_k * acc_n) / (acc_k + acc_n), 3)
    
    return h_mean, acc_k, acc_n, class_wise_acc


def hungarian(preds:np.array, target:list) -> tuple[dict, list]:
    """Method for mapping the predictions of a clusterin algorithm (such as 
    kMeans) to the target values.

    Args:
        preds (np.array): 1D np array of predicted values
        target (list): List of target values

    Returns:
        tuple[dict, list]: The dictionary contains the result of the hungarian algorithm for mapping the predicted values to the target values; the list contains the assigned clustering labels
    """

    dim = max(preds.max(), max(target)) + 1
    contingency_matrix = np.zeros((dim, dim))
       
    clustering_labels = []
    for i in range(len(preds)):
        contingency_matrix[preds[i], target[i]] += 1
        clustering_labels.append(target[i])

    row_ind, col_ind =\
        linear_sum_assignment(contingency_matrix.max() - contingency_matrix)            
    
    return dict(zip(row_ind, col_ind)), clustering_labels