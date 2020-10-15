"""
Helpers to facilitate leave one out decoding.
Train classifier on all but one data point, then decide if projection of 
left out point onto the classifier axis is correct / incorrect
"""
import numpy as np
from nems_lbhb.decoding import compute_dprime

def get_proportion_correct(A, B):
    """
    A and B are two distinct class distributions. (rep x neurons)
    Perform leave-one-out CV estimate of probability of correctly IDing single
    trial resps as either A or B
    """
    
    outcome = np.concatenate((np.zeros(A.shape[0]), np.ones(B.shape[0])))
    AB = np.concatenate((A, B), axis=0)
    ntrials = AB.shape[0]
    classified = []
    for idx in range(ntrials):
        nidx = list(set(range(0, ntrials)).difference(set([idx])))
        _AB = AB[nidx, :]
        _outcome = outcome[nidx]
        _A = _AB[_outcome==0, :]
        _B = _AB[_outcome==1, :]
        if _A.shape[1]==1:
            ap = _A.mean()
            bp = _B.mean()
            p = AB[idx, 0]
        else:
            _, wopt, _, _, _, _ = compute_dprime(_A.T, _B.T)
            # project train data
            ap = _A.dot(wopt / np.linalg.norm(wopt)).mean()
            bp = _B.dot(wopt / np.linalg.norm(wopt)).mean()
            # determine which class the left out point is classified as
            p = AB[idx, :].dot(wopt / np.linalg.norm(wopt))[0]
        if outcome[idx]==0:
            if abs(p-ap) < abs(p-bp):
                # correct
                classified.append(1)
            else:
                # incorrect
                classified.append(0)
        elif outcome[idx]==1:
            if abs(p-ap) > abs(p-bp):
                # correct
                classified.append(1)
            else:
                # incorrect
                classified.append(0)

    return sum(classified) / ntrials