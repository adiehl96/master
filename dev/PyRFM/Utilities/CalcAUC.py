import numpy as np


def calc_auc(probs=None, links=None):
    if probs is None or probs.size == 0:
        raise Exception("This shouldn't happen")
        return -1, -1, -1

    order = np.argsort(probs)
    probs = probs[order]
    links = links[order]

    n0 = np.sum(1.0 - links)
    n1 = np.sum(links)
    fnr = np.array([0.0] * len(links) + [1.0])
    tnr = np.array([0.0] * len(links) + [1.0])

    for k in range(1, len(links)):
        (ii,) = np.asarray(probs < probs[k]).nonzero()
        fnr[k] = np.sum(links[ii]) / n1
        tnr[k] = np.sum(1 - links[ii]) / n0

    tpr = 1.0 - fnr
    fpr = 1.0 - tnr
    auc = -np.trapz(tpr, fpr)
    return auc, tpr, fpr
