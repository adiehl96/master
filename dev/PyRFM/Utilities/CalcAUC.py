import numpy as np


def calc_auc(p, o, a, probs=None, links=None):
    if probs is None or links is None:
        probs = np.array([])
        links = np.array([])
        for dd in range(a):
            probs += p[dd][o[dd]]
            links += a[dd][o[dd]]

    if probs.size == 0:
        return -1, -1, -1

    order = np.argsort(probs)
    probs = probs[order]
    links = links[order]

    N0 = np.sum(1 - links)
    N1 = np.sum(links)
    FNR = np.array([0] * len(links) + [1])
    TNR = np.array([0] * len(links) + [1])

    for k in range(1, len(links)):
        ii = np.where(probs < probs[k])
        FNR[k] = np.sum(links[ii]) / N1
        TNR[k] = np.sum(1 - links[ii]) / N0

    TPR = 1 - FNR
    FPR = 1 - TNR
    AUC = -np.trapz(FPR, TPR)

    return AUC, TPR, FPR
