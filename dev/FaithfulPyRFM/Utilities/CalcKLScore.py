import numpy as np


def calc_kl_score(probs, links):
    if probs.size == 0:
        eps = np.finfo(np.float64).eps
        probs[probs < eps] = eps
        probs[probs > (1 - eps)] = 1 - eps
        klscore = np.sum(links * np.log(probs) + (1 - links) * np.log(1 - probs))
        klscore = klscore / len(links)
        return klscore
    else:
        return -1
