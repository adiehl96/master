import numpy as np


def calc_classifier_score(probs, links):
    if probs.size != 0:
        classifier_score = np.sum(links * (probs < 0.5) + (1 - links) * (probs >= 0.5))
        classifier_score = classifier_score / len(links)
        return classifier_score
    else:
        return -1
