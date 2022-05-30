import numpy as np


def calc_class_score(probs, links):
    if probs.size != 0:
        class_score = np.sum(links * probs + (1 - links) * (1 - probs))
        class_score = class_score / len(links)
        return class_score
    else:
        return -1
