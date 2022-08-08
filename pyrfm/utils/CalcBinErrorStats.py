from sklearn import metrics
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np


def calc_class_score(probs, links):
    if probs.size != 0:
        class_score = np.sum(links * probs + (1 - links) * (1 - probs))
        class_score = class_score / len(links)
        return class_score
    else:
        return -1


def calc_classifier_score(probs, links):
    if probs.size != 0:
        classifier_score = np.sum(links * (probs < 0.5) + (1 - links) * (probs >= 0.5))
        classifier_score = classifier_score / len(links)
        return classifier_score
    else:
        return -1


def calc_bin_error_stats(probs, links):
    fpr, tpr, _ = metrics.roc_curve(links, probs)
    auc = metrics.auc(fpr, tpr)
    kls = -log_loss(links, probs, normalize=False, labels=[0, 1]) / len(links)
    cls = calc_class_score(probs, links)
    classifier_error = calc_classifier_score(probs, links)
    rmse = mean_squared_error(links, probs, squared=False)
    results = {
        "auc": auc,
        "kls": kls,
        "cls": cls,
        "classifier_error": classifier_error,
        "rmse": rmse,
    }
    return results
