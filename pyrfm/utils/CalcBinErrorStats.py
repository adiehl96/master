from Utilities.CalcAUC import calc_auc
from Utilities.CalcKLScore import calc_kl_score
from Utilities.CalcClassScore import calc_class_score
from Utilities.CalcClassifierScore import calc_classifier_score
from Utilities.CalcRMSE import calc_rmse


def calc_bin_error_stats(probs, links):
    auc, _, _ = calc_auc(probs, links)
    kls = calc_kl_score(probs, links)
    cls = calc_class_score(probs, links)
    classifier_error = calc_classifier_score(probs, links)
    rmse = calc_rmse(probs, links)
    return {
        "auc": auc,
        "kls": kls,
        "cls": cls,
        "classifier_error": classifier_error,
        "rmse": rmse,
    }
