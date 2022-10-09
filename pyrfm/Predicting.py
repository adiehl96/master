from utils.create_gp_input_points import create_gp_input_points
from Kernel import matrix
from Enumerations import ObservationModels
from utils.calc_bin_error_stats import calc_bin_error_stats

from scipy.special import expit

import numpy as np


def prediction(
    test_data, uu, k, kernel_params, params
):  # Returns a cell with predictions
    uu["pred_ip_uu"] = create_gp_input_points(test_data["i"], test_data["j"], uu["u"])
    k["k_pred_pp_uu"] = matrix(kernel_params, uu["pred_ip_uu"], uu["pp_uu"])
    if params["uu_observation_model"] == ObservationModels.Logit:
        prediction_uu = expit(
            k["k_pred_pp_uu"]
            @ np.linalg.lstsq(k["k_pp_pp_uu"], uu["t_uu"], rcond=-1)[0]
        )
        return prediction_uu, uu, k
    else:
        raise Exception("only logit observation model implemented")


def performance(
    test_data,
    uu,
    k,
    kernel_params,
    params,
    predict=True,
    prediction_uu=None,
):  # Returns a struct with various error parameters

    if predict:
        prediction_uu, uu, k = prediction(test_data, uu, k, kernel_params, params)
    performance_uu = calc_bin_error_stats(prediction_uu, test_data["v"])
    return performance_uu, uu, k
