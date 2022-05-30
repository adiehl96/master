import numpy as np
from Enumerations import ObservationModels
from Refactored.Kernel import kernel_prior_llh, matrix
from scipy.special import expit
from Utilities.CalcBinErrorStats import calc_bin_error_stats

from Utilities.CondLlh2Array import cond_llh_2array
from Utilities.CreateGPInputPoints import create_gp_input_points
from scipy.linalg import cholesky, cho_solve


def new_permutation(train_data, rng):
    perm = np.arange(len(train_data["v"]))
    rng.shuffle(perm)
    iperm = np.argsort(perm)

    return perm, iperm


def permute(train_data, uu, k, perm):
    train_data["i"] = train_data["i"][perm]
    train_data["j"] = train_data["j"][perm]
    train_data["v"] = train_data["v"][perm]
    uu["ip_uu"] = uu["ip_uu"][
        perm
    ]  # todo check if this permutation makes sense (given that it's a symmetric matrix)
    k["k_ip_pp_uu"] = k["k_ip_pp_uu"][perm]
    return train_data, uu, k


def cond_llh_u(train_data, uu, k, kernel_params, params, u_index):
    llh = 0
    active = (train_data["i"] == u_index).astype(np.int8) + (
        train_data["j"] == u_index
    ).astype(np.int8)
    (active,) = active.nonzero()
    uu = update_kernel_matrices_ip_uu(train_data, uu, k, kernel_params, u_index)
    uu["w_uu"][active] = (
        k["k_ip_pp_uu"][active]
        @ np.linalg.lstsq(k["k_pp_pp_uu"], uu["t_uu"], rcond=-1)[0]
    )
    llh = llh + cond_llh_2array(
        uu["w_uu"][active],
        train_data["v"][active],
        params["uu_observation_model"],
    )

    llh = llh + prior_u(uu, params)

    return llh, uu


def update_kernel_matrices_ip_uu(train_data, uu, k, kernel_params, ip_indices):
    uu["ip_uu"][ip_indices] = create_gp_input_points(
        train_data["i"][ip_indices],
        train_data["j"][ip_indices],
        uu["u"],
    )
    k["k_ip_pp_uu"][ip_indices] = matrix(
        kernel_params,
        uu["ip_uu"][ip_indices],
        uu["pp_uu"],
    )
    return uu


def prior_u(uu, params):
    flat_u = uu["u"].flatten(order="F")
    return (
        -0.5 * (flat_u.T @ flat_u) / np.square(params["u_sd"])
    )  # todo check on matlab notation (obj.u(:)')


def cond_llh_pp_uu(train_data, uu, k, kernel_params, params, pp_index):
    k = update_kernel_matrices_pp_uu(uu, k, kernel_params, pp_index)
    return cond_llh_pp_uu_no_update(train_data, uu, k, params), k


def update_kernel_matrices_pp_uu(uu, k, kernel_params, pp_index):
    k["k_pp_pp_uu"][:, pp_index] = matrix(
        kernel_params, uu["pp_uu"], uu["pp_uu"][pp_index]
    ).flatten()
    k["k_pp_pp_uu"][pp_index] = k["k_pp_pp_uu"][:, pp_index]

    k["k_pp_pp_uu"][pp_index, pp_index] = matrix(kernel_params, uu["pp_uu"][pp_index])
    k["chol_k_pp_pp_uu"] = cholesky(k["k_pp_pp_uu"])
    k["k_ip_pp_uu"][:, pp_index] = matrix(
        kernel_params, uu["ip_uu"], uu["pp_uu"][pp_index]
    ).flatten()
    return k


def cond_llh_pp_uu_no_update(train_data, uu, k, params):
    llh = -np.sum(np.log(np.diag(k["chol_k_pp_pp_uu"]))) - 0.5 * (
        uu["t_uu"] @ cho_solve((k["chol_k_pp_pp_uu"], False), uu["t_uu"])
    )
    uu["w_uu"] = (
        k["k_ip_pp_uu"] @ np.linalg.lstsq(k["k_pp_pp_uu"], uu["t_uu"], rcond=-1)[0]
    )
    llh = llh + cond_llh_2array(
        uu["w_uu"], train_data["v"], params["uu_observation_model"]
    )
    llh = llh + prior_pp_uu(uu, params)
    return llh


def prior_pp_uu(uu, params):
    flat_pp_uu = uu["pp_uu"].flatten(order="F")
    return -0.5 * (flat_pp_uu.T @ flat_pp_uu) / np.square(params["pp_uu_sd"])


def array_llh_uu(train_data, uu, k, kernel_params, kernel_priors, params):
    llh = 0
    llh = (
        llh
        - np.sum(np.log(np.diag(k["chol_k_pp_pp_uu"])))
        - 0.5 * (uu["t_uu"] @ cho_solve((k["chol_k_pp_pp_uu"], False), uu["t_uu"]))
    )  # todo check how solve chol can be replaced in python
    uu["w_uu"] = (
        k["k_ip_pp_uu"] @ np.linalg.lstsq(k["k_pp_pp_uu"], uu["t_uu"], rcond=-1)[0]
    )
    llh = llh + cond_llh_2array(
        uu["w_uu"],
        train_data["v"],
        params["uu_observation_model"],
    )
    llh = llh + prior_pp_uu(uu, params)

    if train_data["v"].size != 0:
        llh = llh + kernel_prior_llh(
            params["kernel_prior_type"], kernel_priors, kernel_params
        )
    return llh, uu


def state(
    train_data, uu, k, kernel_params, kernel_priors, params
):  # Returns a struct with the current variable values
    state = {}
    state["u"] = uu["u"]
    state["pp_uu"] = uu["pp_uu"]
    state["t_uu"] = uu["t_uu"]
    state["array_kern_uu"] = kernel_params
    # state["llh"] = llh()
    return state


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
    print("prediction_uu.shape")
    print(prediction_uu.shape)
    print("test_data['v'].shape")
    print(test_data["v"].shape)
    performance_uu = calc_bin_error_stats(prediction_uu, test_data["v"])
    return performance_uu, uu, k


def talk(
    params, performance
):  # Tell the world about various performance stats, but why is the model talking to us?

    print("")
    print(f"UU : ", end="")
    if params["uu_observation_model"] == ObservationModels.Logit:
        print(
            f"AUC = {performance['auc']:.3f} : Error = {performance['classifier_error']:.3f}"
        )
    else:
        raise Exception("Only ObservationModels Logit Implemented")
