import numpy as np
from Kernel import kernel_prior_llh, matrix

from utils.cond_llh_2array import cond_llh_2array
from utils.create_gp_input_points import create_gp_input_points
from scipy.linalg import cholesky, cho_solve


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
    uu, k = update_kernel_matrices_ip_uu(train_data, uu, k, kernel_params, active)
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
    return uu, k


def prior_u(uu, params):
    flat_u = uu["u"].flatten(order="F")
    return (
        -0.5 * (flat_u.T @ flat_u) / np.square(params["u_sd"])
    )  # todo check on matlab notation (obj.u(:)')


def cond_llh_pp_uu(train_data, uu, k, kernel_params, params, pp_index):
    k = update_kernel_matrices_pp_uu(uu, k, kernel_params, pp_index)
    llh, uu = cond_llh_pp_uu_no_update(train_data, uu, k, params)
    return llh, uu, k


def update_kernel_matrices_pp_uu(uu, k, kernel_params, pp_index):
    k["k_pp_pp_uu"][:, pp_index] = matrix(
        kernel_params, uu["pp_uu"], uu["pp_uu"][pp_index]
    ).flatten(order="F")
    k["k_pp_pp_uu"][pp_index] = k["k_pp_pp_uu"][:, pp_index]

    k["k_pp_pp_uu"][pp_index, pp_index] = matrix(kernel_params, uu["pp_uu"][pp_index])
    k["chol_k_pp_pp_uu"] = cholesky(k["k_pp_pp_uu"])
    k["k_ip_pp_uu"][:, pp_index] = matrix(
        kernel_params, uu["ip_uu"], uu["pp_uu"][pp_index]
    ).flatten(order="F")
    return k


def cond_llh_pp_uu_no_update(train_data, uu, k, params):
    llh = -np.sum(np.log(np.diag(k["chol_k_pp_pp_uu"]))) - 0.5 * (
        uu["t_uu"] @ cho_solve((k["chol_k_pp_pp_uu"], False), uu["t_uu"])
    )  # removed transpose for speed
    uu["w_uu"] = (
        k["k_ip_pp_uu"] @ np.linalg.lstsq(k["k_pp_pp_uu"], uu["t_uu"], rcond=-1)[0]
    )
    llh = llh + cond_llh_2array(
        uu["w_uu"], train_data["v"], params["uu_observation_model"]
    )
    llh = llh + prior_pp_uu(uu, params)
    return llh, uu


def prior_pp_uu(uu, params):
    flat_pp_uu = uu["pp_uu"].flatten(order="F")
    prior = (
        -0.5 * (flat_pp_uu @ flat_pp_uu) / np.square(params["pp_uu_sd"])
    )  # no transpose to increase speed
    return prior


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


def update_kernel_matrices_uu(uu, k, kernel_params):
    k["k_ip_pp_uu"] = matrix(kernel_params, uu["ip_uu"], uu["pp_uu"])
    k["k_pp_pp_uu"] = matrix(kernel_params, uu["pp_uu"])
    k["chol_k_pp_pp_uu"] = cholesky(k["k_pp_pp_uu"])
    return k


def cond_llh_array_params_uu(
    train_data, uu, k, kernel_params, kernel_priors, params, new_kernel_params
):
    kernel_params["lls"], kernel_params["lsv"], kernel_params["ldn"] = new_kernel_params
    k = update_kernel_matrices_uu(uu, k, kernel_params)
    llh, uu = array_llh_uu(train_data, uu, k, kernel_params, kernel_priors, params)
    return llh, kernel_params, uu, k
