import numpy as np
from Refactored.Kernel import matrix
from Refactored.RFM import (
    array_llh_uu,
    cond_llh_pp_uu,
    cond_llh_pp_uu_no_update,
    cond_llh_u,
    update_kernel_matrices_ip_uu,
    update_kernel_matrices_pp_uu,
)
from SliceSampling.SliceSampleMax import slice_sample_max
from scipy.linalg import cholesky, cho_solve


def slice_u(train_data, uu, k, kernel_params, params):
    (m,) = uu["u"].shape
    print("m", m)
    perm = np.arange(m)
    params["rng"].shuffle(perm)
    for mm in perm:
        log_pstar, uu = cond_llh_u(train_data, uu, k, kernel_params, params, mm)
        log_pstar = log_pstar + np.log(params["rng"].uniform())

        direction = params["rng"].uniform()
        direction = direction / np.sqrt(np.sum(np.square(direction)))

        rr = params["rng"].uniform()
        u_l = uu["u"][mm] - rr * params["u_slice_width"] * direction
        u_r = uu["u"][mm] + (1 - rr) * params["u_slice_width"] * direction
        u_saved = uu["u"].copy()

        attempts = 0

        if params["u_step_out"]:
            while attempts < params["u_max_attempts"]:
                uu["u"][mm] = u_l
                test_p, uu = cond_llh_u(train_data, uu, k, kernel_params, params, mm)
                if test_p > log_pstar:
                    u_l = uu["u"][mm] - params["u_slice_width"] * direction
                else:
                    break
                attempts += 1
            uu["u"][mm] = u_saved[mm]
            while attempts < params["u_max_attempts"]:
                uu["u"][mm] = u_r
                test_p, uu = cond_llh_u(train_data, uu, k, kernel_params, params, mm)
                if test_p > log_pstar:
                    u_r = uu["u"][mm] + params["u_slice_width"] * direction
                else:
                    break
                attempts += 1

        uu["u"][mm] = u_saved[mm]
        attempts = 0

        while attempts < params["u_max_attempts"]:
            uu["u"][mm] = params["rng"].uniform() * (u_r - u_l) + u_l
            log_p_prime, uu = cond_llh_u(train_data, uu, k, kernel_params, params, mm)
            if log_p_prime >= log_pstar:
                break
            else:
                if (uu["u"][mm] - u_saved[mm]) * direction > 0:
                    u_r = uu["u"][mm]
                elif (uu["u"][mm] - u_saved[mm]) * direction < 0:
                    u_l = uu["u"][mm]
                else:
                    raise Exception(
                        "BUG DETECTED: Shrunk to current position and still not acceptable."
                    )
            attempts += 1
        if attempts >= params["u_max_attempts"]:
            uu["u"][mm] = u_saved[mm]
            ip_indices = (train_data["i"] == mm).astype(np.int8) + (
                train_data["j"] == mm
            ).astype(np.int8)
            (ip_indices,) = ip_indices.nonzero()
            uu = update_kernel_matrices_ip_uu(
                train_data, uu, k, kernel_params, ip_indices
            )
    return uu


def ss_pp(train_data, uu, k, kernel_params, params):
    for _ in range(params["pp_iterations"]):
        for i in range(len(uu["pp_uu"])):
            if params["surf_sample"]:
                uu, k = surf_slice_pp_uu(train_data, uu, k, kernel_params, params)
            else:
                raise Exception("Only Surf Slice Sampling implemented")
    return uu, k


def surf_slice_pp_uu(train_data, uu, k, kernel_params, params):
    (m, d) = uu["pp_uu"].shape
    sucess_counts = 0
    perm = np.arange(m)
    params["rng"].shuffle(perm)

    for mm in perm:
        log_pstar, k = cond_llh_pp_uu(train_data, uu, k, kernel_params, params, mm)
        log_pstar = log_pstar + np.log(params["rng"].uniform())

        direction = params["rng"].uniform(size=(d))
        direction = direction / np.sqrt(np.sum(np.square(direction)))

        rr = params["rng"].uniform()
        pp_l = uu["pp_uu"][mm] - rr * params["pp_slice_width"] * direction
        pp_r = uu["pp_uu"][mm] + (1 - rr) * params["pp_slice_width"] * direction

        pp_saved = uu["pp_uu"].copy()
        t_saved = uu["t_uu"].copy()

        not_mm = np.array(list(range(mm)) + list(range(mm + 1, m)))
        full_conditional_surf = (
            k["k_pp_pp_uu"][np.ix_([mm], not_mm)]
            @ np.linalg.lstsq(
                k["k_pp_pp_uu"][np.ix_(not_mm, not_mm)], uu["t_uu"][not_mm], rcond=-1
            )[0]
        )
        surf_height = uu["t_uu"][mm] - full_conditional_surf

        attempts = 0
        if params["pp_step_out"]:
            while attempts < params["pp_max_attempts"]:
                uu["pp_uu"][mm] = pp_l
                k = update_kernel_matrices_pp_uu(uu, k, kernel_params, mm)
                full_conditional_surf = (
                    k["k_pp_pp_uu"][np.ix_([mm], not_mm)]
                    @ np.linalg.lstsq(
                        k["k_pp_pp_uu"][np.ix_(not_mm, not_mm)],
                        uu["t_uu"][not_mm],
                        rcond=-1,
                    )[0]
                )
                uu["t_uu"][mm] = full_conditional_surf + surf_height
                test_p = cond_llh_pp_uu_no_update(train_data, uu, k, params)
                if test_p > log_pstar:
                    pp_l = uu["pp_uu"][mm] - params["pp_slice_width"] * direction
                else:
                    break
                attempts += 1
            while attempts < params["pp_max_attempts"]:
                uu["pp_uu"][mm] = pp_r
                k = update_kernel_matrices_pp_uu(uu, k, kernel_params, mm)
                full_conditional_surf = (
                    k["k_pp_pp_uu"][np.ix_([mm], not_mm)]
                    @ np.linalg.lstsq(
                        k["k_pp_pp_uu"][np.ix_(not_mm, not_mm)],
                        uu["t_uu"][not_mm],
                        rcond=-1,
                    )[0]
                )
                uu["t_uu"][mm] = full_conditional_surf + surf_height
                test_p = cond_llh_pp_uu_no_update(train_data, uu, k, params)
                if test_p > log_pstar:
                    pp_r = uu["pp_uu"][mm] + params["pp_slice_width"] * direction
                else:
                    break
                attempts += 1

        uu["pp_uu"][mm] = pp_saved[mm]
        uu["t_uu"][mm] = t_saved[mm]

        attempts = 0

        while attempts < params["pp_max_attempts"]:
            uu["pp_uu"][mm] = params["rng"].uniform() * (pp_r - pp_l) + pp_l
            k = update_kernel_matrices_pp_uu(uu, k, kernel_params, mm)
            full_conditional_surf = (
                k["k_pp_pp_uu"][np.ix_([mm], not_mm)]
                @ np.linalg.lstsq(
                    k["k_pp_pp_uu"][np.ix_(not_mm, not_mm)],
                    uu["t_uu"][not_mm],
                    rcond=-1,
                )[0]
            )
            uu["t_uu"][mm] = full_conditional_surf + surf_height
            log_p_prime = cond_llh_pp_uu_no_update(train_data, uu, k, params)
            if log_p_prime >= log_pstar:
                break
            else:
                if (uu["pp_uu"][mm] - pp_saved[mm]) @ direction > 0:
                    pp_r = uu["pp_uu"][mm]
                elif (uu["pp_uu"][mm] - pp_saved[mm]) @ direction < 0:
                    pp_l = uu["pp_uu"][mm]
                else:
                    raise (
                        "BUG DETECTED: Shrunk to current position and still not acceptable."
                    )
            attempts += 1
        if attempts < params["pp_max_attempts"]:
            sucess_counts += 1
        else:
            uu["pp_uu"][mm] = pp_saved[mm]
            uu["t_uu"][mm] = t_saved[mm]
            k = update_kernel_matrices_pp_uu(uu, k, kernel_params, mm)
        return uu, k


def ss_array_kern_params(train_data, uu, k, kernel_params, kernel_priors, params):
    slice_fn = lambda x: cond_llh_array_params_uu(
        train_data, uu, k, kernel_params, kernel_priors, params, x
    )
    x, uu, k = slice_sample_max(
        1,
        0,
        slice_fn,
        np.array([kernel_params["lls"], kernel_params["lsv"], kernel_params["ldn"]]),
        np.array(
            [
                params["lengthscale_slice_width"],
                params["signal_variance_slice_width"],
                params["diag_noise_slice_width"],
            ]
        ),
        params["kern_par_max_attempts"],
        params["rng"],
        params["kern_par_step_out"],
    )
    kernel_params["lls"], kernel_params["lsv"], kernel_params["ldn"] = x.flatten()
    k = update_kernel_matrices_uu(uu, k, kernel_params)
    return kernel_params, k


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
    return llh, uu, k
