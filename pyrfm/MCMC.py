import numpy as np
from RFM import (
    cond_llh_array_params_uu,
    cond_llh_pp_uu,
    cond_llh_pp_uu_no_update,
    cond_llh_u,
    update_kernel_matrices_ip_uu,
    update_kernel_matrices_pp_uu,
    update_kernel_matrices_uu,
)


def slice_u(train_data, uu, k, kernel_params, params):
    (m,) = uu["u"].shape
    perm = np.arange(m)
    params["rng"].shuffle(perm)
    u_saved = uu["u"].copy()
    
    for mm in perm:
        log_pstar, uu = cond_llh_u(train_data, uu, k, kernel_params, params, mm)
        log_pstar = log_pstar + np.log(params["rng"].uniform())

        direction = params["rng"].uniform()
        direction = direction / np.sqrt(np.sum(np.square(direction)))

        rr = params["rng"].uniform()
        u_l = uu["u"][mm] - rr * params["u_slice_width"] * direction
        u_r = uu["u"][mm] + (1 - rr) * params["u_slice_width"] * direction
        

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
            uu, k = update_kernel_matrices_ip_uu(
                train_data, uu, k, kernel_params, ip_indices
            )
    return uu, k


def ss_pp(train_data, uu, k, kernel_params, params):
    for _ in range(params["pp_iterations"]):
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
        log_pstar, uu, k = cond_llh_pp_uu(train_data, uu, k, kernel_params, params, mm)
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
                test_p, uu = cond_llh_pp_uu_no_update(train_data, uu, k, params)
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
                test_p, uu = cond_llh_pp_uu_no_update(train_data, uu, k, params)
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
            log_p_prime, uu = cond_llh_pp_uu_no_update(train_data, uu, k, params)
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


def gppu_elliptical(xx, chol_sigma, log_like_fn, rng, angle_range=0):
    cur_log_like = log_like_fn(xx)

    dimension = len(xx)
    if chol_sigma.shape != (dimension, dimension):
        raise Exception("chol_sigma has the wrong dimension")

    randnumbers = rng.standard_normal((dimension))
    nu = (chol_sigma.conj().T @ randnumbers).reshape(xx.shape, order="F").copy()
    hh = np.log(rng.uniform()) + cur_log_like

    if angle_range <= 0:
        phi = rng.uniform() * 2 * np.pi
        phi_min = phi - 2 * np.pi
        phi_max = phi
    else:
        phi_min = -angle_range * rng.uniform()
        phi_max = phi_min + angle_range
        phi = rng.uniform() * (phi_max - phi_min) + phi_min

    while True:
        xx_prop = xx * np.cos(phi) + nu * np.sin(phi)
        cur_log_like = log_like_fn(xx_prop)
        if cur_log_like > hh:
            return xx_prop
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception(
                "BUG DETECTED: Shrunk to current position and still not acceptable."
            )
        phi = rng.uniform() * (phi_max - phi_min) + phi_min


def ss_array_kern_params(train_data, uu, k, kernel_params, kernel_priors, params):
    slice_fn = lambda x: cond_llh_array_params_uu(
        train_data, uu, k, kernel_params, kernel_priors, params, x
    )
    x, kernel_params, uu, k = slice_sample_max(
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
    return kernel_params, uu, k


def slice_sample_max(logdist, xx, widths, max_attempts, rng, step_out=False):
    dimension = len(xx)
    log_px, kernel_params, uu, k = logdist(xx)
    log_uprime = np.log(rng.uniform()) + log_px

    perm = np.arange(dimension)
    rng.shuffle(perm)
    for dd in perm:
        x_l = xx.copy()
        x_r = xx.copy()
        xprime = xx.copy()

        rr = rng.uniform()
        x_l[dd] = xx[dd] - rr * widths[dd]
        x_r[dd] = xx[dd] + (1 - rr) * widths[dd]

        if step_out:
            raise Exception("step_out not implemented")

        zz = 0
        num_attempts = 0
        while True:
            zz += 1
            xprime[dd] = rng.uniform() * (x_r[dd] - x_l[dd]) + x_l[dd]
            log_px, kernel_params, uu, k = logdist(xprime)
            if log_px > log_uprime:
                xx[dd] = xprime[dd]
                break
            else:
                num_attempts += 1
                if num_attempts >= max_attempts:
                    break
                elif xprime[dd] > xx[dd]:
                    x_r[dd] = xprime[dd]
                elif xprime[dd] < xx[dd]:
                    x_l[dd] = xprime[dd]
                else:
                    # raise Exception("BUG DETECTED: Shrunk to current position and still not acceptable")
                    break
    return xx, kernel_params, uu, k
