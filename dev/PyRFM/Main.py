import numpy as np
from scipy.linalg import cholesky

from Initialisation import initialise_pp_uu
from Kernel import matrix
from MCMC import slice_u, ss_array_kern_params, ss_pp, gppu_elliptical
from RFM import permute
from Predicting import prediction, performance
from Settings import establish_settings
from DataHandling import load_partitioned_data
from Utilities.CondLlh2Array import cond_llh_2array
from Utilities.CreateGPInputPoints import create_gp_input_points
from Utilities.Permutation import new_permutation
from OutputHandling import talk


def rfm_experiment_refactored(params):
    params = {} if params is None else params
    params = establish_settings(params)

    train_data, test_data, network_size = load_partitioned_data(
        params["uu_filename"], params["uu_folds"], params["uu_fold"], params["rng"]
    )

    # Initialise kernel parameters
    kernel_params = {}
    kernel_params["lls"] = params["init_log_length_scale"]
    kernel_params["lsv"] = params["init_log_signal_variance"]
    kernel_params["ldn"] = params["init_log_diag_noise"]

    # Initialise hidden variables
    uu = {}
    uu["u"] = params["rng"].standard_normal((network_size))
    uu["pp_uu"] = initialise_pp_uu(
        train_data["i"], train_data["j"], uu["u"], params["n_pp_uu"], params["rng"]
    )
    uu["t_uu"] = params["rng"].standard_normal((params["n_pp_uu"]))
    uu["ip_uu"] = create_gp_input_points(train_data["i"], train_data["j"], uu["u"])
    uu["pred_ip_uu"] = create_gp_input_points(test_data["i"], test_data["j"], uu["u"])
    uu["w_uu"] = np.zeros((len(train_data["v"])))
    uu["pred_uu"] = np.zeros((len(test_data["v"])))

    # Initialise covariance matrices
    k = {}
    k["k_ip_pp_uu"] = matrix(kernel_params, uu["ip_uu"], uu["pp_uu"])
    k["k_pp_pp_uu"] = matrix(kernel_params, uu["pp_uu"])
    k["chol_k_pp_pp_uu"] = cholesky(k["k_pp_pp_uu"])
    k["k_pred_pp_uu"] = matrix(kernel_params, uu["pred_ip_uu"], uu["pp_uu"])

    # rfm_states = []
    predictions = []
    performances = []
    for i in range(params["burn"] + params["iterations"]):
        perm, iperm = new_permutation(len(train_data["v"]), params["rng"])
        train_data, uu, k = permute(train_data, uu, k, perm)

        if i % params["t_modulus"] == 0:
            k_pp_pp_invk_ppip_t = np.linalg.lstsq(
                k["k_pp_pp_uu"], k["k_ip_pp_uu"].T, rcond=-1
            )[0].T
            half_t_iterations = int(params["t_iterations"] / 2)
            for _ in range(half_t_iterations):
                llh_fn = lambda T: cond_llh_2array(
                    k_pp_pp_invk_ppip_t @ T,
                    train_data["v"],
                    params["uu_observation_model"],
                )
                uu["t_uu"] = gppu_elliptical(
                    uu["t_uu"], k["chol_k_pp_pp_uu"], llh_fn, params["rng"]
                )
        if i % params["lv_modulus"] == 0:
            uu = slice_u(train_data, uu, k, kernel_params, params)
        if i % params["pp_modulus"] == 0:
            uu, k = ss_pp(train_data, uu, k, kernel_params, params)
        if i % params["kern_par_modulus"] == 0:
            kernel_priors = np.array(
                [
                    params["lengthscale_prior"],
                    params["signal_variance_prior"],
                    params["diag_noise_prior"],
                ]
            )
            kernel_params, uu, k = ss_array_kern_params(
                train_data, uu, k, kernel_params, kernel_priors, params
            )
        if i % params["t_modulus"] == 0:
            k_pp_pp_invk_ppip_t = np.linalg.lstsq(
                k["k_pp_pp_uu"], k["k_ip_pp_uu"].T, rcond=-1
            )[0].T

            half_t_iterations = int(params["t_iterations"] / 2)
            for _ in range(half_t_iterations):
                llh_fn = lambda T: cond_llh_2array(
                    k_pp_pp_invk_ppip_t @ T,
                    train_data["v"],
                    params["uu_observation_model"],
                )
                uu["t_uu"] = gppu_elliptical(
                    uu["t_uu"], k["chol_k_pp_pp_uu"], llh_fn, params["rng"]
                )
        train_data, uu, k = permute(train_data, uu, k, iperm)

        # rfm_states.append(self.rfm.state())
        prediction_uu, uu, k = prediction(test_data, uu, k, kernel_params, params)
        predictions.append(prediction_uu)
        performance_uu, uu, k = performance(test_data, uu, k, kernel_params, params)
        performances.append(performance_uu)

        # if (
        #     i % params["plot_modulus"] == 0
        #     and params["plot_modulus"] < params["iterations"]
        # ):
        #     plot()

        if i % params["talk_modulus"] == 0:
            print(f"Iter {i-params['burn']} / {params['iterations']}")
            talk(params, performance_uu)
            print("")

    avg_predictions = np.array(predictions).sum(axis=0) / params["iterations"]
    avg_performance_uu, _, _ = performance(
        test_data, uu, k, kernel_params, params, False, avg_predictions
    )
    print("\n **** Average performance **** \n")
    talk(params, avg_performance_uu)
