import numpy as np
from Enumerations import KernelPriors, ObservationModels, InitialisationMethods


def establish_settings(params=None):
    params = {} if params is None else params
    params["seed"] = params.get("seed", 42)
    params["rng"] = np.random.default_rng(seed=params["seed"])
    params["uu_filename"] = params.get("uu_filename", "2Clique20.csv")
    params["uu_folds"] = params.get("uu_folds", 5)
    params["uu_fold"] = params.get("uu_fold", 1)
    params["init_log_length_scale"] = params.get("init_log_length_scale", np.log(1))
    params["init_log_signal_variance"] = params.get(
        "init_log_signal_variance", np.log(2)
    )
    params["init_log_diag_noise"] = params.get("init_log_diag_noise", np.log(0.1))

    params["kernel_prior_type"] = params.get(
        "kernel_prior_type", KernelPriors.LogNormals
    )

    if params["kernel_prior_type"] == KernelPriors.LogNormals:
        params["lengthscale_prior"] = params.get(
            "lengthscale_prior", np.array([np.log(1), 0.5])
        )
        params["signal_variance_prior"] = params.get(
            "signal_variance_prior", np.array([np.log(2), 0.5])
        )
    params["diag_noise_prior"] = params.get(
        "diag_noise_prior", np.array([np.log(0.1), 0.5])
    )

    params["kernel_jitter"] = params.get("kernel_jitter", 10e-6)
    params["uu_observation_model"] = params.get(
        "uu_observation_model", ObservationModels.Logit
    )
    params["u_sd"] = params.get("u_sd", 1)
    params["pp_uu_sd"] = params.get("pp_uu_sd", 1)
    params["n_pp_uu"] = params.get("n_pp_uu", 50)

    params["lv_modulus"] = params.get("lv_modulus", 1)
    params["pp_modulus"] = params.get("pp_modulus", 1)
    params["kern_par_modulus"] = params.get("kern_par_modulus", 1)
    params["data_par_modulus"] = params.get("data_par_modulus", 1)
    params["t_modulus"] = params.get("t_modulus", 1)
    params["t_iterations"] = params.get("t_iterations", 50)
    params["pp_iterations"] = params.get("pp_iterations", 10)
    params["burn"] = params.get("burn", 5)
    params["iterations"] = params.get("iterations", 20)
    params["plot_modulus"] = params.get("plot_modulus", 1)
    params["talk_modulus"] = params.get("talk_modulus", 1)
    params["surf_sample"] = params.get("surf_sample", True)

    # Slice Sampling Parameters
    params["diag_noise_slice_width"] = params.get("diag_noise_slice_width", 0.1)
    params["lengthscale_slice_width"] = params.get("lengthscale_slice_width", 0.5)
    params["signal_variance_slice_width"] = params.get(
        "signal_variance_slice_width", 0.5
    )
    params["u_slice_width"] = params.get("u_slice_width", 4)
    params["u_step_out"] = params.get("u_step_out", True)
    params["u_max_attempts"] = params.get("u_max_attempts", 6)

    params["pp_slice_width"] = params.get("pp_slice_width", 4)
    params["pp_step_out"] = params.get("pp_step_out", False)
    params["pp_max_attempts"] = params.get("pp_max_attempts", 6)

    params["kern_par_step_out"] = params.get("kern_par_step_out", False)
    params["kern_par_max_attempts"] = params.get("kern_par_max_attempts", 6)

    params["init_method"] = params.get("init_method", InitialisationMethods.NONE)

    params["save_filename"] = params.get("save_filename", "DefaultSaveFile.pickle")
    params["save_traces"] = params.get("save_filename", False)
    params["save_prediction"] = params.get("save_filename", False)

    return params
