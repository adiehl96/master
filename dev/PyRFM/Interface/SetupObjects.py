import numpy as np
from Enumerations.KernelPriors import KernelPriors
from Enumerations.ObservationModels import ObservationModels
from Enumerations.InitialisationMethods import InitialisationMethods
from HCRFM.HCRFM import HCRFM
from HCMCMC_SS_RFM.HCMCMC_SS_RFM import HCMCMC_SS_RFM
from HC2ArrayData.HC2ArrayData import HC2ArrayData


def setup_objects(params=None):
    params = {} if params is None else params
    seed = params.get("seed", 42)
    rng = np.random.default_rng(seed=seed)
    u_dim = params.get("u_dim", 1)
    uu_filename = params.get("uu_filename", "2Clique20.csv")
    uu_folds = params.get("uu_folds", 5)
    uu_fold = params.get("uu_fold", 1)

    uu_kernel_name = params.get("uu_kernel_name", "covSEard_sym")
    uu_diag_noise = params.get("uu_diag_noise", np.log(0.1))
    if uu_kernel_name == "covSEiso_sym" or uu_kernel_name == "covSEiso":
        dims = 1
    elif uu_kernel_name == "covSEard_sym":
        dims = u_dim
    else:
        dims = 2 * u_dim

    uu_kernel_params = params.get(
        "uu_kernel_params", [np.log(1) * np.ones((dims)), np.log(2)]
    )
    uu_prior = params.get("uu_prior", KernelPriors.InverseGammas)

    if uu_prior == KernelPriors.LogNormals:
        param_defaults = [uu_kernel_params, 0.5 * np.ones((len(uu_kernel_params), 1))]
        uu_noise_params = [np.log(0.1), 0.5]
    elif uu_prior == KernelPriors.InverseGammas:
        param_defaults = 0.1 * np.ones((len(uu_kernel_params), 2))
        uu_noise_params = 0.1 * np.ones((1, 2))
    else:
        raise Exception("uu_prior not defined")

    uu_kern_prior_params = params.get("uu_kern_prior_params", param_defaults)

    uu_observation_model = params.get("uu_observation_model", ObservationModels.Logit)
    uu_data_precision = params.get("uu_data_precision", 1)
    n_pp_uu = params.get("n_pp_uu", 50)

    # MCMC

    lv_modulus = params.get("lv_modulus", 1)
    pp_modulus = params.get("pp_modulus", 1)
    kern_par_modulus = params.get("kern_par_modulus", 1)
    data_par_modulus = params.get("data_par_modulus", 1)
    t_modulus = params.get("t_modulus", 1)
    t_iterations = params.get("t_iterations", 50)
    pp_iterations = params.get("pp_iterations", 10)
    burn = params.get("burn", 5)
    iterations = params.get("iterations", 20)
    plot_modulus = params.get("plot_modulus", 1)
    talk_modulus = params.get("talk_modulus", 1)

    surf_sample = params.get("surf_sample", True)

    uu_kern_par_widths = params.get(
        "uu_kern_par_widths", [0.5 * np.ones((len(uu_kernel_params), 1)), 0.1]
    )
    init_method = params.get("init_method", InitialisationMethods.NONE)

    # Setup

    # % Load path and create objects

    # generic_startup; # Not needed in python

    rfm = HCRFM(rng)
    mcmc = HCMCMC_SS_RFM()
    mcmc.rfm = rfm

    rfm.d_l_u = u_dim

    # Load UU array

    uu_data = HC2ArrayData(uu_filename)
    data = uu_data.partition(uu_folds, uu_fold, rng)
    rfm.data_uu = data

    rfm.observation_model_uu = uu_observation_model
    rfm.dataprecision_uu = uu_data_precision

    # Setup UU kernel

    rfm.array_kern_uu.name = uu_kernel_name
    rfm.array_kern_uu.diag_noise = uu_diag_noise
    rfm.array_kern_uu.params = uu_kernel_params
    rfm.array_kern_uu.prior_params = uu_kern_prior_params
    rfm.array_kern_uu.noise_params = uu_noise_params
    rfm.array_kern_uu.priorType = uu_prior

    rfm.n_pp_uu = n_pp_uu

    # Set MCMC parameters

    mcmc.burn = burn
    mcmc.iterations = iterations

    mcmc.lv_modulus = lv_modulus
    mcmc.pp_modulus = pp_modulus
    mcmc.pp_iterations = pp_iterations
    mcmc.kern_par_modulus = kern_par_modulus
    mcmc.data_par_modulus = data_par_modulus
    mcmc.t_modules = t_modulus
    mcmc.t_iterations = t_iterations
    mcmc.plot_modulus = plot_modulus
    mcmc.talk_modulus = talk_modulus

    mcmc.surf_sample = surf_sample

    mcmc.uu_kern_par_widths = uu_kern_par_widths

    mcmc.init_method = init_method

    return rfm, mcmc
