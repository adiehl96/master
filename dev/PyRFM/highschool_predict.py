from Enumerations import ObservationModels, KernelPriors
from Main import rfm_experiment_refactored


params = {}

params["save_filename"] = "temp.mat"
params["uu_filename"] = "HighSchool.csv"
params["u_dim"] = 1  # Number of latent dimensions
params["uu_folds"] = 5  # ; % Number of folds for cross validation
params["uu_fold"] = 1  # ; % Fold to use
params["burn"] = 10  # ;
params["iterations"] = 100  # ;
params["plot_modulus"] = 1000000  # ; % How often to plot - i.e. never
params["uu_observation_model"] = ObservationModels.Logit  # ;
params["n_pp_uu"] = 50  # ; % Number of inducing points
params["seed"] = 1  # ; % Random seed
params["kernel_prior_type"] = KernelPriors.LogNormals
params["pp_iterations"] = 1

rfm_experiment_refactored(params)
