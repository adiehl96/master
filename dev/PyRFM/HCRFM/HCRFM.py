import numpy as np
from Enumerations import ObservationModels
from VCKernel.VCKernel import VCKernel
from Utilities.CondLlh2Array import cond_llh_2array
from Utilities.CreateGPInputPoints import create_gp_input_points
from scipy.linalg import cholesky, cho_solve
from SliceSampling.SliceSampleMax import slice_sample_max
from SliceSampling.GppuElliptical import gppu_elliptical
from scipy.special import expit
from Utilities.CalcBinErrorStats import calc_bin_error_stats
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class HCRFM:
    def __init__(self, rng):

        self.rng = rng
        # Local reference to data

        self.data_uu = None

        # Dimensions / specifications of model

        self.d_l_u = 1  # Number of latent row dimensions
        self.n_pp_uu = 50  # Number of pseudo points

        # Prior specification

        self.observation_model_uu = ObservationModels.Logit

        self.data_precision_uu = 1

        self.array_kern_uu = (
            VCKernel()
        )  # Kernel for 2-array (same for each array dimension)
        self.u_sd = 1
        self.pp_uu_sd = 1

        # Variables

        self.u = np.array([])
        self.pp_uu = np.array([])
        self.t_uu = np.array([])

        # Cached quantities

        self.ip_uu = np.array([])  # 2-array training input points

        self.pred_ip_uu = np.array([])  # 2-array test input points
        self.k_ip_pp_uu = np.array([])  # Kernel matrix between input and pseudo points
        self.k_pp_pp_uu = np.array([])  # Kernel matrix between pseudo points
        self.chol_k_pp_pp_uu = np.array([])  # Cholesky decompostion of above
        self.perm_uu = np.array([])  # Saved permutation for MCMC mixing
        self.iperm_uu = np.array([])  # Saved inverse permutation for MCMC mixing

        self.prediction_uu = np.array([])  # Saved predictions for each data set

        # Memory pre-allocation

        self.w_uu = np.array([])  # Output of GP at input points
        self.k_pred_pp_uu = np.array(
            []
        )  # Kernel matrix between prediction and pseudo points

    # Prior

    def prior_u(self):
        flat_u = self.u.flatten(order="F")
        return (
            -0.5 * (flat_u.T @ flat_u) / np.square(self.u_sd)
        )  # todo check on matlab notation (obj.u(:)')

    def prior_pp_uu(self):
        flat_pp_uu = self.pp_uu.flatten(order="F")
        return -0.5 * (flat_pp_uu.T @ flat_pp_uu) / np.square(self.pp_uu_sd)

    # Likelihoods

    def array_llh_uu(self):
        llh = 0
        llh = (
            llh
            - np.sum(np.log(np.diag(self.chol_k_pp_pp_uu)))
            - 0.5 * (self.t_uu @ cho_solve((self.chol_k_pp_pp_uu, False), self.t_uu))
        )  # todo check how solve chol can be replaced in python
        self.w_uu = (
            self.k_ip_pp_uu @ np.linalg.lstsq(self.k_pp_pp_uu, self.t_uu, rcond=-1)[0]
        )
        params = {"precision": self.data_precision_uu}
        llh = llh + cond_llh_2array(
            self.w_uu,
            self.data_uu.train_x_v,
            self.observation_model_uu,
            params,
        )
        llh = llh + self.prior_pp_uu()

        if self.data_uu.train_x_v.size != 0:
            llh = llh + self.array_kern_uu.prior()
        return llh

    def cond_llh_array_params_uu(self, new_params):
        self.array_kern_uu.params = new_params[:-1]
        self.array_kern_uu.diag_noise = new_params[-1]
        self.update_kernel_matrices_uu()
        return self.array_llh_uu()

    def cond_llh_pp_uu_no_update(self):
        llh = -np.sum(np.log(np.diag(self.chol_k_pp_pp_uu))) - 0.5 * (
            self.t_uu @ cho_solve((self.chol_k_pp_pp_uu, False), self.t_uu)
        )
        self.w_uu = (
            self.k_ip_pp_uu @ np.linalg.lstsq(self.k_pp_pp_uu, self.t_uu, rcond=-1)[0]
        )
        params = {"precision": self.data_precision_uu}
        llh = llh + cond_llh_2array(
            self.w_uu, self.data_uu.train_x_v, self.observation_model_uu, params
        )
        llh = llh + self.prior_pp_uu()
        return llh

    def cond_llh_pp_uu(self, pp_index):
        self.update_kernel_matrices_pp_uu(pp_index)
        return self.cond_llh_pp_uu_no_update()

    def cond_llh_u(self, u_index):
        llh = 0
        active = (self.data_uu.train_x_i == u_index).astype(np.int8) + (
            self.data_uu.train_x_j == u_index
        ).astype(np.int8)
        (active,) = active.nonzero()
        self.update_kernel_matrices_ip_uu(active)
        self.w_uu[active] = (
            self.k_ip_pp_uu[active]
            @ np.linalg.lstsq(self.k_pp_pp_uu, self.t_uu, rcond=-1)[0]
        )
        params = {"precision": self.data_precision_uu}
        llh = llh + cond_llh_2array(
            self.w_uu[active],
            self.data_uu.train_x_v[active],
            self.observation_model_uu,
            params,
        )

        if self.data_uu.train_x_v.size != 0:
            llh = llh + self.prior_u()

        return llh

    # MCMC routines
    def init_u_rand(self):
        if self.data_uu.m == 0:
            rows = 0
        else:
            rows = self.data_uu.m
        self.u = self.rng.standard_normal((rows, self.d_l_u))

    def init_pp_t(self):
        self.pp_uu = np.zeros((len(self.data_uu.train_x_v), 1))
        self.t_uu = self.rng.standard_normal((self.n_pp_uu))
        if self.n_pp_uu <= len(self.data_uu.train_x_v):
            perm = np.arange(len(self.data_uu.train_x_v))
            self.rng.shuffle(perm)
            rand_subset = perm[: self.n_pp_uu]
            self.pp_uu = create_gp_input_points(
                self.data_uu.train_x_i[rand_subset],
                self.data_uu.train_x_j[rand_subset],
                self.u,
            )
        else:
            raise Exception(
                "Fewer datapoints than pseudo inputs is currently not supported."
            )

    def init_cache(self):
        self.ip_uu = create_gp_input_points(
            self.data_uu.train_x_i,
            self.data_uu.train_x_j,
            self.u,
        )
        self.pred_ip_uu = create_gp_input_points(
            self.data_uu.test_x_i,
            self.data_uu.test_x_j,
            self.u,
        )
        self.w_uu = np.zeros((len(self.data_uu.train_x_i)))
        self.prediction_uu = np.zeros((len(self.data_uu.test_x_i)))
        self.k_ip_pp_uu = self.array_kern_uu.matrix(self.ip_uu, self.pp_uu)
        self.k_pp_pp_uu = self.array_kern_uu.matrix(self.pp_uu)
        self.chol_k_pp_pp_uu = cholesky(self.k_pp_pp_uu)
        self.k_pred_pp_uu = self.array_kern_uu.matrix(self.pred_ip_uu, self.pp_uu)

    def initialise_pca(self):
        self.init_u_rand()
        self.init_pp_t()
        self.init_cache()

    def initialise_rand(self):
        self.init_u_rand()
        self.init_pp_t()
        self.init_cache()

    def ess_t(self, iterations):
        k_pp_pp_invk_ppip_t = np.linalg.lstsq(
            self.k_pp_pp_uu, self.k_ip_pp_uu.T, rcond=-1
        )[0].T
        for _ in range(iterations):
            params = {"precision": self.data_precision_uu}
            llh_fn = lambda T: cond_llh_2array(
                k_pp_pp_invk_ppip_t @ T,
                self.data_uu.train_x_v,
                self.observation_model_uu,
                params,
            )
            self.t_uu = gppu_elliptical(
                self.t_uu, self.chol_k_pp_pp_uu, llh_fn, self.rng
            )

    def ss_array_kern_params(self, widths, step_out, max_attempts):
        slice_fn = lambda x: self.cond_llh_array_params_uu(x)
        x = slice_sample_max(
            1,
            0,
            slice_fn,
            np.hstack((self.array_kern_uu.params, [self.array_kern_uu.diag_noise])),
            list(widths[0].flatten()) + widths[1:],
            max_attempts,
            self.rng,
            step_out,
        ).flatten()
        self.array_kern_uu.params = x[:-1]
        self.array_kern_uu.diag_noise = x[-1]
        self.update_kernel_matrices_uu()

    def surf_slice_pp_uu(self, slice_width, step_out, max_attempts):
        (m, d) = self.pp_uu.shape
        sucess_counts = 0
        perm = np.arange(m)
        self.rng.shuffle(perm)

        for mm in perm:
            log_pstar = self.cond_llh_pp_uu(mm)
            log_pstar = log_pstar + np.log(self.rng.uniform())

            direction = self.rng.uniform(size=(d))
            direction = direction / np.sqrt(np.sum(np.square(direction)))

            rr = self.rng.uniform()
            pp_l = self.pp_uu[mm] - rr * slice_width * direction
            pp_r = self.pp_uu[mm] + (1 - rr) * slice_width * direction

            pp_saved = self.pp_uu.copy()
            t_saved = self.t_uu.copy()

            not_mm = np.array(list(range(mm)) + list(range(mm + 1, m)))
            full_conditional_surf = (
                self.k_pp_pp_uu[np.ix_([mm], not_mm)]
                @ np.linalg.lstsq(
                    self.k_pp_pp_uu[np.ix_(not_mm, not_mm)], self.t_uu[not_mm], rcond=-1
                )[0]
            )
            surf_height = self.t_uu[mm] - full_conditional_surf

            attempts = 0
            if step_out:
                while attempts < max_attempts:
                    self.pp_uu[mm] = pp_l
                    self.update_kernel_matrices_pp_uu(mm)
                    full_conditional_surf = (
                        self.k_pp_pp_uu[np.ix_([mm], not_mm)]
                        @ np.linalg.lstsq(
                            self.k_pp_pp_uu[np.ix_(not_mm, not_mm)],
                            self.t_uu[not_mm],
                            rcond=-1,
                        )[0]
                    )
                    self.t_uu[mm] = full_conditional_surf + surf_height
                    test_p = self.cond_llh_pp_uu_no_update()
                    if test_p > log_pstar:
                        pp_l = self.pp_uu[mm] - slice_width * direction
                    else:
                        break
                    attempts += 1
                while attempts < max_attempts:
                    self.pp_uu[mm] = pp_r
                    self.update_kernel_matrices_pp_uu(mm)
                    full_conditional_surf = (
                        self.k_pp_pp_uu[np.ix_([mm], not_mm)]
                        @ np.linalg.lstsq(
                            self.k_pp_pp_uu[np.ix_(not_mm, not_mm)],
                            self.t_uu[not_mm],
                            rcond=-1,
                        )[0]
                    )
                    self.t_uu[mm] = full_conditional_surf + surf_height
                    test_p = self.cond_llh_pp_uu_no_update()
                    if test_p > log_pstar:
                        pp_r = self.pp_uu[mm] + slice_width * direction
                    else:
                        break
                    attempts += 1

            self.pp_uu[mm] = pp_saved[mm]
            self.t_uu[mm] = t_saved[mm]

            attempts = 0

            while attempts < max_attempts:
                self.pp_uu[mm] = self.rng.uniform() * (pp_r - pp_l) + pp_l
                self.update_kernel_matrices_pp_uu(mm)
                full_conditional_surf = (
                    self.k_pp_pp_uu[np.ix_([mm], not_mm)]
                    @ np.linalg.lstsq(
                        self.k_pp_pp_uu[np.ix_(not_mm, not_mm)],
                        self.t_uu[not_mm],
                        rcond=-1,
                    )[0]
                )
                self.t_uu[mm] = full_conditional_surf + surf_height
                log_p_prime = self.cond_llh_pp_uu_no_update()
                if log_p_prime >= log_pstar:
                    break
                else:
                    if (self.pp_uu[mm] - pp_saved[mm]) @ direction > 0:
                        pp_r = self.pp_uu[mm]
                    elif (self.pp_uu[mm] - pp_saved[mm]) @ direction < 0:
                        pp_l = self.pp_uu[mm]
                    else:
                        raise (
                            "BUG DETECTED: Shrunk to current position and still not acceptable."
                        )
                attempts += 1
            if attempts < max_attempts:
                sucess_counts += 1
            else:
                self.pp_uu[mm] = pp_saved[mm]
                self.t_uu[mm] = t_saved[mm]
                self.update_kernel_matrices_pp_uu(mm)
            return sucess_counts / m

    def slice_u(self, slice_width, step_out, max_attempts):
        (m, d) = self.u.shape
        perm = np.arange(m)
        self.rng.shuffle(perm)
        for mm in perm:
            log_pstar = self.cond_llh_u(mm)
            log_pstar = log_pstar + np.log(self.rng.uniform())

            direction = self.rng.uniform(size=(d))
            direction = direction / np.sqrt(np.sum(np.square(direction)))

            rr = self.rng.uniform()
            u_l = self.u[mm] - rr * slice_width * direction
            u_r = self.u[mm] + (1 - rr) * slice_width * direction
            u_saved = self.u.copy()

            attempts = 0

            if step_out:
                while attempts < max_attempts:
                    self.u[mm] = u_l
                    test_p = self.cond_llh_u(mm)
                    if test_p > log_pstar:
                        u_l = self.u[mm] - slice_width * direction
                    else:
                        break
                    attempts += 1
                self.u[mm] = u_saved[mm]
                while attempts < max_attempts:
                    self.u[mm] = u_r
                    test_p = self.cond_llh_u(mm)
                    if test_p > log_pstar:
                        u_r = self.u[mm] + slice_width * direction
                    else:
                        break
                    attempts += 1

            self.u[mm] = u_saved[mm]
            attempts = 0

            while attempts < max_attempts:
                self.u[mm] = self.rng.uniform() * (u_r - u_l) + u_l
                log_p_prime = self.cond_llh_u(mm)
                if log_p_prime >= log_pstar:
                    break
                else:
                    if (self.u[mm] - u_saved[mm]) * direction > 0:
                        u_r = self.u[mm]
                    elif (self.u[mm] - u_saved[mm]) * direction < 0:
                        u_l = self.u[mm]
                    else:
                        raise Exception(
                            "BUG DETECTED: Shrunk to current position and still not acceptable."
                        )
                attempts += 1
            if attempts >= max_attempts:
                self.u[mm] = u_saved[mm]
                ip_indices = (self.data_uu.train_x_i == mm).astype(np.int8) + (
                    self.data_uu.train_x_j == mm
                ).astype(np.int8)
                (ip_indices,) = ip_indices.nonzero()
                self.update_kernel_matrices_ip_uu(ip_indices)

    def slice_pp_uu(self, array_index, slice_width, step_out, max_attempts):
        raise Exception("slice_pp_u not yet implemented")

    def ss_pp(self, iterations, width, step_out, max_attempts, surf=True):
        for _ in range(iterations):
            for i in range(len(self.pp_uu)):
                if surf:
                    self.surf_slice_pp_uu(width, step_out, max_attempts)
                else:
                    self.slice_pp_uu(width, step_out, max_attempts)

    def ss_u(self, width, step_out, max_attempts):
        self.slice_u(width, step_out, max_attempts)

    def state(self):  # Returns a struct with the current variable values
        state = {}
        state["u"] = self.u
        state["pp_uu"] = self.pp_uu
        state["t_uu"] = self.t_uu
        state["array_kern_uu"] = self.array_kern_uu
        state["data_precision_uu"] = self.data_precision_uu
        state["llh"] = self.llh()
        return state

    def prediction(self):  # Returns a cell with predictions
        self.pred_ip_uu = create_gp_input_points(
            self.data_uu.test_x_i, self.data_uu.test_x_j, self.u
        )
        self.k_pred_pp_uu = self.array_kern_uu.matrix(self.pred_ip_uu, self.pp_uu)
        if self.observation_model_uu == ObservationModels.Logit:
            self.prediction_uu = expit(
                self.k_pred_pp_uu
                @ np.linalg.lstsq(self.k_pp_pp_uu, self.t_uu, rcond=-1)[0]
            )
        if self.observation_model_uu == ObservationModels.Gaussian:
            raise Exception("Only ObservationModels Logit Implemented")
            self.prediction_uu[i] = self.k_pred_pp_uu[i] * (
                self.k_pp_pp_uu[i] / self.t_uu[i]
            )
        if self.observation_model_uu == ObservationModels.Poisson:
            raise Exception("Only ObservationModels Logit Implemented")
            self.prediction_uu[i] = np.exp(
                self.k_pred_pp_uu[i] * (self.k_pp_pp_uu[i] / self.t_uu[i])
            )
        return self.prediction_uu

    def performance(
        self, predict, prediction_uu
    ):  # Returns a struct with various error parameters
        if predict:
            self.prediction()
            prediction_uu = np.array([])

        return self.evaluate_performance_uu(prediction_uu)

    def evaluate_performance_uu(
        self, prediction_uu=None
    ):  # Returns a struct with various error parameters
        if prediction_uu is not None and prediction_uu.size != 0:
            self.prediction_uu = prediction_uu

        performance = np.empty((len(self.prediction_uu), 1))
        if self.observation_model_uu == ObservationModels.Logit:
            performance = calc_bin_error_stats(
                self.prediction_uu, self.data_uu.test_x_v
            )
        elif (
            self.observation_model_uu == ObservationModels.Gaussian
            or self.observation_model_uu == ObservationModels.Poisson
        ):
            raise Exception("Only ObservationModels Logit Implemented")
        return performance

    # utilities

    def duplicate(self):
        return deepcopy(self)

    def llh(self):  # Full log likelihood; calculated using cache
        llh = 0
        llh = llh + self.array_kern_uu.prior()
        llh = llh + self.prior_u()

        llh = llh + self.prior_pp_uu()
        llh = (
            llh
            - np.sum(np.log(np.diag(self.chol_k_pp_pp_uu)))
            - 0.5 * (self.t_uu @ cho_solve((self.chol_k_pp_pp_uu, False), self.t_uu))
        )
        self.update_w_uu()
        llh = llh + cond_llh_2array(
            self.w_uu,
            self.data_uu.train_x_v,
            self.observation_model_uu,
            {"precision": self.data_precision_uu},
        )

    def update_kernel_matrices_uu(self):
        self.k_ip_pp_uu = self.array_kern_uu.matrix(self.ip_uu, self.pp_uu)
        self.k_pp_pp_uu = self.array_kern_uu.matrix(self.pp_uu)
        self.chol_k_pp_pp_uu = cholesky(self.k_pp_pp_uu)

    def update_kernel_matrices_ip_uu(self, ip_indices):

        self.ip_uu[ip_indices] = create_gp_input_points(
            self.data_uu.train_x_i[ip_indices],
            self.data_uu.train_x_j[ip_indices],
            self.u,
        )
        self.k_ip_pp_uu[ip_indices] = self.array_kern_uu.matrix(
            self.ip_uu[ip_indices], self.pp_uu
        )

    def update_kernel_matrices_pp_uu(self, pp_index):
        self.k_pp_pp_uu[:, pp_index] = self.array_kern_uu.matrix(
            self.pp_uu, self.pp_uu[pp_index]
        ).flatten()
        self.k_pp_pp_uu[pp_index] = self.k_pp_pp_uu[:, pp_index]

        self.k_pp_pp_uu[pp_index, pp_index] = self.array_kern_uu.matrix(
            self.pp_uu[pp_index]
        )
        self.chol_k_pp_pp_uu = cholesky(self.k_pp_pp_uu)
        self.k_ip_pp_uu[:, pp_index] = self.array_kern_uu.matrix(
            self.ip_uu, self.pp_uu[pp_index]
        ).flatten()

    def update_w_uu(self):
        self.w_uu = (
            self.k_ip_pp_uu @ np.linalg.lstsq(self.k_pp_pp_uu, self.t_uu, rcond=-1)[0]
        )

    def new_permutation(self):
        perm = np.arange(len(self.data_uu.train_x_v))
        self.rng.shuffle(perm)
        self.perm_uu = perm
        self.iperm_uu = np.argsort(perm)

    def permute(self):
        self.data_uu.train_x_i = self.data_uu.train_x_i[self.perm_uu]
        self.data_uu.train_x_j = self.data_uu.train_x_j[self.perm_uu]
        self.data_uu.train_x_v = self.data_uu.train_x_v[self.perm_uu]
        self.ip_uu = self.ip_uu[self.perm_uu]
        self.k_ip_pp_uu = self.k_ip_pp_uu[self.perm_uu]

    def inverse_permute(self):
        self.data_uu.train_x_i = self.data_uu.train_x_i[self.iperm_uu]
        self.data_uu.train_x_j = self.data_uu.train_x_j[self.iperm_uu]
        self.data_uu.train_x_v = self.data_uu.train_x_v[self.iperm_uu]
        self.ip_uu = self.ip_uu[self.iperm_uu]
        self.k_ip_pp_uu = self.k_ip_pp_uu[self.iperm_uu]

    def plot(self):
        raise Exception("Not yet implemented, still todo")
        if self.data_uu.m.size != 0:
            n_1d = 100
            min_x = np.min(self.ip_uu[0])
            max_x = np.max(self.ip_uu[0])
            xrange = np.linspace(min_x, max_x, n_1d)

            X1, X2 = np.meshgrid(xrange)
            testpoints = np.array([X1, X2])

            k_tp_pp_uu = self.array_kern_uu.matrix(testpoints, self.pp_uu[0])

            m = k_tp_pp_uu * (self.k_pp_pp_uu[0] / self.t_uu[0])
            if self.observation_model_uu == ObservationModels.Logit:
                m = expit(m)

            X = np.reshape(testpoints[:, 0], n_1d, n_1d)
            Y = np.reshape(testpoints[:, 1], n_1d, n_1d)
            Z = np.reshape(m, n_1d, n_1d)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z)
            plt.xlim(min_x, max_x)
            plt.ylim(min_x, max_x)

            if self.observation_model_uu == ObservationModels.Logit:
                ax.set_zlim(0, 1)

            plt.colorbar()

    def talk(
        self, performance=None
    ):  # Tell the world about various performance stats, but why is the model talking to us?
        if performance is None:
            performance = self.performance(False, np.array([]))

        print("")
        print(f"UU : ", end="")
        if self.observation_model_uu == ObservationModels.Logit:
            print(
                f"AUC = {performance['auc']:.3f} : Error = {performance['classifier_error']:.3f}"
            )
        if (
            self.observation_model_uu == ObservationModels.Gaussian
            or self.observation_model_uu == ObservationModels.Poisson
        ):
            raise Exception("Only ObservationModels Logit Implemented")

    def sum_predictions(self, p1, p2):
        p_sum = np.empty((len(p1.uu)))
        for i in range(p1.uu):
            p_sum[i] = p1.uu[i] + p2.uu[i]

    def divide_predictions(self, p, divisor):
        p_div = np.empty((len(p)))
        for i in range(len(p)):
            p_div[i] = p[i] / divisor
