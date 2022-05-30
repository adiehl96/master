import numpy as np
from Enumerations.InitialisationMethods import InitialisationMethods


class HCMCMC_SS_RFM:
    def __init__(self):
        self.rfm = None

        # Slice Sampling Parameters
        self.u_width = 4
        self.u_step_out = True
        self.u_max_attempts = 6

        self.pp_width = 4
        self.pp_step_out = False
        self.pp_max_attempts = 6

        self.uu_kern_par_widths = None
        self.kern_par_step_out = False
        self.kern_par_max_attempts = 6

        self.t_iterations = 50
        self.pp_iterations = 10

        self.surf_sample = True

        # Sampling Parameters

        self.lv_modulus = 1
        self.pp_modulus = 1
        self.kern_par_modulus = 1
        self.data_par_modulus = 1
        self.t_modulus = 1
        self.plot_modulus = 1
        self.talk_modulus = 1

        self.burn = 200
        self.iterations = 1000

        self.init_method = InitialisationMethods.NONE

        # Sampling Traces

        self.rfm_state = np.array([])
        self.performance = np.array([])
        self.predictions = np.array([])
        self.predictions_average = np.array([])

        # Results

        # self.MAP # Is not needed actually

        # Experiment Practicalities

        self.batch = 0
        self.batches = 0
        self.predictions_sum = np.array([])

    def sample(self, newrun=True):
        if newrun:
            if self.init_method == InitialisationMethods.NONE:
                self.rfm.initialise_rand()
            if self.init_method == InitialisationMethods.PCA:
                self.rfm.initialise_pca()
        else:
            self.rfm.init_cache()

        if (
            newrun
            and self.init_method != InitialisationMethods.NONE
            and self.init_method != InitialisationMethods.PCA
        ):
            for repeat in range(5):

                for i in range(50):
                    self.rfm.new_permutation()
                    self.rfm.permute()

                    self.samplestep(i)
                    self.rfm.inverse_permute()

                    self.rfm.state()
                    self.rfm.prediction()
                    self.rfm.performance(False, [])

                    if (
                        i % self.plot_modulus == 0
                        and self.plot_modulus < self.iterations
                    ):
                        self.rfm.plot()

                    if i % self.talk_modulus == 0:
                        print(f"Initialising {i} / {50}")
                        self.rfm.talk()
                        print("")

                if self.init_method == InitialisationMethods.ResamplePseudo:
                    self.rfm.init_pp_t()
                if self.init_method == InitialisationMethods.MAPU:
                    self.rfm.init_u_map()
                if self.init_method == InitialisationMethods.Both:
                    if repeat % 2 == 1:
                        self.rfm.init_u_map()
                    else:
                        self.rfm.init_pp_t()

                self.rfm.init_cache()

        # Initialise MCMC traces if starting a new run

        self.predictions = []
        self.performance = []

        end_index = self.burn + self.iterations

        for i in range(end_index):
            self.rfm.new_permutation()
            self.rfm.permute()

            self.sample_step(i)

            self.rfm.inverse_permute()

            # self.rfm_state[i] = self.rfm.state()
            self.predictions.append(self.rfm.prediction())
            self.performance.append(self.rfm.performance(False, np.array([])))

            if i % self.plot_modulus == 0 and self.plot_modulus < self.iterations:
                self.rfm.plot()

            if i % self.talk_modulus == 0:
                print(f"Iter {i-self.burn} / {self.iterations}")
                self.rfm.talk()
                print("")

        self.predictions_average = (
            np.array(self.predictions).sum(axis=0) / self.iterations
        )

    def sample_step(self, i):
        if i % self.t_modulus == 0:
            self.rfm.ess_t(int(np.round(self.t_iterations / 2)))
        if i % self.lv_modulus == 0:
            self.rfm.ss_u(self.u_width, self.u_step_out, self.u_max_attempts)
        if i % self.pp_modulus == 0:
            self.rfm.ss_pp(
                self.pp_iterations,
                self.pp_width,
                self.pp_step_out,
                self.pp_max_attempts,
                self.surf_sample,
            )
        if i % self.kern_par_modulus == 0:
            self.rfm.ss_array_kern_params(
                self.uu_kern_par_widths,
                self.kern_par_step_out,
                self.kern_par_max_attempts,
            )
        # if i % self.data_par_modulus == 0: # todo I deleted this, might have been prematurely?
        #     self.rfm.gs_dataparams
        if i % self.t_modulus == 0:
            self.rfm.ess_t(int(np.round(self.t_iterations / 2)))
