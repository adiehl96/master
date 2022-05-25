import numpy as np


def bernoulli_llh(prob, data):
    # print("prob.shape")
    # print(prob.shape)
    # print("data.shape")
    # print(data.shape)
    eps = np.finfo(np.float64).eps
    prob[prob < eps] = eps
    prob[prob > (1 - eps)] = 1 - eps
    return np.sum(data @ np.log(prob) + (1 - data) @ np.log(1 - prob))
