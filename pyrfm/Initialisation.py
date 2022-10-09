import numpy as np
from utils.create_gp_input_points import create_gp_input_points


def initialise_pp_uu(train_x_i, train_x_j, u, n_pp_uu, rng):
    if len(train_x_i) < n_pp_uu:
        raise Exception("Less data then pseudo inputs is not allowed.")
    perm = np.arange(len(train_x_i))
    rng.shuffle(perm)
    rand_subset = perm[:n_pp_uu]
    pp_uu = create_gp_input_points(
        train_x_i[rand_subset],
        train_x_j[rand_subset],
        u,
    )
    return pp_uu
