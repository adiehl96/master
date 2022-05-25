import numpy as np


def slice_sample_max(
    N, burn, logdist, xx, widths, max_attempts, rng, step_out=False, varargin=None
):
    dimension = len(xx)
    samples = np.zeros((dimension, N))
    log_px = logdist(xx)

    # print("log_px")
    # print(log_px)
    # raise Exception("lel")

    for ii in range(N + burn):
        log_uprime = np.log(rng.uniform()) + log_px

        perm = np.arange(dimension)
        rng.shuffle(perm)
        # print(perm)
        # raise Exception("lel")
        for dd in perm:
            x_l = xx
            x_r = xx
            xprime = xx

            rr = rng.uniform()
            x_l[dd] = xx[dd] - rr * widths[dd]
            x_r[dd] = xx[dd] + (1 - rr) * widths[dd]

            if step_out:
                raise Exception("step_out not implemented")

            zz = 0
            num_attempts = 0
            while True:
                zz = zz + 1
                xprime[dd] = rng.uniform() * (x_r[dd] - x_l[dd]) + x_l[dd]
                log_px = logdist(xprime)
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
        if ii >= burn:
            samples[:, ii - burn] = xx
    return samples
