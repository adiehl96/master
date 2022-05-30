import csv
import numpy as np


def load_partitioned_data(filename, folds, fold, rng):
    array, network_size = load_from_file(filename)
    train_x_i, train_x_j, train_x_v = flatten_adjacency_matrix(array)
    train_x_i, train_x_j, train_x_v, test_x_i, test_x_j, test_x_v = partition(
        train_x_i, train_x_j, train_x_v, folds, fold, rng
    )
    train_data = {}
    train_data["i"] = train_x_i
    train_data["j"] = train_x_j
    train_data["v"] = train_x_v
    test_data = {}
    test_data["i"] = test_x_i
    test_data["j"] = test_x_j
    test_data["v"] = test_x_v

    return train_data, test_data, network_size


def load_from_file(filename):
    with open(f"./datasets/{filename}", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        read = np.array([item for row in reader for item in row]).astype(np.int8)
        read = read.reshape((int(np.sqrt(len(read))), int(np.sqrt(len(read)))))
    array = read
    m, n = read.shape
    if m == n:
        network_size = m
    else:
        raise Exception("Adjacency matrix is not symmetric.")
    return array, network_size


def flatten_adjacency_matrix(array):
    O = np.ones_like(array)
    O = np.triu(O, 1)
    train_x_i, train_x_j = np.asarray(O > 0).nonzero()
    train_x_v = array[O == 1]
    return train_x_i, train_x_j, train_x_v


def partition(train_x_i, train_x_j, train_x_v, folds, fold, rng):
    perm = np.arange(len(train_x_v))
    rng.shuffle(perm)
    boundaries = np.floor(np.linspace(0, len(train_x_v), folds + 1)).astype(np.int32)
    predict = perm[: boundaries[fold] - 1]
    observed = perm[boundaries[fold] - 1 :]
    return (
        train_x_i[observed],
        train_x_j[observed],
        train_x_v[observed],
        train_x_i[predict],
        train_x_j[predict],
        train_x_v[predict],
    )
