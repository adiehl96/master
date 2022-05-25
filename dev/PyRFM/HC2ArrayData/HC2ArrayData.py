import csv
import numpy as np
from copy import deepcopy


class HC2ArrayData:
    def __init__(self, filename):
        self.filename = filename
        self.load_from_file()
        self.flatten_adjacency_matrix()
        self.test_x_i = np.array([])
        self.test_x_j = np.array([])
        self.test_x_v = np.array([])

    def load_from_file(self):
        with open(f"./datasets/{self.filename}", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            read = np.array([item for row in reader for item in row]).astype(np.int8)
            read = read.reshape((int(np.sqrt(len(read))), int(np.sqrt(len(read)))))
        self.file = read
        self.m = read.shape[0]
        self.n = read.shape[1]

    def flatten_adjacency_matrix(self):
        O = np.ones_like(self.file)
        O = np.triu(O, 1)
        self.train_x_i = np.array(np.where(O > 0))[0, :]
        self.train_x_j = np.array(np.where(O > 0))[1, :]
        self.train_x_v = self.file[O == 1]

    def partition(self, folds, fold, rng):
        data = deepcopy(self)
        perm = np.arange(len(self.train_x_v))
        rng.shuffle(perm)
        boundaries = np.floor(np.linspace(0, len(self.train_x_v), folds + 1)).astype(
            np.int32
        )
        predict = perm[: boundaries[fold] - 1]
        observed = perm[boundaries[fold] - 1 :]
        data.train_x_i = self.train_x_i[observed]
        data.train_x_j = self.train_x_j[observed]
        data.train_x_v = self.train_x_v[observed]
        data.test_x_i = self.train_x_i[predict]
        data.test_x_j = self.train_x_j[predict]
        data.test_x_v = self.train_x_v[predict]
        return data

    def duplicate(self):
        return deepcopy(self)

    def restore_adjacency_matrix(self):
        restored = np.zeros_like(self.file)
        for i, j, v in zip(self.train_x_i, self.train_x_j, self.train_x_v):
            restored[i, j] = v
            restored[j, i] = v
        for i, j, v in zip(self.test_x_i, self.test_x_j, self.test_x_v):
            restored[i, j] = v
            restored[j, i] = v
        return restored

    def restored_equal_to_dataset(self):
        restored = self.restore_adjacency_matrix()
        return np.equal(self.file, restored).all()
