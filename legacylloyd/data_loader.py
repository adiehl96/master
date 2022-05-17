import csv
import networkx as nx
import numpy as np


def csv_to_graph(name):
    with open(f"{name}.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        read = np.array([item for row in reader for item in row]).astype(np.int8)
        read = read.reshape((int(np.sqrt(len(read))), int(np.sqrt(len(read)))))

    return nx.from_numpy_matrix(read)
