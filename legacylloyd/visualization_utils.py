import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def print_graph(nodes=None, edges=None, G=None, dpi=60):
    if not isinstance(G, nx.Graph):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(
        G, prog="sfdp"
    )  # neato, dot, twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor, ccomps, sccmap, tred, sfdp, unflatten
    options = {
        "node_color": "white",
        "edgecolors": "blue",
        "font_size": 11,
        "node_size": 100,
    }
    _, axes = plt.subplots(figsize=(10, 10), dpi=dpi)
    nx.draw(G, pos, axes, **options)
    plt.show()


def nodes_sorted_by_degree(graph):
    return [
        x[0]
        for x in sorted(
            {n: graph.degree(n) for n in graph.nodes}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]


def sort_matrix_by_indices(labels, matrix):
    order = [
        x[0]
        for x in sorted(
            {n: l for n, l in enumerate(labels)}.items(),
            key=lambda x: x[1],
            reverse=False,
        )
    ]
    order = np.array(order)
    ordermatrix = np.tile(order, (len(labels), 1))
    matrix_sorted_columns = np.take_along_axis(matrix, ordermatrix, axis=1)
    return sorted(labels), matrix_sorted_columns[order]
