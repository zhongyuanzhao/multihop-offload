# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix


def local_greedy_search(adj, wts):
    '''
    Return MWIS set and the total weights of MWIS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :return: mwis, total_wt
    '''
    wts = np.array(wts).flatten()
    verts = np.array(range(wts.size))
    mwis = set()
    remain = set(verts.flatten())
    vidx = list(remain)
    nb_is = set()
    while len(remain) > 0:
        for v in remain:
            # if v in nb_is:
            #     continue
            _, nb_set = np.nonzero(adj[v])
            nb_set = set(nb_set).intersection(remain)
            if len(nb_set) == 0:
                mwis.add(v)
                continue
            nb_list = list(nb_set)
            nb_list.sort()
            wts_nb = wts[nb_list]
            w_bar_v = wts_nb.max()
            if wts[v] > w_bar_v:
                mwis.add(v)
                nb_is = nb_is.union(set(nb_set))
            elif wts[v] == w_bar_v:
                i = list(wts_nb).index(wts[v])
                nbv = nb_list[i]
                if v < nbv:
                    mwis.add(v)
                    nb_is = nb_is.union(set(nb_set))
            else:
                pass
        remain = remain - mwis - nb_is
    total_ws = np.sum(wts[list(mwis)])
    return mwis, total_ws

def vis_edges(graph, pos, edge_labels, ax=None, font_size = 12):
    nx.draw_networkx_edge_labels(graph, pos = pos, edge_labels = edge_labels, ax =ax, font_size = font_size)

def vis_network(graph, src_nodes, dst_nodes, pos, weights=None, delays=None, with_labels=True, ax=None,
                colors=['g','r','b'], alpha=1.0
                ):
    color_list = ['b', 'm', 'g', 'r', 'k', 'w']
    g_size = graph.number_of_nodes()
    node_colors = ['y' for node in range(g_size)]
    node_sizes = list(np.ones((g_size,)) * 300) # [300 for node in range(g_size)]
    node_shapes = ['o' for node in range(g_size)]
    edge_colors = ['k' for edge in range(len(weights))]
    for i in range(len(weights)):
        if weights[i] > 0.99:
            edge_colors[i] = colors[0]

    if delays is not None:
        # node_sizes = 10 * delays + 5
        node_sizes = (delays/5)**2 + 20

    for i in range(len(src_nodes)):
        node_colors[src_nodes[i]] = colors[1]
        # node_colors[src_nodes[i]] = color_list[i]
        node_shapes[src_nodes[i]] = 'd'
        # if delays is None:
        if node_sizes[src_nodes[i]] < 200:
            node_sizes[src_nodes[i]] = 200

    for i in range(len(dst_nodes)):
        node_colors[dst_nodes[i]] = colors[2]
        # node_colors[dst_nodes[i]] = color_list[i]
        node_sizes[dst_nodes[i]] = 200
        node_shapes[dst_nodes[i]] = 's'

    nx.draw(
        graph,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=with_labels,
        pos=pos,
        width=weights,
        ax=ax,
        edge_color=edge_colors,
        alpha=alpha,
        )
    #return None


def all_pairs_shortest_paths(graph, weight=None):
    g_size = graph.number_of_nodes()
    sp_mtx = np.zeros((g_size, g_size))
    lengths = nx.all_pairs_dijkstra_path_length(graph, weight=weight)
    lengths = dict(lengths)
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            # sp_mtx[n1][n2] = nx.shortest_path_length(graph, n1, n2)
            sp_mtx[n1][n2] = lengths[n1][n2]
    return sp_mtx


def softmax(x_in):
    x = x_in[:]
    ans = np.exp(x) / sum(np.exp(x))
    return ans

