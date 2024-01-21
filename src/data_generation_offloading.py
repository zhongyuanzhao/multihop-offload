# python3
# This software was created by William Marsh Rice University under Army Research Office (ARO) 
# Award Number W911NF-24-2-0008 and W911NF-19-2-0269. ARO, as the Federal awarding agency, reserves 
# a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this 
# software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
import os
import argparse
import sys
import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import networkx as nx
from offloading import *

# input arguments calling from command line
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="../ba_graph_100", type=str, help="test data directory.")
parser.add_argument("--gtype", default="ba", type=str, help="graph type.")
parser.add_argument("--size", default=100, type=int, help="size of dataset")
parser.add_argument("--seed", default=500, type=int, help="initial seed")
args = parser.parse_args()

data_path = args.datapath
gtype = args.gtype.lower()
size = args.size
seed0 = args.seed
# Create fig folder if not exist
if not os.path.isdir(data_path):
    os.mkdir(data_path)


def poisson_graph(size, nb=4, radius=1.0, seed=None):
    """
    Create a Poisson point process 2D graph
    """
    N = int(size)
    density = float(nb)/np.pi
    area = float(N) / density
    side = np.sqrt(area)
    if seed is not None:
        np.random.seed(int(seed))
    xys = np.random.uniform(0, side, (N, 2))
    d_mtx = distance_matrix(xys, xys)
    adj_mtx = np.zeros([N, N], dtype=int)
    adj_mtx[d_mtx <= radius] = 1
    np.fill_diagonal(adj_mtx, 0)
    graph = nx.from_numpy_matrix(adj_mtx)
    return graph, xys


m = 2
graph_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
# if gtype == 'poisson':
#     graph_sizes = [60, 70, 80, 90, 100, 110]

for id in range(size):
    seed = id + seed0
    for num_nodes in graph_sizes:
        if gtype == 'poisson':
            not_connected = True
            m = 3
            while not_connected:
                m += 1
                graph, pos_c = poisson_graph(num_nodes, nb=m, seed=seed)
                if nx.is_connected(graph):
                    not_connected = False
        else:
            ec_env = AdhocCloud(num_nodes, 100, seed=seed, m=m, pos='new', gtype=gtype)
            graph = ec_env.graph_c
            pos_c = np.zeros((num_nodes, 2))
            for i in range(num_nodes):
                pos_c[i, :] = ec_env.pos_c[i]

        adj = nx.adjacency_matrix(graph)
        num_links = graph.number_of_edges()
        jobs_perc = np.random.randint(15, 40)
        server_perc = np.random.randint(10, 25)
        num_jobs = round(jobs_perc/100 * num_nodes)
        num_servers = round(server_perc/100 * num_nodes)
        nodes = graph.nodes()
        # arrival_rates = np.random.uniform(0.2, 1.0, (num_jobs,))
        # link_rates = np.random.randint(12, (num_flows,))
        link_rates = np.random.uniform(30, 70, size=(num_links,))

        # Cut the graph to two regions
        relay_set = nx.minimum_node_cut(graph)
        cut_value, partition = nx.stoer_wagner(graph)
        nodes_info = np.zeros((num_nodes, 2), dtype=int)
        for idx in list(relay_set):
            nodes_info[idx, 0] = 2 # role
            nodes_info[idx, 1] = 0 # proc_bw

        partition0 = np.random.permutation(list(set(partition[0]) - relay_set)).tolist()
        partition1 = np.random.permutation(list(set(partition[1]) - relay_set)).tolist()
        partition = (partition0, partition1)
        if len(partition[0]) >= len(partition[1]):
            server_set_idx = 1
        else:
            server_set_idx = 0

        for sidx in range(2):
            if sidx == server_set_idx:
                if num_servers >= len(partition[sidx]):
                    s_proc_bws = (np.random.pareto(2.0, len(partition[sidx])) + 1) * 100
                    s_proc_bws = np.flip(np.sort(s_proc_bws))
                    for idx in range(len(partition[sidx])):
                        nidx = partition[sidx][idx]
                        nodes_info[nidx, 0] = 1 # role as server
                        nodes_info[nidx, 1] = s_proc_bws[idx]
                else:
                    s_proc_bws = (np.random.pareto(2.0, num_servers) + 1) * 100
                    s_proc_bws = np.flip(np.sort(s_proc_bws))
                    for idx in range(num_servers):
                        nidx = partition[sidx][idx]
                        nodes_info[nidx, 0] = 1
                        nodes_info[nidx, 1] = s_proc_bws[idx]
            else:
                num_near_servers = 0
                if num_servers >= len(partition[server_set_idx]):
                    num_near_servers = num_servers - len(partition[server_set_idx])
                    s_proc_bws = (np.random.pareto(2.0, num_near_servers) + 1) * 100
                    for idx in range(num_near_servers):
                        nidx = partition[sidx][idx]
                        nodes_info[nidx, 0] = 1
                        nodes_info[nidx, 1] = s_proc_bws[idx]
                num_mobile_nodes = len(partition[sidx]) - num_near_servers
                m_proc_bws = (np.random.pareto(2.0, num_mobile_nodes) + 1) * 8
                for idx in range(num_near_servers, len(partition[sidx])):
                    nidx = partition[sidx][idx]
                    nodes_info[nidx, 0] = 0
                    nodes_info[nidx, 1] = m_proc_bws[idx-num_near_servers]

        # ec_env.flows_init()
        filename = "aco_case_seed{}_m{}_n{}_s{}.mat".format(seed, m, num_nodes, num_servers)
        filepath = os.path.join(data_path, filename)
        sio.savemat(filepath,
                    {"network": {"num_nodes": num_nodes, "seed": seed, "m": m, "gtype": gtype},
                     "adj": adj.astype(float),
                     "link_rate": link_rates,
                     "nodes_info": nodes_info,
                     "pos_c": pos_c
                    })






