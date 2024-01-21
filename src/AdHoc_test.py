# python3
# This software was created by William Marsh Rice University under Army Research Office (ARO) 
# Award Number W911NF-24-2-0008 and W911NF-19-2-0269. ARO, as the Federal awarding agency, reserves 
# a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this 
# software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import pickle
import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scipy.io as sio
import sparse
import copy
np.set_printoptions(threshold=np.inf)
from offloading_v3 import AdhocCloud
# Import utility functions
from util import *
from gnn_offloading_agent import ACOAgent, FLAGS

agent = ACOAgent(FLAGS, 1000)

arrival_scale = FLAGS.arrival_scale
std = 0.01
T = FLAGS.T
batch_size = FLAGS.batch
prob = FLAGS.prob
datapath = FLAGS.datapath
val_mat_names = sorted(os.listdir(datapath))
output_dir = FLAGS.out
output_csv = os.path.join(
    output_dir,
    "Adhoc_test_data_{}_load_{:.2f}_T_{}.csv".format(datapath.split("/")[-1], arrival_scale, T)
)
df_res = pd.DataFrame(
    columns=["filename", "seed", "num_nodes", "m", "num_mobile", "num_servers", "num_relays", "num_jobs", "n_instance",
             "Algo", "runtime", "tau", "congest_jobs", "gnn_bl_ratio", "gap_2_bl"]
)

# Create fig folder if not exist
modeldir = os.path.join("..", "model")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)

log_dir = os.path.join("..", "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# load models
actor_model = os.path.join(modeldir, 'model_ChebConv_{}_a{}_c{}_ACO_agent'.format(FLAGS.training_set, 5, 5))
# critic_model = os.path.join(modeldir, 'model_ChebConv_{}_a{}_c{}_critic'.format(FLAGS.training_set, 5, 5))

try:
    agent.load(actor_model)
except:
    print("unable to load {}".format(actor_model))

# try:
#     agent.load_critic(critic_model)
# except:
#     print("unable to load {}".format(critic_model))


# Define tensorboard
# agent.log_init()
gidx = 0
num_networks = len(val_mat_names)
num_instances = 10

results = np.full((num_instances, 7), np.nan)
bi = 0
for fid in range(num_networks):
    # Load configuration
    filepath = os.path.join(datapath, val_mat_names[fid])
    mat_contents = sio.loadmat(filepath)
    net_cfg = mat_contents['network'][0,0]
    link_rates = mat_contents["link_rate"].flatten()
    nodes_info = mat_contents["nodes_info"]
    adj_c = mat_contents['adj']
    pos_c = mat_contents["pos_c"]
    seed = int(net_cfg['seed'].flatten()[0])
    NUM_NODES = int(net_cfg['num_nodes'].flatten()[0])
    m = net_cfg['m'].flatten()[0]

    # Random case generation
    # Configuration
    ec_env = AdhocCloud(NUM_NODES, T, seed,
                        cf_radius=0.0,
                        gtype=filepath,
                        trace=True)
    ec_env.links_init(link_rates)

    for nidx in range(NUM_NODES):
        if nodes_info[nidx, 0] == 2:
            ec_env.add_relay(nidx)
        elif nodes_info[nidx, 0] == 1:
            ec_env.add_server(nidx, float(nodes_info[nidx, 1]))
        elif nodes_info[nidx, 0] == 0:
            ec_env.proc_bws[nidx] = nodes_info[nidx, 1]

    start_time_case = time.time()
    for ni in range(num_instances):
        ec_env.clear_all_jobs()
        mobile_nodes, = np.nonzero(nodes_info[:, 0] == 0)
        num_mobile = mobile_nodes.size
        mobile_nodes = np.random.permutation(mobile_nodes)
        # add jobs
        num_jobs = np.random.randint(int(0.3 * num_mobile), num_mobile)
        arrival_rates = np.random.uniform(0.1, 0.5, (num_jobs,))
        for idx in range(num_jobs):
            ec_env.add_job(mobile_nodes[idx], rate=arrival_scale * arrival_rates[idx])

        delay_dict = {}
        block_dict = {}
        for method in ["baseline", "local", "GNN"]:
            start_time = time.time()
            if method == "baseline":
                # Baseline method
                dmtx_bl, dlist_bl, dproc_bl = ec_env.dmtx_baseline()
                dproc_bl[dproc_bl <= 0] = float(ec_env.T)
                for link, delay in zip(ec_env.link_list, dlist_bl):
                    src, dst = link
                    ec_env.graph_c[src][dst]["delay"] = delay if delay > 0 else float(ec_env.T)

                sp_baseline = all_pairs_shortest_paths(ec_env.graph_c, weight="delay")
                sp_hop = all_pairs_shortest_paths(ec_env.graph_c, weight=None)
                np.fill_diagonal(sp_baseline, dproc_bl)
                decisions, delay_est = ec_env.offloading(sp_baseline, sp_hop)
                delay_links, delay_nodes, delay_unit_bl = ec_env.run()
                # ec_env.plot_routes(delay_links, delay_nodes, 0)
                delay_emp = np.nansum(delay_links, axis=0) + np.nansum(delay_nodes, axis=0)
                flows_bl = copy.deepcopy(ec_env.flows)
                # print("Delay estimation: {}\nDelay empircal: {}".format(delay_est, delay_emp))
            elif method == "local":
                dmtx_bl, dlist_bl, dproc_bl = ec_env.dmtx_baseline()
                decisions, delay_est = ec_env.local_compute(dproc_bl)
                delay_links, delay_nodes, delay_unit_bl = ec_env.run()
                delay_emp = np.nansum(delay_links, axis=0) + np.nansum(delay_nodes, axis=0)
                flows_local = copy.deepcopy(ec_env.flows)
            elif method == "GNN":
                obj = ec_env.graph_expand()
                (delay_mtx_np, delay_links_gnn, delay_nodes_gnn, delay_unit_gnn, flows_gnn,
                 loss_fn, loss_mse) = agent.forward_backward(obj, ec_env)
                delay_emp = np.nansum(delay_links_gnn, axis=0) + np.nansum(delay_nodes_gnn, axis=0)

            runtime = time.time() - start_time
            delay_dict[method] = delay_emp

            congest_jobs = np.count_nonzero(delay_emp > float(ec_env.T))
            result = {
                "filename": val_mat_names[fid],
                "seed": seed,
                "n_instance": ni,
                "num_nodes": NUM_NODES,
                "m": m,
                "num_servers": len(ec_env.servers),
                "num_relays": len(ec_env.relays),
                "num_mobile": NUM_NODES - len(ec_env.servers) - len(ec_env.relays),
                "num_jobs": num_jobs,
                "Algo": method,
                "runtime": runtime,
                "gap_2_bl": np.nanmean(delay_dict[method] - delay_dict["baseline"]),
                "gnn_bl_ratio": np.nanmean(delay_dict[method] / delay_dict["baseline"]),
                "tau": np.nanmean(delay_emp),
                "congest_jobs": congest_jobs,
            }
            congest_ratio = congest_jobs / num_jobs
            df_res = df_res.append(result, ignore_index=True)

        # start_time = time.time()
        # # Baseline method
        # dmtx_bl, dlist_bl, dproc_bl = ec_env.dmtx_baseline()
        # dproc_bl[dproc_bl <= 0] = float(ec_env.T)
        # for link, delay in zip(ec_env.link_list, dlist_bl):
        #     src, dst = link
        #     ec_env.graph_c[src][dst]["delay"] = delay if delay > 0 else float(ec_env.T)
        #
        # sp_baseline = all_pairs_shortest_paths(ec_env.graph_c, weight="delay")
        # np.fill_diagonal(sp_baseline, dproc_bl)
        # decisions, delay_est = ec_env.offloading(sp_baseline)
        # delay_links, delay_nodes, delay_unit_bl = ec_env.run()
        # # ec_env.plot_routes(delay_links, delay_nodes, 0)
        # delay_emp = np.nansum(delay_links, axis=0) + np.nansum(delay_nodes, axis=0)
        # flows_bl = copy.deepcopy(ec_env.flows)
        # # print("Delay estimation: {}\nDelay empircal: {}".format(delay_est, delay_emp))
        # runtime0 = time.time() - start_time
        #
        # # GNN method
        # start_time = time.time()
        # obj = ec_env.graph_expand()
        # delay_links_gnn, delay_nodes_gnn, delay_unit_gnn = agent.forward_env(obj, ec_env)
        # delay_emp_gnn = np.nansum(delay_links_gnn, axis=0) + np.nansum(delay_nodes_gnn, axis=0)
        # runtime1 = time.time() - start_time
        #
        # gap_mean = np.nanmean(delay_emp_gnn - delay_emp)
        # gnn_bl_ratio = np.nanmean(delay_emp_gnn / delay_emp)
        # congest_jobs_bl = np.count_nonzero(delay_emp > float(ec_env.T))
        # congest_jobs_gnn = np.count_nonzero(delay_emp_gnn > float(ec_env.T))
        # results[ni, :] = [runtime0, runtime1, gap_mean, gnn_bl_ratio, num_jobs, congest_jobs_bl/num_jobs, congest_jobs_gnn/num_jobs]
        #
        # result = {
        #     "filename": val_mat_names[fid],
        #     "seed": seed,
        #     "n_instance": ni,
        #     "num_nodes": NUM_NODES,
        #     "m": m,
        #     "num_servers": len(ec_env.servers),
        #     "num_relays": len(ec_env.relays),
        #     "num_mobile": NUM_NODES - len(ec_env.servers) - len(ec_env.relays),
        #     "num_jobs": num_jobs,
        #     "runtime_b": runtime0,
        #     "runtime_g": runtime1,
        #     "gap": gap_mean,
        #     "gnn_bl_ratio": gnn_bl_ratio,
        #     "tau_bl": np.nanmean(delay_emp),
        #     "tau_gnn": np.nanmean(delay_emp_gnn),
        #     "congest_jobs_bl": congest_jobs_bl,
        #     "congest_jobs_gnn": congest_jobs_gnn
        #     }
        # df_res = df_res.append(result, ignore_index=True)

    runtime_case = time.time() - start_time_case
    stats = np.nanmean(results, axis=0)
    max_jobs = np.nanmax(results[:, 4])
    min_jobs = np.nanmin(results[:, 4])
    avg_jobs = stats[4]
    print(
        "Runtime {:.3f}s".format(runtime_case),
        " for network of {} nodes, {} servers, {} relays\n".format(NUM_NODES, len(ec_env.servers), len(ec_env.relays)),
        # "GNN-baseline gap: {:.2f} t, {:.2f}%\n".format(stats[2], 100*stats[3]),
        # "Congestion jobs: GNN: {:.2f}%, baseline {:.2f}%\n".format(100*stats[5], 100*stats[6]),
    )

    gidx += 1

    df_res.to_csv(output_csv, index=False)

