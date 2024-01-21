# python3
# This software was created by William Marsh Rice University under Army Research Office (ARO) 
# Award Number W911NF-24-2-0008 and W911NF-19-2-0269. ARO, as the Federal awarding agency, reserves 
# a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this 
# software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
from __future__ import division
from __future__ import print_function
import sys
import os
import time
import datetime
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import random
from multiprocessing import Queue
from collections import deque
from copy import deepcopy
import networkx as nx
# Tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, schedules
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
# Spektral
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ECCConv, ChebConv
from spektral.utils import sp_matrix_to_sp_tensor, reorder
from spektral.transforms import LayerPreprocess
# Graph utility
from util import all_pairs_shortest_paths
from graph_util import *
import warnings
warnings.filterwarnings('ignore')

# input flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('datapath', '../data_100', 'input data path.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('out', '../out', 'output data path.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_integer('T', 1000, 'Number of time slots with traffic inputs.')
flags.DEFINE_boolean('prob', False, 'If probabilistic decision.')
flags.DEFINE_string('training_set', 'BAm2', 'Name of training dataset')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('learning_decay', 1.0, 'Initial learning rate.')
flags.DEFINE_float('arrival_scale', 0.1, 'Scale of arrival rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('num_layer', 5, 'number of layers.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('epsilon', 1.0, 'initial exploration rate')
flags.DEFINE_float('epsilon_min', 0.001, 'minimal exploration rate')
flags.DEFINE_float('epsilon_decay', 0.985, 'exploration rate decay per replay')
flags.DEFINE_float('gamma', 1.0, 'gamma')
flags.DEFINE_integer('batch', 100, 'batch size.')


# Agent
class ACOAgent:
    def __init__(self, input_flags, memory_size=5000):
        # super(GDPGAgent, self).__init__(input_flags, memory_size)
        self.flags = input_flags
        self.learning_rate = self.flags.learning_rate
        self.n_node_features = 4
        self.output_size = 1
        self.max_degree = 1
        self.num_supports = 1 + self.max_degree
        self.l2_reg = self.flags.weight_decay
        self.epsilon = self.flags.epsilon
        self.model = self._build_model()
        self.memory = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.mse = MeanSquaredError()
        # self.log_init()

    def _build_model(self):
        # Neural Net for Actor Model
        x_in = Input(shape=(self.n_node_features,), dtype=tf.float64, name="x_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        gc_l = x_in
        for l in range(self.flags.num_layer):
            if l < self.flags.num_layer - 1:
                act = "leaky_relu"
                output_dim = 32
            else:
                act = "relu"
                output_dim = self.output_size
            do_l = Dropout(self.flags.dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                channels=output_dim,
                # kernel_network=None,
                # kernel_network=[32, 48, 32],
                # root=True,
                activation=act,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=l2(self.l2_reg),
                bias_regularizer=l2(self.l2_reg),
                # activity_regularizer=None,
                kernel_constraint=max_norm(1.0),
                bias_constraint=max_norm(1.0),
                dtype='float64'
            )([do_l, a_in])

        # Build model
        model = Model(inputs=[x_in, a_in], outputs=gc_l)
        if self.flags.learning_decay == 1.0:
            self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        else:
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=100,
                decay_rate=self.flags.learning_decay)
            self.optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.summary()
        return model

    def load(self, name):
        ckpt = tf.train.latest_checkpoint(name)
        if ckpt:
            self.model.load_weights(ckpt)
            print('Actor loaded ' + ckpt)

    def save(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def makestate(self, adj, node_features):
        # reduced_nn = node_features.shape[0]
        # support = simple_polynomials(adj, self.max_degree)
        state = {"node_features": node_features,
                 "support": adj}
        return state

    def memorize(self, grad, loss, reward):
        self.memory.append((grad.copy(), loss, reward))

    def predict(self, state):
        x_in = tf.convert_to_tensor(state["node_features"], dtype=tf.float64)
        # coord, values, shape = state["support"]
        # a_in = tf.sparse.SparseTensor(coord, values, shape)
        a_in = sp_matrix_to_sp_tensor(state["support"])
        act_values = self.model([x_in, a_in])
        return act_values

    def act(self, state):
        high_dimensional_action = self.predict(state)
        return high_dimensional_action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return float('NaN')
        self.reward_mem.clear()
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for grad, loss, _ in minibatch:
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
            losses.append(loss)

        # self.memory.clear()
        if self.epsilon > self.flags.epsilon_min:
            self.epsilon *= self.flags.epsilon_decay
        return np.nanmean(losses)

    def forward_gcn(self, obj, env):
        """
        Output the shortest path matrix, with diagonal as self-loop
        :param obj:
        :param env:
        :return:
        """
        adj = nx.adjacency_matrix(obj.gi_ext)
        nn = obj.num_edges_ext
        node_features = np.zeros((nn, 3))
        node_features[:, 0] = obj.edge_self_loop
        node_features[:, 1] = obj.edge_rate_ext
        node_features[:, 2] = obj.jobs_arrivals

        state = self.makestate(adj, node_features)
        lambda_array = self.act(state)

        # Compute link mu based on lambda and rate of all links
        # The probability of a link with empty queue is (1-lambda/mu)
        link_delay = tf.gather(lambda_array, obj.maps_ol_el)
        node_delay = tf.gather(lambda_array, obj.maps_on_el)

        # Fill the link distance matrix
        link_delay_np = link_delay.numpy()
        node_delay_np = node_delay.numpy()
        delay_mtx_np = np.zeros((env.num_nodes, env.num_nodes))
        delay_mtx_ts = tf.zeros((env.num_nodes, env.num_nodes), dtype=link_delay.dtype)
        indices0 = np.zeros((env.num_links, 1, 2), dtype=np.int64)
        indices1 = np.zeros((env.num_links, 1, 2), dtype=np.int64)
        for (e0, e1) in env.graph_c.edges:
            indices0[env.link_matrix[e0, e1], 0, :] = [e0, e1]
            indices1[env.link_matrix[e1, e0], 0, :] = [e1, e0]
            delay_mtx_np[e0, e1] = delay_mtx_np[e1, e0] = link_delay_np[env.link_matrix[e0, e1]]
        delay_mtx_ts = tf.tensor_scatter_nd_update(delay_mtx_ts, indices0, link_delay)
        delay_mtx_ts = tf.tensor_scatter_nd_update(delay_mtx_ts, indices1, link_delay)
        np.fill_diagonal(delay_mtx_np, node_delay_np)
        delay_mtx_ts = tf.linalg.set_diag(delay_mtx_ts, node_delay[:, 0])

        return state, delay_mtx_ts, delay_mtx_np

    def forward(self, obj, env):
        """
        Output the shortest path matrix, with diagonal as self-loop
        :param obj:
        :param env:
        :return:
        """
        adj = nx.adjacency_matrix(obj.gi_ext)
        nn = obj.num_edges_ext
        node_features = np.zeros((nn, 4))
        node_features[:, 0] = obj.edge_self_loop
        node_features[:, 1] = obj.edge_rate_ext
        node_features[:, 2] = obj.jobs_arrivals
        node_features[:, 3] = obj.edge_as_server

        state = self.makestate(adj, node_features)
        lambda_array = self.act(state)

        # Compute link mu based on lambda and rate of all links
        # The probability of a link with empty queue is (1-lambda/mu)
        link_lambda = tf.gather(lambda_array, obj.maps_ol_el)
        node_lambda = tf.gather(lambda_array, obj.maps_on_el)
        node_mu = (env.proc_bws.copy()).reshape((env.num_nodes, 1))
        comp_nodes, _ = np.where(node_mu > 0)
        node_mu = node_mu[comp_nodes, :]
        link_mu = (env.link_rates / (env.cf_degs + 1)).reshape((env.num_links, 1))
        link_rates = env.link_rates.reshape((env.num_links, 1))
        adj_sp_ts = sp_matrix_to_sp_tensor(env.adj_i)
        adj_sp_ts = tf.cast(adj_sp_ts, dtype=node_lambda.dtype)
        for it in range(10):
            link_busy = tf.clip_by_value(link_lambda / link_mu, 0, 1.0)
            neighbor_busy = tf.sparse.sparse_dense_matmul(adj_sp_ts, link_busy)
            link_ratio = 1.0 / (1.0 + neighbor_busy)
            link_mu = link_rates * link_ratio
        link_delay = 1 / (link_mu - link_lambda)
        node_delay = 1 / (node_mu - node_lambda)
        link_congest = tf.where((link_lambda - link_mu) > 0)
        node_congest = tf.where((node_lambda - node_mu) > 0)
        link_delay_congest = tf.gather(float(env.T) * (link_lambda / (101 * link_mu)), link_congest[:, 0])
        node_delay_congest = tf.gather(float(env.T) * (node_lambda / (100 * node_mu)), node_congest[:, 0])
        link_delay_congest = tf.squeeze(link_delay_congest, 1)
        node_delay_congest = tf.squeeze(node_delay_congest, 1)
        link_delay = tf.tensor_scatter_nd_update(link_delay, link_congest, link_delay_congest)
        node_delay = tf.tensor_scatter_nd_update(node_delay, node_congest, node_delay_congest)

        # Fill the link distance matrix
        link_delay_np = link_delay.numpy()
        node_delay_np = node_delay.numpy()
        delay_mtx_np = np.full((env.num_nodes, env.num_nodes), fill_value=np.nan)
        delay_mtx_ts = tf.zeros((env.num_nodes, env.num_nodes), dtype=link_delay.dtype)
        indices0 = np.zeros((env.num_links, 1, 2), dtype=np.int64)
        indices1 = np.zeros((env.num_links, 1, 2), dtype=np.int64)
        for (e0, e1) in env.graph_c.edges:
            indices0[env.link_matrix[e0, e1], 0, :] = [e0, e1]
            indices1[env.link_matrix[e1, e0], 0, :] = [e1, e0]
            delay_mtx_np[e0, e1] = delay_mtx_np[e1, e0] = link_delay_np[env.link_matrix[e0, e1]]
        delay_mtx_ts = tf.tensor_scatter_nd_update(delay_mtx_ts, indices0, link_delay)
        delay_mtx_ts = tf.tensor_scatter_nd_update(delay_mtx_ts, indices1, link_delay)
        np.fill_diagonal(delay_mtx_np, node_delay_np)
        node_delay_full = tf.fill([env.num_nodes], float("inf"))
        node_delay_full = tf.cast(node_delay_full, dtype=node_delay.dtype)
        indices = comp_nodes.reshape((-1, 1))
        node_delay_full = tf.tensor_scatter_nd_update(node_delay_full, indices, node_delay[:, 0])
        delay_mtx_ts = tf.linalg.set_diag(delay_mtx_ts, node_delay_full)

        return state, delay_mtx_ts, delay_mtx_np

    def forward_env(self, obj, env):
        state, delay_mtx_ts, delay_mtx_np = self.forward(obj, env)
        # delay_mtx_np[delay_mtx_np <= 0] = float(env.T)
        for (src, dst) in env.graph_c.edges:
            env.graph_c[src][dst]["delay"] = delay_mtx_np[src, dst]

        delay_servers = np.diagonal(delay_mtx_np)
        # Create the shortest path matrix, run the environment
        sp_gnn = all_pairs_shortest_paths(env.graph_c, weight="delay")
        sp_hop = all_pairs_shortest_paths(env.graph_c, weight=None)
        np.fill_diagonal(sp_gnn, delay_servers)
        decisions, delay_est = env.offloading(sp_gnn, sp_hop)
        delay_links_gnn, delay_nodes_gnn, delay_unit_gnn = env.run()
        return delay_links_gnn, delay_nodes_gnn, delay_unit_gnn

    def forward_backward(self, obj, env, explore=0.0):
        # GNN
        with tf.GradientTape() as g:
            g.watch(self.model.trainable_weights)
            state, delay_mtx_ts, delay_mtx_np = self.forward(obj, env)
            # delay_mtx_np[delay_mtx_np <= 0] = float(env.T)
            for (src, dst) in env.graph_c.edges:
                env.graph_c[src][dst]["delay"] = delay_mtx_np[src, dst]

            delay_servers = np.diagonal(delay_mtx_np)
            # Create the shortest path matrix, run the environment
            sp_gnn = all_pairs_shortest_paths(env.graph_c, weight="delay")
            sp_hop = all_pairs_shortest_paths(env.graph_c, weight=None)
            np.fill_diagonal(sp_gnn, delay_servers)
            decisions, delay_est = env.offloading(sp_gnn, sp_hop, explore)
            delay_links_gnn, delay_nodes_gnn, delay_unit_gnn = env.run()

            routes_np = np.zeros((obj.num_edges_ext, env.num_jobs))
            jobs_load = np.zeros((env.num_jobs, 1))
            jobs_data = np.zeros((1, env.num_jobs))
            for i in range(env.num_jobs):
                src = env.jobs[i].source_node
                arr = env.jobs[i].arrival_rate * env.jobs[i].ul_data
                jobs_load[i, 0] += arr
                jobs_data[0, i] += env.jobs[i].ul_data + env.jobs[i].dl_data
                n0 = src
                if n0 != env.flows[i].dst:
                    for n1 in env.flows[i].route[1:]:
                        if (n0, n1) in obj.link_list_ext:
                            lidx = obj.link_list_ext.index((n0, n1))
                        elif (n1, n0) in obj.link_list_ext:
                            lidx = obj.link_list_ext.index((n1, n0))
                        else:
                            raise ValueError("Link not exist, check route")
                        routes_np[lidx, i] = 1
                        n0 = n1
                n1 = n0 + env.num_nodes
                lidx = obj.link_list_ext.index((n0, n1))
                routes_np[lidx, i] = 1
            # Critic with nested GradientTape
            with tf.GradientTape() as gg:
                routes_ts = tf.convert_to_tensor(routes_np)
                jobs_load_ts = tf.convert_to_tensor(jobs_load)
                jobs_data_ts = tf.convert_to_tensor(jobs_data)
                gg.watch(routes_ts)
                link_load_ts = tf.linalg.matmul(routes_ts, jobs_load_ts)
                link_lambda = tf.gather(link_load_ts, obj.maps_ol_el)
                node_lambda = tf.gather(link_load_ts, obj.maps_on_el)
                node_mu = (env.proc_bws.copy()).reshape((env.num_nodes, 1))
                comp_nodes, _ = np.where(node_mu > 0)
                node_mu = node_mu[comp_nodes, :]
                link_mu = (env.link_rates / (env.cf_degs + 1)).reshape((env.num_links, 1))
                link_rates = env.link_rates.reshape((env.num_links, 1))
                adj_sp_ts = sp_matrix_to_sp_tensor(env.adj_i)
                adj_sp_ts = tf.cast(adj_sp_ts, dtype=node_lambda.dtype)
                for it in range(10):
                    link_busy = tf.clip_by_value(link_lambda / link_mu, 0, 1.0)
                    neighbor_busy = tf.sparse.sparse_dense_matmul(adj_sp_ts, link_busy)
                    link_ratio = 1.0 / (1.0 + neighbor_busy)
                    link_mu = link_rates * link_ratio
                link_delay = 1 / (link_mu - link_lambda)
                node_delay = 1 / (node_mu - node_lambda)
                link_congest = tf.where((link_lambda - link_mu) > 0)
                node_congest = tf.where((node_lambda - node_mu) > 0)
                link_delay_congest = tf.gather(float(env.T) * (link_lambda/(101*link_mu)), link_congest[:, 0])
                node_delay_congest = tf.gather(float(env.T) * (node_lambda/(100*node_mu)), node_congest[:, 0])
                link_delay_congest = tf.squeeze(link_delay_congest, 1)
                node_delay_congest = tf.squeeze(node_delay_congest, 1)
                link_delay = tf.tensor_scatter_nd_update(link_delay, link_congest, link_delay_congest)
                node_delay = tf.tensor_scatter_nd_update(node_delay, node_congest, node_delay_congest)

                unit_delay_edge_ts = tf.zeros((obj.num_edges_ext,), dtype=link_delay.dtype)
                indices = obj.maps_ol_el.reshape(env.num_links, 1, 1)
                unit_delay_edge_ts = tf.tensor_scatter_nd_update(unit_delay_edge_ts, indices, link_delay)
                indices = obj.maps_on_el.reshape((-1, 1, 1))
                unit_delay_edge_ts = tf.tensor_scatter_nd_update(unit_delay_edge_ts, indices, node_delay)
                unit_delay_edge_ts = tf.reshape(unit_delay_edge_ts, (obj.num_edges_ext, 1))
                unit_job_edge_ts = tf.math.multiply_no_nan(unit_delay_edge_ts, routes_ts)
                delay_job_edge_ts = tf.math.multiply_no_nan(jobs_data_ts, unit_job_edge_ts)
                delay_job_edge_ts = tf.math.maximum(delay_job_edge_ts, routes_ts)
                loss_fn = tf.reduce_sum(delay_job_edge_ts)
                grad_routes = gg.gradient(loss_fn, routes_ts)

            delay_emp_gnn = np.nansum(delay_links_gnn, axis=0) + np.nansum(delay_nodes_gnn, axis=0)
            delay_critic = delay_job_edge_ts.numpy().sum(axis=0)
            delay_err = delay_critic - delay_emp_gnn
            # Manual processing of gradient toward distances, based on decisions
            grad_dist_np = np.zeros_like(delay_mtx_np)
            grad_routes_np = grad_routes.numpy()
            # Method 2: assign grad_route to corresponding downstream shortest path bias
            # grad_distances = foo(grad_routes)
            with tf.GradientTape() as gl:
                gl.watch(unit_delay_edge_ts)
                bias_ts = tf.zeros_like(routes_ts)
                for jidx in range(env.num_jobs):
                    job = env.jobs[jidx]
                    flow = env.flows[jidx]
                    n1 = flow.dst + env.num_nodes
                    tmp_ts = tf.constant([0], dtype=unit_delay_edge_ts.dtype)
                    for n0 in reversed(flow.route):
                        if (n0, n1) in obj.link_list_ext:
                            lidx = obj.link_list_ext.index((n0, n1))
                        elif (n1, n0) in obj.link_list_ext:
                            lidx = obj.link_list_ext.index((n1, n0))
                        else:
                            raise ValueError("Link not exist, check route")
                        # bias_ts[lidx, jidx] = unit_delay_edge_ts[lidx] + tmp_ts
                        indices = [[lidx, jidx]]
                        updates = unit_delay_edge_ts[lidx] + tmp_ts
                        bias_ts = tf.tensor_scatter_nd_update(bias_ts, indices, updates)
                        tmp_ts = bias_ts[lidx, jidx]
                        if n0 == job.source_node:
                            break
                        else:
                            n1 = n0
                grad_edge_ts = gl.gradient(bias_ts, unit_delay_edge_ts, output_gradients=-grad_routes_np, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                grad_edge_np = grad_edge_ts.numpy()
            for lidx in range(len(obj.link_list_ext)):
                n0, n1 = obj.link_list_ext[lidx]
                if n1 >= env.num_nodes:
                    grad_dist_np[n0, n0] = grad_edge_np[lidx]
                else:
                    grad_dist_np[n0, n1] = grad_edge_np[lidx]
                    grad_dist_np[n1, n0] = grad_edge_np[lidx]

            # Method 1: just assign negative gradient to corresponding link & node
            # grad_distances = -grad_routes
            # for jidx in range(env.num_jobs):
            #     job = env.jobs[jidx]
            #     flow = env.flows[jidx]
            #     n0 = job.source_node
            #     if n0 != flow.dst:
            #         for n1 in flow.route[1:]:
            #             if (n0, n1) in obj.link_list_ext:
            #                 lidx = obj.link_list_ext.index((n0, n1))
            #             elif (n1, n0) in obj.link_list_ext:
            #                 lidx = obj.link_list_ext.index((n1, n0))
            #             else:
            #                 raise ValueError("Link not exist, check route")
            #             grad_dist_np[n0, n1] -= grad_routes_np[lidx, jidx]
            #             grad_dist_np[n1, n0] -= grad_routes_np[lidx, jidx]
            #             n0 = n1
            #     n1 = n0 + env.num_nodes
            #     lidx = obj.link_list_ext.index((n0, n1))
            #     grad_dist_np[n0, n0] -= grad_routes_np[lidx, jidx]

            # manually compute the gradient of MSE loss, add to grad_distances
            delay_unit_gnn[np.isinf(delay_unit_gnn)] = np.nan
            loss_mse = np.nanmean((delay_mtx_np - delay_unit_gnn) ** 2)
            grad_dist_mse = 0.001 * (delay_mtx_np - delay_unit_gnn)
            # grad_dist_mse = (1 / obj.num_edges_ext) * (delay_mtx_np - 100*delay_unit_gnn)
            grad_dist_np += np.nan_to_num(grad_dist_mse, nan=0.0)
            # grad_distances = tf.convert_to_tensor(grad_dist_np)
            # delay_mtx_ts
            # Loss function for actor GNN
            gradients = g.gradient(delay_mtx_ts, self.model.trainable_weights, output_gradients=grad_dist_np)
            # gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
            self.memorize(gradients, loss_fn.numpy(), loss_mse)

        flows_gnn = deepcopy(env.flows)
        return delay_mtx_np, delay_links_gnn, delay_nodes_gnn, delay_unit_gnn, flows_gnn, loss_fn.numpy(), loss_mse

    def log_init(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('..', 'logs', 'gradient_tape', current_time, 'train')
        test_log_dir = os.path.join('..', 'logs', 'gradient_tape', current_time, 'train')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def log_scalar(self, name, variable, step, test=False):
        if not test:
            with self.train_summary_writer.as_default():
                tf.summary.scalar(name, variable, step=step)
        else:
            with self.test_summary_writer.as_default():
                tf.summary.scalar(name, variable, step=step)


# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
