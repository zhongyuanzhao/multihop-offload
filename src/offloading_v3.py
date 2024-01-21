# python3
# This software was created by William Marsh Rice University under Army Research Office (ARO) 
# Award Number W911NF-24-2-0008 and W911NF-19-2-0269. ARO, as the Federal awarding agency, reserves 
# a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this 
# software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
from __future__ import division
from __future__ import print_function

import queue
import sys
import os
import time
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scipy.io as sio
# import sparse
import copy
np.set_printoptions(threshold=np.inf)
# Import utility functions
from util import *


class AdhocCloud:
    def __init__(self, num_nodes, t_max=1000, seed=3, m=2, pos=None, cf_radius=0.0, gtype='ba', trace=False):
        self.num_nodes = int(num_nodes)
        self.T = int(t_max)
        self.seed = int(seed) # other format such as int64 won't work
        self.m = int(m)
        self.gtype = gtype.lower()
        self.trace = trace
        self.cf_radius = cf_radius
        self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, self.gtype)
        if self.gtype == 'ba':
            graph_c = nx.barabasi_albert_graph(self.num_nodes, self.m, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'grp':
            graph_c = nx.gaussian_random_partition_graph(self.num_nodes, 15, 3, 0.4, 0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'ws':
            graph_c = nx.connected_watts_strogatz_graph(self.num_nodes, k=6, p=0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'er':
            graph_c = nx.fast_gnp_random_graph(self.num_nodes, 15.0/float(self.num_nodes), seed=self.seed)  # Conectivity graph
        elif '.mat' in self.gtype:
            postfix = self.gtype.split('/')[-1]
            postfix = postfix.split('.')[0]
            self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, postfix)
            try:
                mat_contents = sio.loadmat(self.gtype)
                adj = mat_contents['adj'].todense()
                pos = mat_contents['pos_c']
                graph_c = nx.from_numpy_array(adj)
            except:
                raise RuntimeError("Error creating object, check {}".format(self.gtype))
        else:
            raise ValueError("unsupported graph model for connectivity graph")
        self.connected = nx.is_connected(graph_c)
        self.graph_c = graph_c
        self.node_positions(pos)
        self.nodes_init()
        self.box = self.bbox()
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        self.link_list = list(self.graph_i.nodes)
        self.edge_maps = np.zeros((self.num_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_links,), dtype=int)
        self.link_mapping()
        if cf_radius > 0.5:
            self.add_conflict_relations(cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.cf_degs = np.asarray(self.adj_i.sum(axis=0)).flatten()
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        self.clear_all_jobs()

    def random_walk(self, ss=0.1, n=10):
        disconnected = True
        while disconnected:
            mask = np.random.choice(np.arange(0, self.num_nodes), size=n, replace=False)
            d_pos = np.random.normal(0, ss, size=(n, 2))
            pos_c_np = self.pos_c_np
            pos_c_np[mask, :] += d_pos
            b_min = np.min(self.box)
            b_max = np.max(self.box)
            pos_c_np = pos_c_np.clip(b_min, b_max)
            d_mtx = distance_matrix(pos_c_np, pos_c_np)
            adj_mtx = np.zeros([self.num_nodes, self.num_nodes], dtype=int)
            adj_mtx[d_mtx <= 1.0] = 1
            np.fill_diagonal(adj_mtx, 0)
            graph_c = nx.from_numpy_array(adj_mtx)
            self.connected = nx.is_connected(graph_c)
            disconnected = not self.connected
        return graph_c, pos_c_np

    def topology_update(self, graph_c, pos_c_np):
        self.graph_c = graph_c
        self.node_positions(pos_c_np)
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        link_list_old = self.link_list
        self.link_list = list(self.graph_i.nodes)
        new_links_map = np.zeros((self.num_links,), dtype=int)
        for i in range(self.num_links):
            e0, e1 = self.link_list[i]
            if (e0, e1) in link_list_old:
                j = link_list_old.index((e0, e1))
            elif (e1, e0) in link_list_old:
                j = link_list_old.index((e1, e0))
            else:
                j = -1
            new_links_map[i] = j
        self.edge_maps = np.zeros((self.num_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_links,), dtype=int)
        self.link_mapping()
        if self.cf_radius > 0.5:
            self.add_conflict_relations(self.cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        self.W = np.zeros((self.num_links, self.T))
        self.WSign = np.ones((self.num_links, self.T))
        self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
        self.link_comd_cnts = np.zeros((self.num_links, self.num_nodes))
        return new_links_map

    class Job:
        def __init__(self, source_node, arrival_rate, ul_data=100, dl_data=1):
            self.source_node = source_node
            self.arrival_rate = arrival_rate
            self.ul_data = ul_data
            self.dl_data = dl_data
            self.status = 0
            self.id = str(source_node) + "_" + str(ul_data) + "_" + str(dl_data)

    class Flow:
        def __init__(self, job_id, src, dst):
            self.src = src
            self.dst = dst
            self.route = []
            self.job_id = job_id
            self.rate = 0
            self.status = 0
            self.nhop = 0
            self.ul_log = {}
            self.dl_log = {}

    def node_positions(self, pos):
        if pos is None:
            pos_file = os.path.join('..', 'pos', "graph_c_pos_{}.p".format(self.case_name))
            if not os.path.isfile(pos_file):
                pos_c = nx.spring_layout(self.graph_c)
                with open(pos_file, 'wb') as fp:
                    pickle.dump(pos_c, fp, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(pos_file, 'rb') as fp:
                    pos_c = pickle.load(fp)
        elif isinstance(pos, str) and pos == 'new':
            pos_c = nx.spring_layout(self.graph_c)
        elif isinstance(pos, np.ndarray):
            pos_c = dict(zip(list(range(self.num_nodes)), pos))
        else:
            raise ValueError("unsupported pos format in backpressure object initialization")
        self.pos_c = pos_c

    def nodes_init(self):
        self.roles = np.zeros((self.num_nodes,), dtype=float)
        self.proc_bws = 2*np.ones_like(self.roles)
        self.servers = []
        self.relays = []

    def add_server(self, node, proc_bw):
        self.roles[node] = 1
        self.proc_bws[node] = proc_bw
        self.servers.append(node)

    def add_relay(self, node):
        self.roles[node] = 2
        self.proc_bws[node] = 0
        self.relays.append(node)

    def bbox(self):
        pos_c = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            pos_c[i, :] = self.pos_c[i]
        self.pos_c_np = pos_c
        return [np.amin(pos_c[:,0])-0.05, np.amax(pos_c[:,1])+0.05, np.amin(pos_c[:,1])-0.05, np.amax(pos_c[:,1])+0.05]

    def add_conflict_relations(self, cf_radius):
        """
        Adding conflict relationship between links whose nodes are within cf_radius * median_link_distance
        :param cf_radius: multiple of median link distance
        :return: None (modify self.adj_i, and self.graph_i inplace)
        """
        pos_c_vec = np.zeros((self.num_nodes, 2))
        for key, item in self.pos_c.items():
            pos_c_vec[key, :] = item
        dist_mtx = distance_matrix(pos_c_vec, pos_c_vec)
        rows, cols = np.nonzero(self.adj_c)
        link_dist = dist_mtx[rows, cols]
        median_dist = np.nanmedian(link_dist)
        intf_dist = cf_radius * median_dist
        for link in self.link_list:
            src, dst = link
            intf_nbs_s, = np.where(dist_mtx[src, :] < intf_dist)
            intf_nbs_d, = np.where(dist_mtx[dst, :] < intf_dist)
            intf_nbs = np.union1d(intf_nbs_s,intf_nbs_d)
            for v in intf_nbs:
                _, nb2hop = np.nonzero(self.adj_c[v])
                for u in nb2hop:
                    if {v, u} == {src, dst}:
                        continue
                    elif (v, u) in self.link_list:
                        self.graph_i.add_edge((v, u), (src, dst))
                    elif (u, v) in self.link_list:
                        self.graph_i.add_edge((u, v), (src, dst))
                    else:
                        pass
                        # raise RuntimeError("Something wrong with adding conflicting edge")
        self.adj_i = nx.adjacency_matrix(self.graph_i)

    def link_mapping(self):
        # Mapping between links in connectivity graph and nodes in conflict graph
        j = 0
        self.link_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for e0, e1 in self.graph_c.edges:
            try:
                i = self.link_list.index((e0, e1))
            except:
                i = self.link_list.index((e1, e0))
            self.edge_maps[j] = i
            # self.edge_maps[j + self.num_links] = i + self.num_links
            self.edge_maps_rev[i] = j
            # self.edge_maps_rev[i + self.num_links] = j + self.num_links
            self.link_matrix[e0, e1] = i
            self.link_matrix[e1, e0] = i
            j += 1

    def add_job(self, src, rate=0.1, ul=100, dl=1):
        fi = self.Job(src, rate, ul, dl)
        self.jobs.append(fi)
        self.num_jobs = len(self.jobs)

    def clear_all_jobs(self):
        self.jobs = []
        self.flows = []

    def links_init(self, rates, std=2):
        if hasattr(rates, '__len__'):
            assert len(rates) == self.num_links
            stds = std*np.ones_like(rates)
        else:
            stds = std
        link_rates = np.zeros((self.num_links,))
        link_rates[:] = np.clip(np.random.normal(rates, stds, size=(self.num_links,)), 0, rates + 3*std)
        self.link_rates = np.round(link_rates)

    def graph_expand(self):
        """
        Call this function after initialization (jobs)
        :return:
        """
        class Empty:
            pass
        obj = Empty() # create an empty object
        obj.gc_ext = copy.deepcopy(self.graph_c)
        # add nodes to non-relay nodes
        for idx in range(self.num_nodes):
            if self.roles[idx] < 2:
                new_node = self.num_nodes + idx
                obj.gc_ext.add_node(new_node)
                obj.gc_ext.add_edge(idx, new_node)
        # create a vector of arrival rates of jobs
        jobs_info = np.zeros((self.num_nodes,))
        for i in range(self.num_jobs):
            src = self.jobs[i].source_node
            arr = self.jobs[i].arrival_rate * self.jobs[i].ul_data
            jobs_info[src] += arr

        obj.gi_ext = nx.line_graph(obj.gc_ext)
        j = 0
        obj.link_list_ext = list(obj.gi_ext.nodes)
        obj.num_edges_ext = len(obj.link_list_ext)
        obj.edge_maps_ext = np.zeros((obj.num_edges_ext,), dtype=int)
        obj.edge_maps_rev_ext = np.zeros((obj.num_edges_ext,), dtype=int)
        obj.edge_self_loop = np.zeros((obj.num_edges_ext,), dtype=int)
        obj.edge_as_server = np.zeros((obj.num_edges_ext,), dtype=int)
        obj.maps_ol_el = np.zeros((self.num_links,), dtype=int)
        obj.maps_on_el = -np.ones((self.num_nodes,), dtype=int)
        obj.edge_rate_ext = np.zeros((obj.num_edges_ext,))
        obj.jobs_arrivals = np.zeros((obj.num_edges_ext,))
        # edge_rate_mapped2 = np.zeros((obj.num_edges_ext,))
        for e0, e1 in obj.gc_ext.edges:
            try:
                i = obj.link_list_ext.index((e0, e1))
            except:
                i = obj.link_list_ext.index((e1, e0))
            obj.edge_maps_ext[j] = i
            obj.edge_maps_rev_ext[i] = j
            if (e0, e1) in self.link_list:
                ii = self.link_list.index((e0, e1))
                obj.edge_rate_ext[i] = self.link_rates[ii]
                obj.maps_ol_el[ii] = i
                # edge_rate_mapped2[j] = self.link_rates[ii]
            elif (e1, e0) in self.link_list:
                ii = self.link_list.index((e1, e0))
                obj.edge_rate_ext[i] = self.link_rates[ii]
                obj.maps_ol_el[ii] = i
                # edge_rate_mapped2[j] = self.link_rates[ii]
            elif e0 >= self.num_nodes:
                obj.edge_rate_ext[i] = self.proc_bws[e1]
                obj.edge_self_loop[i] = 1
                if self.roles[e1] == 1:
                    obj.edge_as_server[i] = 1
                obj.maps_on_el[e1] = i
                obj.jobs_arrivals[i] = jobs_info[e1]
                # edge_rate_mapped2[j] = self.proc_bws[e1]
            elif e1 >= self.num_nodes:
                obj.edge_rate_ext[i] = self.proc_bws[e0]
                obj.edge_self_loop[i] = 1
                if self.roles[e0] == 1:
                    obj.edge_as_server[i] = 1
                obj.maps_on_el[e0] = i
                obj.jobs_arrivals[i] = jobs_info[e0]
                # edge_rate_mapped2[j] = self.proc_bws[e0]
            else:
                pass
            j += 1
        obj.maps_on_el = obj.maps_on_el[obj.maps_on_el >= 0]
        edge_rate_mapped = obj.edge_rate_ext[obj.edge_maps_ext]
        nx.set_edge_attributes(obj.gc_ext, edge_rate_mapped, "rate")
        nx.set_node_attributes(obj.gi_ext, obj.edge_rate_ext, "rate")
        nx.set_node_attributes(obj.gi_ext, obj.edge_self_loop, "loop")
        nx.set_node_attributes(obj.gi_ext, obj.jobs_arrivals, "job")
        return obj

    def dmtx_baseline(self):
        """
        Create a distance matrix based on input distance list
        :param dist_list:
        :return:
        """
        dmtx = np.full((self.num_nodes, self.num_nodes), np.inf)
        dlist = np.zeros((self.num_links,))
        diag_bw = np.array(self.proc_bws)
        dproc = 1.0/diag_bw
        np.fill_diagonal(dmtx, dproc)
        for lidx in range(self.num_links):
            link = self.link_list[lidx]
            # link_rate = self.link_rates[lidx] / (self.cf_degs[lidx] + 1 )
            link_rate = self.link_rates[lidx] #/ (self.cf_degs[lidx] + 1 )
            delay = 1.0 / link_rate
            src, dst = link
            dlist[lidx] = delay
            dmtx[src, dst] = delay
            dmtx[dst, src] = delay
        return dmtx, dlist, dproc

    def local_compute(self, unit_delay_servers):
        """
        Distributed offloading heuristics based on spmtx and dmtx
        :param prob: Probabilistic decision flag
        :return:
        """
        decisions = []
        delays = []
        self.flows = []
        for job in self.jobs:
            ul_data = job.ul_data
            src = job.source_node
            dst = src
            local_delay = unit_delay_servers[src] * ul_data
            flow = self.Flow(job.id, src, src)
            route = [src, src]
            nhop = 0
            job_delay = np.max([local_delay, 1])
            flow.route = route
            flow.nhop = nhop
            self.flows.append(flow)
            decisions.append(dst)
            delays.append(job_delay)
        return decisions, delays

    def offloading(self, spmtx_in, hpmtx, explore=0.0, prob=False):
        """
        Distributed offloading heuristics based on spmtx and dmtx
        :param spmtx_in: distance matrix of shortest path
        :param prob: Probabilistic decision flag
        :return:
        """
        unit_delay_servers = np.diagonal(spmtx_in)
        spmtx = copy.deepcopy(spmtx_in)
        np.fill_diagonal(spmtx, 0)
        decisions = []
        delays = []
        self.flows = []
        for job in self.jobs:
            dl_data = job.dl_data
            ul_data = job.ul_data
            src = job.source_node
            dst = src
            local_delay = unit_delay_servers[src] * ul_data
            server_ul_delays = spmtx[src, self.servers] * ul_data
            server_dl_delays = spmtx[self.servers, src] * dl_data
            server_proc_delays = unit_delay_servers[self.servers] * ul_data
            # should consider a minimal delay as the number of hops of shortest path
            ul_delays = np.max([server_ul_delays,hpmtx[src,self.servers]], axis=0)
            dl_delays = np.max([server_dl_delays, hpmtx[self.servers, src]], axis=0)
            proc_delays = np.max([server_proc_delays, np.ones_like(server_proc_delays)], axis=0)
            server_delays = ul_delays + dl_delays + proc_delays
            costs = np.append(server_delays, local_delay)
            if np.random.uniform(0, 1) < explore:
                jidx = np.random.choice(range(len(self.servers) + 1))
            elif not prob:
                jidx = np.argmin(costs)
            else:
                probs = softmax(costs)
                jidx = np.random.choice(range(len(self.servers)+1), p=probs)

            flow = self.Flow(job.id, src, src)
            if jidx < len(self.servers):
                job_delay = server_delays[jidx]
                dst = self.servers[jidx]
                flow.dst = dst
                route, nhop = self.routing(flow, spmtx)
            else:
                route = [src, src]
                nhop = 0
                job_delay = local_delay
            flow.route = route
            flow.nhop = nhop
            self.flows.append(flow)
            decisions.append(dst)
            delays.append(job_delay)
        return decisions, delays

    def routing(self, flow, spmtx):
        src = flow.src
        dst = flow.dst
        route = [src]
        node = src
        num_hop = 0
        while node != dst:
            _, nbs = np.nonzero(self.adj_c[node])
            nb_idx = np.argmin(spmtx[nbs, dst])
            num_hop += 1
            node = nbs[nb_idx]
            route.append(node)
        return route, num_hop

    def run(self):
        """
        Compute the per flow per link delay based on all the information
        A link can be considered as a queueing system, e.g. M/M/1 queue
        The arrival rate is the total arrival rate of packets, lambda
        The service rate is the average link rate of the link, mu = r/(deg+1)
        The average response time can be found as 1/(mu-lambda), Little's law
        :return:
        """
        assert self.num_jobs == len(self.flows)
        server_delay_empirical = np.full((self.num_nodes, self.num_jobs), np.nan)
        link_delay_empirical = np.full((self.num_links, self.num_jobs), np.nan)
        link_load_array = np.zeros_like(link_delay_empirical)
        server_load_array = np.zeros((self.num_nodes,))
        flow_load_array = np.zeros((self.num_jobs,))
        flow_ul_size = np.zeros((self.num_jobs,))
        flow_dl_size = np.zeros((self.num_jobs,))
        for idx in range(self.num_jobs):
            job = self.jobs[idx]
            flow = self.flows[idx]
            job_id = job.id
            arrival_rate = job.arrival_rate
            ul_rate = job.ul_data * arrival_rate
            dl_rate = job.dl_data * arrival_rate
            flow_load_array[idx] = ul_rate
            flow_ul_size[idx] = job.ul_data
            flow_dl_size[idx] = job.dl_data
            assert job_id == flow.job_id
            route = flow.route
            dst = flow.dst
            if job.source_node != dst:
                n0 = job.source_node
                for n1 in route[1:]:
                    if (n0, n1) in self.link_list:
                        lidx = self.link_list.index((n0, n1))
                    elif (n1, n0) in self.link_list:
                        lidx = self.link_list.index((n1, n0))
                    else:
                        raise ValueError("Link not exist, check route")
                    link_load_array[lidx, idx] += ul_rate + dl_rate
                    n0 = n1
            server_load_array[dst] += ul_rate

        link_lambda = link_load_array.sum(axis=1)
        # The probability of a link with empty queue is (1-lambda/mu)
        link_mu = self.link_rates / (self.cf_degs + 1)
        for it in range(10):
            link_busy = np.clip(link_lambda/link_mu, 0, 1.0)
            neighbor_busy = link_busy * self.adj_i
            # link_ratio = link_busy / (link_busy + neighbor_busy)
            link_ratio = 1.0 / (1.0 + neighbor_busy)
            link_mu = self.link_rates * link_ratio

        unit_link_delay_matrix = np.full((self.num_nodes, self.num_nodes), fill_value=np.nan)
        # unit_link_delay_empirical = 1/(link_mu - link_lambda)
        # unit_node_delay_empirical = 1/(self.proc_bws - server_load_array)
        # unit_link_delay_congest = (float(self.T)*link_lambda)/(101.0*link_mu)
        # unit_node_delay_congest = (float(self.T)*server_load_array)/(100.0*self.proc_bws)
        # link_congest = np.where(unit_link_delay_empirical < 0)
        # node_congest = np.where(unit_node_delay_empirical < 0)
        # unit_link_delay_empirical[link_congest] = unit_link_delay_congest[link_congest]
        # unit_node_delay_empirical[node_congest] = unit_node_delay_congest[node_congest]
        # for lidx in range(self.num_links):
        #     (n0, n1) = self.link_list[lidx]
        #     unit_link_delay_matrix[n0, n1] = unit_link_delay_empirical[lidx]
        #     unit_link_delay_matrix[n1, n0] = unit_link_delay_empirical[lidx]
        # np.fill_diagonal(unit_link_delay_matrix, unit_node_delay_empirical)
        for idx in range(self.num_jobs):
            job = self.jobs[idx]
            flow = self.flows[idx]
            dst = flow.dst
            nhop = float(flow.nhop)
            if job.source_node != dst:
                n0 = job.source_node
                for n1 in flow.route[1:]:
                    if (n0, n1) in self.link_list:
                        lidx = self.link_list.index((n0, n1))
                    elif (n1, n0) in self.link_list:
                        lidx = self.link_list.index((n1, n0))
                    else:
                        raise ValueError("Link not exist, check route")
                    # unit_delay = unit_link_delay_matrix[n0, n1]
                    unit_delay = 1 / (link_mu[lidx] - link_lambda[lidx])
                    if link_mu[lidx] - link_lambda[lidx] <= 0:
                        unit_delay = float(self.T) * (link_lambda[lidx] / ((job.ul_data + job.dl_data) * link_mu[lidx]))
                    unit_link_delay_matrix[n0, n1] = unit_delay
                    unit_link_delay_matrix[n1, n0] = unit_delay
                    link_delay_empirical[lidx, idx] = np.max([job.ul_data * unit_delay, nhop]) + np.max([job.dl_data * unit_delay, nhop])
                    n0 = n1
            # unit_delay = unit_link_delay_matrix[dst, dst]
            unit_delay = 1 / (self.proc_bws[dst] - server_load_array[dst])
            if self.proc_bws[dst] - server_load_array[dst] <= 0:
                unit_delay = float(self.T) * (server_load_array[dst] / (job.ul_data * self.proc_bws[dst]))
            unit_link_delay_matrix[dst, dst] = unit_delay
            server_delay_empirical[dst, idx] = np.max([job.ul_data * unit_delay, 1])
        return link_delay_empirical, server_delay_empirical, unit_link_delay_matrix

    def plot_routes(self, link_delays, node_delays, opt, with_labels=True):
        delay_f = np.nan_to_num(link_delays).sum(axis=1)
        delay_s = np.nan_to_num(node_delays).sum(axis=1) * 100
        bbox = self.bbox()
        mobile_nodes = []
        server_nodes = self.servers
        for f in self.flows:
            src = f.src
            mobile_nodes.append(src)

        weights = delay_f/10 + 1
        # weights = np.log10(delay_f+1.26)*10
        weights = weights[self.edge_maps]
        vis_network(
            self.graph_c,
            mobile_nodes,
            server_nodes,
            self.pos_c,
            weights,
            delay_s,
            with_labels
        )
        fig_name = "offloading_flow_routes_visual_{}_cf{:.1f}_opt{}.png".format(
            self.case_name,
            self.cf_radius,
            opt)
        fig_name = os.path.join("..", "fig", fig_name)
        ax = plt.gca()
        ax.set_xlim(bbox[0:2])
        ax.set_ylim(bbox[2:4])
        # plt.tight_layout(pad=-0.1)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close()
        print("Flow plot saved to {}".format(fig_name))

    def plot_metrics(self, opt):
        arrivals = np.sum(self.flows_arrivals, axis=0)
        pkts_in_network = np.sum(self.flow_pkts_in_network, axis=0)
        departures = np.sum(self.flows_sink_departures, axis=0)

        plt.plot(arrivals)
        plt.plot(departures)
        plt.plot(pkts_in_network)

        plt.suptitle('Departures, Arrivals, and Current amount pkts in network')
        plt.xlabel('T')
        plt.ylabel('the number of packages')
        plt.legend(['Exogenous arrivals', 'Sink departures', 'Pkts in network'], loc='upper right')
        fig_name = "flow_packets_arrivals_per_timeslot_{}_cf{:.1f}_opt_{}.png".format(self.case_name, self.cf_radius, opt)
        fig_name = os.path.join("..", "fig", fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close()
        print("Metrics plot saved to {}".format(fig_name))
        return arrivals, pkts_in_network, departures


def main(args):
    # Configuration
    NUM_NODES = 15
    # LAMBDA = 1 # we are not use it
    T = 500
    link_rate = 50
    cf_radius = 0.0 # relative conflict radius based on physical distance model

    opt = int(args[0])
    seed = 3
    arrival_const = 2.0

    # Create fig folder if not exist
    if not os.path.isdir(os.path.join("..", "fig")):
        os.mkdir(os.path.join("..", "fig"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "pos")):
        os.mkdir(os.path.join("..", "pos"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    start_time = time.time()
    # bp_env = Backpressure(NUM_NODES, T, seed)
    ec_env = AdhocCloud(NUM_NODES, T, seed, cf_radius=cf_radius, trace=True)

    ec_env.add_server(4, proc_bw=300)
    ec_env.add_server(14, proc_bw=300)
    ec_env.add_server(13, proc_bw=300)
    ec_env.add_server(12, proc_bw=300)
    # ec_env.add_server(8, proc_bw=100)
    ec_env.add_server(1, proc_bw=200)
    ec_env.add_relay(3)
    ec_env.add_relay(0)

    ec_env.add_job(10, rate=arrival_const*0.05)
    ec_env.add_job(11, rate=arrival_const*0.03)
    ec_env.add_job(7, rate=arrival_const*0.02)
    ec_env.add_job(8, rate=arrival_const*0.02)
    ec_env.add_job(6, rate=arrival_const*0.05)

    ec_env.links_init(link_rate)

    dmtx_bl, dlist_bl, dproc_bl = ec_env.dmtx_baseline()
    for link, delay in zip(ec_env.link_list, dlist_bl):
        src, dst = link
        ec_env.graph_c[src][dst]["delay"] = delay

    # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
    # sp_hop = all_pairs_shortest_paths(ec_env.graph_c, weight=None)
    sp_baseline = all_pairs_shortest_paths(ec_env.graph_c, weight="delay")
    np.fill_diagonal(sp_baseline, dproc_bl)

    decisions, delay_est = ec_env.offloading(sp_baseline)

    logfile = os.path.join("..", "out", "Output_{}_opt_{}.txt".format(ec_env.case_name, opt))
    with open(logfile, "a") as f:
        print("Edges:", file=f)
        print(ec_env.graph_i.nodes(), file=f)
        print("Link Rates:", file=f)
        print(ec_env.link_rates, file=f)

    print("Init graph('{}') in {:.3f} seconds".format(ec_env.case_name, time.time() - start_time),
          ": conflict radius {}, degree {:.2f}".format(ec_env.cf_radius, ec_env.mean_conflict_degree))
    start_time = time.time()

    delay_links, delay_nodes = ec_env.run()

    print("Main loop {} time slots in {:.3f} seconds".format(T, time.time() - start_time))
    ec_env.plot_routes(delay_links, delay_nodes, opt)

    delay_emp = np.nansum(delay_links, axis=0) + np.nansum(delay_nodes, axis=0)
    print("Delay estimation: {}\nDelay empircal: {}".format(delay_est, delay_emp))
    print("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
