# [Congestion-aware Distributed Task Offloading in Wireless Multi-hop Networks Using Graph Neural Networks](http://arxiv.org/abs/2312.02471)

This software was created by William Marsh Rice University under Army Research Office (ARO) Award Number W911NF-24-2-0008 and W911NF-19-2-0269. ARO, as the Federal awarding agency, reserves a royalty-free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).

## Abstract

Computational offloading has become an enabling component for edge intelligence in mobile and smart devices. Existing offloading schemes mainly focus on mobile devices and servers, while ignoring the potential network congestion caused by tasks from multiple mobile devices, especially in wireless multi-hop networks. To fill this gap, we propose a low-overhead, congestion-aware distributed task offloading scheme by augmenting a distributed greedy framework with graph-based machine learning. In simulated wireless multi-hop networks with 20-110 nodes and a resource allocation scheme based on shortest path routing and contention-based link scheduling, our approach is demonstrated to be effective in reducing congestion or unstable queues under the context-agnostic baseline, while improving the execution latency over local computing.

__Key words__: Computational offloading, queueing networks, wireless multi-hop networks, graph neural networks, shortest path.

- Preprint <http://arxiv.org/abs/2312.02471>

## Usage

To create data set, run 
```
mkdir ./data
bash bash/data_gen_aco.sh
```


Training
```
bash bash/train.sh
```

Testing

```
bash bash/test.sh
```


To plot results, launch `src/results_plot-Adhoc.ipynb`


