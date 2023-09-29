#%%
import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

#%%


def get_value(pds, idx, opts):
    bench_graph = {}
    for item in pds:
        layer_id = item['layer_id'][idx]
        if layer_id in bench_graph:
            bench_graph[layer_id] += item['kernel_compute_time'][idx] / 1000 / 1000
        else:
            bench_graph[layer_id] = item['kernel_compute_time'][idx] / 1000 / 1000
    
    for key in opts.keys():
        if bench_graph[key] > opts[key]:
            bench_graph[key] = opts[key]

    return (bench_graph, sum(bench_graph.values()))

#%%

#%%
def get_acltuner_all(dir):
    algs = os.listdir(dir)
    algs = list(map(lambda x: int(x), algs))
    algs.sort()
    # print(algs)

    layers = []
    for alg in algs:
        layers.append(os.listdir("{0}/{1}".format(dir, alg)))
    opt_graph = []
    bench_graph = {}
    for idx in range(len(algs)):
        # for alg in range(len(algs)):
        pds = []
        for layer in layers[idx]:
            # print("{0}/{1}/{2}".format(dir, algs[idx], layer))
            d = pd.read_csv("{0}/{1}/{2}".format(dir, algs[idx], layer))
            d = d.drop(["layer_name", "kernel_name", "conv_id", "conv_alg", "conv_simd", "core0", "core1", "core2", "core3", "core4", "core5"], axis=1)
            # items = list(map(lambda x: x['kernel_compute_time'], d))
            pds.append(d)

        rows = pds[0].shape[0]
        for loop in range(rows):
            bench_graph, cur_max = get_value(pds, loop, bench_graph)
            opt_graph.append(cur_max)
        
        print(idx)

    return opt_graph
# plt.plot(opt_graph)


#%%

# pds[0]

# %%

# %%
