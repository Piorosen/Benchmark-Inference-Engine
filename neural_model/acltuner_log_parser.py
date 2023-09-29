#%%
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

#%%
os.getcwd()

#%%
def load_file(path, file: str):
    filePath = path + '/' + file
    print(filePath)
    with open(filePath, 'r') as f:
        return json.load(f)

# %% log 파일 목록을 읽음
def divid_log(path: str): # e.g. './odroid/resnet18/log'
    v = os.listdir(path)
    v.sort()
    data = {}
    for x in v:
        if '.DS_Store' in x:
            continue
        key = x[:7]
        key = key.split(".")[0].replace("log_", "")
        key = int(key)
        if key in data:
            data[key].append(x)
        else:
            data[key] = [x]
    # print(data[-1])
    return data
# %%
def get_alg(d):
    return list(d.keys())

def load_alg(data, key):
    inte = []
    print(key)
    for idx in range(len(data[key])):
        obj = load_file(path, data[key][idx])
        inte.extend(obj)
    
    kernel_size = len(inte[0]['kernels']) / 8
    print(kernel_size)
    return inte, int(kernel_size)

# %% 

def extract_feature(kernel_size, objs):
    k_data = {}
    kc_data = {}
    for k in range(kernel_size):
        layer_name = objs[0]['kernels'][k * 8]['layer_name']        
        layer_id = objs[0]['kernels'][k * 8]['layer_id']
        kernel_name = objs[0]['kernels'][k * 8]['kernel_name']
        
        if 'NeonConvolution2dWorkload' in layer_name:
            d = kernel_name.split("/")
            if len(d) > 1:
                kc_data[layer_id] = d[1]
            if layer_id in k_data:
                k_data[layer_id] += 1
            else: 
                k_data[layer_id] = 1
    return (k_data, kc_data)        

def write(kernel_size, objs, k_data, kc_data, model_name, id):
    if not os.path.isdir('./{0}/{1}'.format(model_name, id)):
        os.makedirs('./{0}/{1}'.format(model_name, id))
    
    for k in range(kernel_size):
        kernel_name = objs[0]['kernels'][k * 8]['kernel_name']
        layer_id = objs[0]['kernels'][k * 8]['layer_id']
        layer_name = objs[0]['kernels'][k * 8]['layer_name']

        with open("./{0}/{1}/{2}.{3}.{4}.{5}.csv".format(model_name, id,
                                                     layer_id, layer_name, 
                                                     kernel_name.replace("/", "."), k), 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["layer_id", "layer_name", "kernel_name", "conv_id", 
                            "conv_alg", "conv_simd", 
                            "cluster0_worksize","cluster1_worksize", "total_worksize", 
                            "core0","core1","core2","core3","core4","core5",
                            "kernel_compute_time"])
            
            conv_alg = 0
            conv_simd = 0
            conv_id = 0
            
            # algs = {1:'direct', 2:'im2col', 3:'winograd'}
            simds = {'a64_hybrid_fp32_mla_4x24':1, 
                    'a64_hybrid_fp32_mla_6x16':2, 
                    'a64_hybrid_fp32_mla_8x4':3, 
                    'a64_sgemm_8x6':4, 
                    'a64_sgemm_8x12':5}
            
            if 'NeonConvolution2dWorkload' in layer_name:
                conv_id = 1
                # print(kernel_name)
                conv_simd = simds[kc_data[layer_id]]
                conv_alg = k_data[layer_id]
                # extract: conv_alg and conv_simd
            
            for step in range(1, len(objs)): # avg core_time, kernel compute, 
                cluster = [0,0]
                core = [0,0,0,0,0,0]
                kernel_compute = 0
                    
                for repeat in range(8):
                    # print(objs[step]['kernels'][k * 8 + repeat])
                    pa = objs[step]['kernels'][k * 8 + repeat]
                    core_list = pa['core']
                    work_list = pa['work']
                    for c in core_list:
                        core[c['id']] += c['time']
                    for w in work_list:
                        cluster[w['cluster']] += w['size']
                    kernel_compute += pa['kernel_compute_time']
                    
                output = [layer_id, layer_name, kernel_name, conv_id, conv_alg, conv_simd]
                output.append(int(np.sum(np.array(cluster) / 8)))
                output.extend(np.array(cluster) / 8)
                output.extend(np.array(core) / 8)
                output.append(kernel_compute / 8)
                writer.writerow(output)
            
#%% 냠냠

path = './linaro_resnet18_8/'
data = divid_log(path)
algs = get_alg(data)
print(algs)

#%%

for idx in range(0, len(algs)):
    objs, kernel_size = load_alg(data, algs[idx])
    k_data, kc_data = extract_feature(kernel_size, objs)
    write(kernel_size, objs, k_data, kc_data, "linaro_resnet18", idx)

# # %%
# print(kernel_size)
# k_data, kc_data = extract_feature(int(kernel_size), objs)
# write(kernel_size, objs, k_data, kc_data, "resnet18", idx)

# layerid, layer_name, kernel_name, is_conv, conv_alg, conv_simd, 
# cluster[0-1]_worksize, core[0-5]_time, kernel_compute_time,

# %%
# plt.figure(figsize=(12,3))

# def get_file_to_inference_times(file):
#     # print(file[1]['compute_inference'])
#     return list(map(lambda x: x['compute_inference'] / 8, file))

# def draw_inference_times(steps):
#     plt.plot(np.arange(len(steps)) / len(steps), steps)

# # 0 ==> Little Core
# # 1 ==> Big Core    
# for idx in range(len(data)):
#     steps = get_file_to_inference_times(data[idx][1:])
#     vis = steps[3:int(len(steps) / 2) + 1]
#     vis.insert(0, steps[2])
#     vis.append(steps[1])
#     draw_inference_times(vis)

# plt.xlabel("Little <---> Big")
# plt.ylabel("Unit : ms")
# plt.legend(list(map(lambda x: x.split(".")[0], f)), loc=(1,-0.1))
# plt.title("64x56x56x64, k: 3x3 (Resnet18 Conv2)")
# plt.show()

# %%
