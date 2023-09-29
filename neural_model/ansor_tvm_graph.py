#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import hashlib as hs

#%% Auto Scheduler

def create(model):
    result = []
    with open(model, 'r') as file:
        # 파일의 끝까지 한 줄씩 읽어옵니다.
        for line in file:
            data = line.strip()
            d = json.loads(data)
            result.append(d)
    return result

# %%
def hash(item):
    b = str(item['input']).encode('utf-8')
    a = hs.md5(b).hexdigest()
    return a

# %%
# class MeasureErrorNo(enum.IntEnum):
#     """Error type for MeasureResult"""
#     NO_ERROR = 0  # no error
#     INSTANTIATION_ERROR = 1  # actively detected error in instantiating a template with a config
#     COMPILE_HOST = 2  # error when compiling code on host (e.g. tvm.build)
#     COMPILE_DEVICE = 3  # error when compiling code on device (e.g. OpenCL JIT on the device)
#     RUNTIME_DEVICE = 4  # error when run program on device
#     WRONG_ANSWER = 5  # answer is wrong when compared to a golden output
#     BUILD_TIMEOUT = 6  # timeout during compilation
#     RUN_TIMEOUT = 7  # timeout during run
#     UNKNOWN_ERROR = 8  # unknown error

def convert(result):
    data = {}

    # idx = 0 
    for i in result:
        if i["result"][1] != 0:
            continue
        h = hash(i)
        if h in data:
            data[h]["data"].append(i)
        else:
            data[h] = {"id": len(data), "data":[i]}
    return data

# %%
def default_perform(data):
    result = 0
    for i in data.keys():
        idx = 0
        result += sum(data[i]["data"][idx]['result'][0])
        # print(data[i]["data"][idx]['result'])
    return result * 1000

#%%



# %%

def dict_to_array(data):
    list = [0 for _ in range(len(data))]
    for i in data.keys():
        list[data[i]["id"]] = data[i]["data"]
    return list
    # print(list)

def get_min_indx(data, idx):
    i = 0
    cur = 1000000000000000
    for m_idx in range(len(data[idx])):
        d = sum(data[idx][m_idx]["result"][0])
        if cur > d:
            i = m_idx
            cur = d
    return i

def get_value(data, idxes):
    result = 0
    for i in range(len(data)):
        result += sum(data[i][idxes[i]]["result"][0])
    return result * 1000

def convert_perform(data):
    list = [0 for _ in range(len(data))]
    result = []

    for layer_idx in range(len(data)):
        for step_idx in range(len(data[layer_idx])):
            list[layer_idx] = step_idx 
            c = get_value(data, list)
            if len(result) == 0:
                result.append(c)
            else:
                lat = min(result)
                result.append(min([lat, c]))
        list[layer_idx] = get_min_indx(data, layer_idx)

    return result



# %%
def get_autotvm_log(file_name, vary):
    data = create(file_name)
    data = convert(data)
    print(default_perform(data))
    p = dict_to_array(data)
    b = convert_perform(p)
    b = np.array(b) / 4
    other_com = vary - np.min(b) 
    b += other_com
    return b
# %%

# %%
