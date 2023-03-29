# pip3 install --user psutil xgboost cloudpickle

import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)