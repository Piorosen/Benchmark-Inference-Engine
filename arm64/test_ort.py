# pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions
import onnxruntime as ort
import time
import numpy as np
import torch

# https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
# https://github.com/microsoft/onnxruntime/issues/3130

def inference(model_name):
    model_file = model_name + ".onnx"
    # model_opt_file = model_name + "-opt.onnx"

    sess_options = ort.SessionOptions()
    # sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # sess_options.optimized_model_filepath = model_opt_file
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(model_file, sess_options, providers=["CPUExecutionProvider"])
    print()

    x = np.random.rand(ort_sess.get_inputs()[0].shape[0], 
                    ort_sess.get_inputs()[0].shape[1], 
                    ort_sess.get_inputs()[0].shape[2], 
                    ort_sess.get_inputs()[0].shape[3]).astype(np.float32)

    ort_inputs = {ort_sess.get_inputs()[0].name: x}

    repeat = 10
    for idx in range(0, repeat):
        start = time.perf_counter_ns()
        # output = ort_sess.run("output", ort_inputs)
        ort_sess.run(None, ort_inputs)
        end = time.perf_counter_ns()
        print("[ " + str(idx + 1) + " / "+ str(repeat) + " ]\t" + str((end - start) / 1000 / 1000) + " ms");


if __name__ == "__main__":
    inference("./resnet101")