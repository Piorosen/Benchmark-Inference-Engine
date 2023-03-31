# pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
import time
import argparse


# https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
m_name = ["alexnet", "googlenet", "mobilenet", "vgg16"]

for item in m_name:
    model_name = item
    model_file = model_name + ".onnx"
    model_opt_file = model_name + "-opt.onnx"
    model_quant_file = model_name + "-quant.onnx"
# https://github.com/microsoft/onnxruntime/issues/3130
    quantized_model = quantize_dynamic(model_file, model_quant_file, weight_type=QuantType.QUInt8)

    sess_options = ort.SessionOptions()
# sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = model_opt_file
    ort_sess = ort.InferenceSession(model_file, sess_options, providers=["CPUExecutionProvider"])

exit(0)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.randn(1, 3, 224, 224, requires_grad=True)

ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(x)}

time_list = []
repeat = 1000
for idx in range(0, repeat):
    start = time.perf_counter_ns()
    output = ort_sess.run(None, ort_inputs)
    end = time.perf_counter_ns()
    time_list.append(end - start)
    print("[ " + str(idx + 1) + " / "+ str(repeat) + " ]\t" + str((end - start) / 1000 / 1000) + " ms");

print("avg : " + str(sum(time_list) / repeat / 1000 / 1000) + " ms")
