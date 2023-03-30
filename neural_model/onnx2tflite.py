# pip install tensorflow onnx onnx-tf uvicorn

import onnx
from onnx_tf.backend import prepare
import os

# Load the ONNX model

print(os.getcwd())


models = ["alexnet", "googlenet", "mobilenet", "vgg16"]
for model_name in models:
    model = onnx.load(".\\neural_model\\onnx\\" + model_name + ".onnx")
    # Check that the IR is well formed  
    onnx.checker.check_model(model)
    # Print a Human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    tf_rep = prepare(model)
    tf_rep.export_graph(".\\neural_model\\tf\\" + model_name + ".pb")

