# pip install tensorflow onnx onnx-tf uvicorn tensorflow-probability onnx2keras


# import onnx
# from onnx_tf.backend import prepare
import os
import numpy as np
import tensorflow as tf

# Load the ONNX model   

print(os.getcwd())

def representative_dataset_gen():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]
      
# models = ["vgg16",  "mobilenet", "googlenet"]
models = ["mobilenet"]
for model_name in models:
    # model = onnx.load(".\\neural_model\\onnx\\" + model_name + ".onnx")
    # # Check that the IR is well formed  
    # onnx.checker.check_model(model)
    # # Print a Human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)
    # tf_rep = prepare(model)
    # tf_rep.export_graph(".\\neural_model\\tf\\" + model_name + ".pb")
    # Convert the model
    
    converter = tf.lite.TFLiteConverter.from_saved_model(".\\neural_model\\tf\\" + model_name + ".pb")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
