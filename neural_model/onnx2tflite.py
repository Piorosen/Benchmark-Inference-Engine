# pip install tensorflow onnx onnx-tf uvicorn tensorflow-probability onnx2keras


# import onnx
# from onnx_tf.backend import prepare
import os
import numpy as np
import tensorflow.compat.v1 as tf
import onnx

from onnx_tf.backend import prepare
# from keras_cv_attention_models import model_surgery
from tensorflow.keras.preprocessing import image

# Load the ONNX model   
def representative_dataset_gen():
    for _ in range(1):
      data = np.random.rand(1, 3, 224, 224)
      yield [data.astype(np.float32)]
    

def onnx2tflite_fp32(model_name):
    model = onnx.load("./neural_model/onnx/" + model_name + ".onnx")
    # Check that the IR is well formed  
    # onnx.checker.check_model(model)
    # Print a Human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)
    tf_rep = prepare(model)
    tf_rep.export_graph("./neural_model/tf/" + model_name)
    
    # Convert the model
    input_nodes = tf_rep.inputs
    output_nodes = tf_rep.outputs

    converter = tf.lite.TFLiteConverter.from_saved_model("./neural_model/tf/" + model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model.
    with open("./neural_model/tflite/" +model_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

def pb2tflite_int8(model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model("./neural_model/tf/" + model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # converter.target_spec.supported_types = [tf.int8]
    tflite_model = converter.convert()
    with open("./neural_model/tflite/" +model_name + "-uint8.tflite", 'wb') as f:
        f.write(tflite_model)
