# pip install tensorflow onnx onnx-tf uvicorn tensorflow-probability onnx2keras


# import onnx
# from onnx_tf.backend import prepare
import os
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

size = 224

# Load the ONNX model   
def representative_dataset_gen():
    global size
    for _ in range(1):
      data = np.random.rand(1, 3, size, size)
      yield [data.astype(np.float32)]
    

def onnx2tflite_fp32(model_name, model_size):
    global size 
    size = model_size

    model = onnx.load("./neural_model/onnx/" + model_name + ".onnx")
    tf_rep = prepare(model)
    tf_rep.export_graph("./neural_model/tf/" + model_name)

    converter = tf.lite.TFLiteConverter.from_saved_model("./neural_model/tf/" + model_name)
    tflite_model = converter.convert()

    # Save the model.
    with open("./neural_model/tflite/" + model_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

def pb2tflite_int8(model_name, model_size):
    # https://github.com/sithu31296/PyTorch-ONNX-TFLite
    global size 
    size = model_size
    converter = tf.lite.TFLiteConverter.from_saved_model("./neural_model/tf/" + model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open("./neural_model/tflite/" +model_name + "-int8.tflite", 'wb') as f:
        f.write(tflite_model)
