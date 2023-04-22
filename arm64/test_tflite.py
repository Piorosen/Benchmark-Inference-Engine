# python3 -m pip install tflite-runtime numpy
# https://gist.github.com/ShawnHymel/f7b5014d6b725cb584a1604743e4e878

import tflite_runtime.interpreter as tflite
import numpy as np
import time

interpreter = tflite.Interpreter(model_path="./mobilenet_v2-int8.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allocate tensors
interpreter.allocate_tensors()

# Print the input and output details of the model
# print()
# print("Input details:")
# print(input_details)
# # print(input_details[0]['dtype'])

# print("Output details:")
# print(output_details)
# print()

# Convert features to NumPy array
np_features = np.random.rand(3,224,224).astype(input_details[0]['dtype'])

# Add dimension to input sample (TFLite model expects (# samples, data))
np_features = np.expand_dims(np_features, axis=0)

# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], np_features)

for _ in range(10):
    start = time.time_ns()
    interpreter.invoke()
    end = time.time_ns()
    print((end - start) / 1000.0 / 1000.0)
