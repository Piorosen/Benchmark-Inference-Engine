import onnxruntime as ort
import numpy as np
import time

x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')

time_list = []
for idx in range(0, 50):
    start = time.perf_counter_ns()
    outputs = ort_sess.run(None, {'input': x.numpy()})
    end = time.perf_counter_ns()
    time_list.append(end - start)
    print("[ " + str(idx + 1) + " / 50 ]\t" + (end - start) / 1000 / 1000 + " ms");

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')