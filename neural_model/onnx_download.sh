$!/bin/bash

mkdir neural_model
mkdir neural_model/onnx

# MobileNetV2
wget https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx -P ./neural_model/onnx/
wget https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx-P ./neural_model/onnx/

# VGG16
wget https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12.onnx -P ./neural_model/onnx/
wget https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12-int8.onnx -P ./neural_model/onnx/

# GoogLeNet
wget https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx -P ./neural_model/onnx/
wget https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12-int8.onnx -P ./neural_model/onnx/

# AlexNet
wget https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx -P ./neural_model/onnx/
wget https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12-int8.onnx -P ./neural_model/onnx/
