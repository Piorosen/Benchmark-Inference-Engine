$!/bin/bash

mkdir onnx

# MobileNetV2
# wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx -P ./onnx/
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx -P ./onnx/ -O mobilenet-int8.onnx

# VGG16
# wget https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-12.onnx -P ./onnx/
wget https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-12-int8.onnx -P ./onnx/ -O vgg16-int8.onnx

# GoogLeNet
# wget https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx -P ./onnx/
wget https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12-int8.onnx -P ./onnx/ -O googlenet-int8.onnx

# AlexNet
# wget https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx -P ./onnx/
wget https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-12-int8.onnx -P ./onnx/ -O alexnet-int8.onnx
