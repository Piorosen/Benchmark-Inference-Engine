#!/bin/bash
# https://velog.io/@kcw4875/Pytorch%EC%97%90%EC%84%9C-TFLite%EB%A1%9C-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0


python3 -m mo --input_model ./onnx/alexnet.onnx ##openvino 파일 생성(xml, bin, mapping)
python3 -m mo --input_model ./onnx/googlenet.onnx ##openvino 파일 생성(xml, bin, mapping)
python3 -m mo --input_model ./onnx/mobilenet.onnx ##openvino 파일 생성(xml, bin, mapping)
python3 -m mo --input_model ./onnx/vgg16.onnx ##openvino 파일 생성(xml, bin, mapping)
