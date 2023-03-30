$!/bin/bash

python -m pip install onnxsim   
cp ./onnx ./ncnn

mkdir alexnet
mkdir vgg16
mkdir mobilenet
mkdir googlenet

python3 -m onnxsim ../onnx/alexnet.onnx ./alexnet/alexnet.onnx
python3 -m onnxsim ../onnx/vgg16.onnx ./vgg16/vgg16.onnx
python3 -m onnxsim ../onnx/googlenet.onnx ./googlenet/googlenet.onnx
python3 -m onnxsim ../onnx/mobilenet.onnx ./mobilenet/mobilenet.onnx

./onnx2ncnn ./alexnet.onnx ./alexnet/alexnet.param ./alexnet/alexnet.bin
./onnx2ncnn ./vgg16.onnx ./vgg16/vgg16.param ./vgg16/vgg16.bin
./onnx2ncnn ./mobilenet.onnx ./mobilenet/mobilenet.param ./mobilenet/mobilenet.bin
./onnx2ncnn ./googlenet.onnx ./googlenet/googlenet.param ./googlenet/googlenet.bin

