$!/bin/bash

python -m pip install onnxsim   
cp ./onnx ./ncnn

mkdir alexnet
mkdir vgg16
mkdir mobilenet
mkdir googlenet

python3 -m onnxsim ../onnx/alexnet.onnx ./alexnet.onnx
python3 -m onnxsim ../onnx/vgg16.onnx ./vgg16.onnx
python3 -m onnxsim ../onnx/googlenet.onnx ./googlenet.onnx
python3 -m onnxsim ../onnx/mobilenet.onnx ./mobilenet.onnx

./onnx2ncnn ./alexnet.onnx ./alexnet.param ./alexnet.bin
./onnx2ncnn ./vgg16.onnx ./vgg16.param ./vgg16.bin
./onnx2ncnn ./mobilenet.onnx ./mobilenet.param ./mobilenet.bin
./onnx2ncnn ./googlenet.onnx ./googlenet.param ./googlenet.bin

.\ncnnoptimize.exe .\alexnet.param .\alexnet.bin .\alexnet-opt.param .\alexnet-opt.bin 0
.\ncnnoptimize.exe .\vgg16.param .\vgg16.bin .\vgg16-opt.param .\vgg16-opt.bin 0
.\ncnnoptimize.exe .\mobilenet.param .\mobilenet.bin .\mobilenet-opt.param .\mobilenet-opt.bin 0
.\ncnnoptimize.exe .\googlenet.param .\googlenet.bin .\googlenet-opt.param .\googlenet-opt.bin 0

.\ncnn2table.exe alexnet-opt.param alexnet-opt.bin imagelist.txt alexnet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe vgg16.param vgg16.bin imagelist.txt vgg16.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe mobilenet.param mobilenet.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe googlenet.param googlenet.bin imagelist.txt googlenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
