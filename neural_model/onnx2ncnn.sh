$!/bin/bash

cd ncnn

# python3 -m onnxsim ../onnx/alexnet.onnx ./alexnet.onnx
# python3 -m onnxsim ../onnx/vgg16.onnx ./vgg16.onnx
# python3 -m onnxsim ../onnx/googlenet.onnx ./googlenet.onnx
# python3 -m onnxsim ../onnx/mobilenet.onnx ./mobilenet.onnx

# cp ../onnx/bvlcalexnet-12.onnx ./alexnet.onnx
# cp ../onnx/vgg16-12.onnx ./vgg16.onnx
# cp ../onnx/googlenet-12.onnx ./googlenet.onnx
# cp ../onnx/mobilenetv2-12.onnx ./mobilenet.onnx

cp ../onnx/alexnet.onnx ./alexnet.onnx
cp ../onnx/vgg16.onnx ./vgg16.onnx
cp ../onnx/googlenet.onnx ./googlenet.onnx
cp ../onnx/mobilenet.onnx ./mobilenet.onnx

./onnx2ncnn.exe ./alexnet.onnx ./alexnet.param ./alexnet.bin
./onnx2ncnn.exe ./vgg16.onnx ./vgg16.param ./vgg16.bin
./onnx2ncnn.exe ./mobilenet.onnx ./mobilenet.param ./mobilenet.bin
./onnx2ncnn.exe ./googlenet.onnx ./googlenet.param ./googlenet.bin

.\ncnnoptimize.exe .\alexnet.param .\alexnet.bin .\alexnet-opt.param .\alexnet-opt.bin 65536
.\ncnnoptimize.exe .\vgg16.param .\vgg16.bin .\vgg16-opt.param .\vgg16-opt.bin 65536
.\ncnnoptimize.exe .\mobilenet.param .\mobilenet.bin .\mobilenet-opt.param .\mobilenet-opt.bin 65536
.\ncnnoptimize.exe .\googlenet.param .\googlenet.bin .\googlenet-opt.param .\googlenet-opt.bin 65536

.\ncnn2table.exe alexnet-opt.param alexnet-opt.bin imagelist.txt alexnet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe vgg16-opt.param vgg16-opt.bin imagelist.txt vgg16.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
.\ncnn2table.exe googlenet-opt.param googlenet-opt.bin imagelist.txt googlenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl

.\ncnn2int8.exe alexnet-opt.param alexnet-opt.bin alexnet-int8.param alexnet-int8.bin alexnet.table
.\ncnn2int8.exe vgg16-opt.param vgg16-opt.bin vgg16-int8.param vgg16-int8.bin vgg16.table
.\ncnn2int8.exe mobilenet-opt.param mobilenet-opt.bin mobilenet-int8.param mobilenet-int8.bin mobilenet.table
.\ncnn2int8.exe googlenet-opt.param googlenet-opt.bin googlenet-int8.param googlenet-int8.bin googlenet.table



cd ..
