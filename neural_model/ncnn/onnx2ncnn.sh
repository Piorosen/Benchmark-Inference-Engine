./onnx2ncnn.exe ./onnx/alexnet.onnx ./param/alexnet.param ./models/alexnet.bin
./ncnnoptimize.exe ./param/alexnet.param ./models/alexnet.bin ./param/alexnet-opt.param ./models/alexnet-opt.bin 65536
./ncnn2table.exe ./param/alexnet-opt.param ./models/alexnet-opt.bin list.txt ./table/alexnet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/alexnet-opt.param ./models/alexnet-opt.bin ./param/alexnet-int8.param ./models/alexnet-int8.bin ./table/alexnet.table
./onnx2ncnn.exe ./onnx/vgg16.onnx ./param/vgg16.param ./models/vgg16.bin
./ncnnoptimize.exe ./param/vgg16.param ./models/vgg16.bin ./param/vgg16-opt.param ./models/vgg16-opt.bin 65536
./ncnn2table.exe ./param/vgg16-opt.param ./models/vgg16-opt.bin list.txt ./table/vgg16.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/vgg16-opt.param ./models/vgg16-opt.bin ./param/vgg16-int8.param ./models/vgg16-int8.bin ./table/vgg16.table
./onnx2ncnn.exe ./onnx/mobilenet_v2.onnx ./param/mobilenet_v2.param ./models/mobilenet_v2.bin
./ncnnoptimize.exe ./param/mobilenet_v2.param ./models/mobilenet_v2.bin ./param/mobilenet_v2-opt.param ./models/mobilenet_v2-opt.bin 65536
./ncnn2table.exe ./param/mobilenet_v2-opt.param ./models/mobilenet_v2-opt.bin list.txt ./table/mobilenet_v2.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/mobilenet_v2-opt.param ./models/mobilenet_v2-opt.bin ./param/mobilenet_v2-int8.param ./models/mobilenet_v2-int8.bin ./table/mobilenet_v2.table
./onnx2ncnn.exe ./onnx/googlenet.onnx ./param/googlenet.param ./models/googlenet.bin
./ncnnoptimize.exe ./param/googlenet.param ./models/googlenet.bin ./param/googlenet-opt.param ./models/googlenet-opt.bin 65536
./ncnn2table.exe ./param/googlenet-opt.param ./models/googlenet-opt.bin list.txt ./table/googlenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/googlenet-opt.param ./models/googlenet-opt.bin ./param/googlenet-int8.param ./models/googlenet-int8.bin ./table/googlenet.table
./onnx2ncnn.exe ./onnx/resnet18.onnx ./param/resnet18.param ./models/resnet18.bin
./ncnnoptimize.exe ./param/resnet18.param ./models/resnet18.bin ./param/resnet18-opt.param ./models/resnet18-opt.bin 65536
./ncnn2table.exe ./param/resnet18-opt.param ./models/resnet18-opt.bin list.txt ./table/resnet18.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/resnet18-opt.param ./models/resnet18-opt.bin ./param/resnet18-int8.param ./models/resnet18-int8.bin ./table/resnet18.table
./onnx2ncnn.exe ./onnx/resnet50.onnx ./param/resnet50.param ./models/resnet50.bin
./ncnnoptimize.exe ./param/resnet50.param ./models/resnet50.bin ./param/resnet50-opt.param ./models/resnet50-opt.bin 65536
./ncnn2table.exe ./param/resnet50-opt.param ./models/resnet50-opt.bin list.txt ./table/resnet50.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/resnet50-opt.param ./models/resnet50-opt.bin ./param/resnet50-int8.param ./models/resnet50-int8.bin ./table/resnet50.table
./onnx2ncnn.exe ./onnx/resnet101.onnx ./param/resnet101.param ./models/resnet101.bin
./ncnnoptimize.exe ./param/resnet101.param ./models/resnet101.bin ./param/resnet101-opt.param ./models/resnet101-opt.bin 65536
./ncnn2table.exe ./param/resnet101-opt.param ./models/resnet101-opt.bin list.txt ./table/resnet101.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[227,227,3] pixel=BGR thread=8 method=kl
./ncnn2int8.exe ./param/resnet101-opt.param ./models/resnet101-opt.bin ./param/resnet101-int8.param ./models/resnet101-int8.bin ./table/resnet101.table