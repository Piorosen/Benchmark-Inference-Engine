$!/bin/bash

vcpkg install ncnn argparse
vcpkg integrate install

pip install onnxruntime onnx

wget https://github.com/DDGRCF/sparseinst_ncnn_demo/releases/download/v1.0.0/sparseinst-resnet-sim-opt.bin
wget https://github.com/DDGRCF/sparseinst_ncnn_demo/releases/download/v1.0.0/sparseinst-resnet-sim-opt.param