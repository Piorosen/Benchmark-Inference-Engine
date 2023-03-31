# Benchmark-Inference-Engine
다양한 종류의 딥러닝 인퍼런스 엔진 구현

# 신경망 모델 정보

1. [AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/)
2. [GoogLeNet](https://pytorch.org/hub/pytorch_vision_googlenet/)
3. [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
4. [VGG16](https://pytorch.org/hub/pytorch_vision_vgg/)

# AutoTVM 관련 문서

1. Android 관련 https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_mobile_gpu.html
2. Android의 Tracker 코드 https://github.com/apache/tvm/tree/main/apps/android_rpc
3. TVM 런타임 배포 https://tvm.apache.org/docs/how_to/deploy/android.html#tvm-runtime-for-android-target
4. Linux 보드에서 Tracker https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_rasp.html#sphx-glr-how-to-deploy-models-deploy-model-on-rasp-py

전에 해봤었으니 간단하게 될 듯!

# Auto Scheduler

1. Intro https://tvm.apache.org/2021/03/03/intro-auto-scheduler
2. Paper: https://arxiv.org/abs/2006.06762
3. Tutorial https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html
4. Code : https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_arm.html#sphx-glr-how-to-tune-with-autoscheduler-tune-network-arm-py

# TVM with INT8

1. https://github.com/tigert1998/tvm-models-baseline
2. https://discuss.tvm.apache.org/t/tvm-int8-quantization-slower-than-float-on-arm/13284
3. https://discuss.tvm.apache.org/t/auto-scheduler-seems-slower-on-int8/9585

# NCNN 

독자적으로 ACL을 최적화한 프로젝트 (vcpkg를 이용하여 NCNN 설치가 가능함)

1. https://github.com/Piorosen/Benchmark-Inference-Engine/tree/main/arm64/ncnn
2. https://github.com/Tencent/ncnn
3. - INT8 https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
4. https://openjournals.uwaterloo.ca/index.php/vsl/article/download/1645/2014

# ONNX Runtime

1. https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#graph-optimizations-in-onnx-runtime
2. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

# ArmCL 

1. https://github.com/ARM-software/ComputeLibrary/blob/v22.05/src/cpu/operators/CpuGemmConv2d.h
2. INT8, UINT8, QINT8, QUINT8
3. https://github.com/ARM-software/ComputeLibrary/blob/v22.05/src/cpu/operators/internal/CpuGemmAssemblyDispatch.cpp


# 신경망 관련 정보 구현 
- [x] 1. AutoTVM
- [x] 2. AutoScheduler of TVM 
- [x] 3. NCNN
- [x] 4. ONNX Runtime
- [x] 5. TFLite

# Android 및 보드간 코드 작성
- 1. Android
  - [ ] 1. AutoTVM
  - [ ] 1. Auto Scheduler
  - [ ] 1. NCNN
  - [ ] 1. ONNX Runtime
  - [ ] 1. TFLite
- 2. Linux
  - [ ] 1. AutoTVM
  - [ ] 1. Auto Scheduler
  - [ ] 1. NCNN
  - [ ] 1. ONNX Runtime
  - [ ] 1. TFLite
