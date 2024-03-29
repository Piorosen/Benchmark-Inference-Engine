import torch
import torch.onnx 
from onnxruntime.quantization import QuantType, quantize_static, QuantFormat
import onnx_data_reader
import onnx2tflite as o2t
import onnx

#Function to Convert to ONNX 
def convert(model, name, size): 
    dummy_input = torch.randn(1, 3, size, size, requires_grad=True)  
    
    # torch.onnx.export(model,         # model being run 
    #      dummy_input,       # model input (or a tuple for multiple inputs) 
    #      "neural_model/onnx/" + name + ".onnx",       # where to save the model  
    #      export_params=True,  # store the trained parameter weights inside the model file 
    #      opset_version=12,    # the ONNX version to export the model to 
    #      do_constant_folding=True,  # whether to execute constant folding for optimization 
    #      input_names = ['input'],   # the model's input names 
    #      output_names = ['output'], # the model's output names
    # ) 

    model_file = "./neural_model/onnx/" + name + ".onnx"
    model_quant_file = "./neural_model/onnx/" + name + "-qint8.onnx"
    # https://github.com/microsoft/onnxruntime/issues/3130
    dr = onnx_data_reader.DataReader(model_file)
    quantize_static(model_file, model_quant_file, dr, quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)

    o2t.onnx2tflite_fp32(name, size)
    o2t.pb2tflite_int8(name, size)

    print(" ") 
    print('Model has been converted to ONNX') 

models = [ 
    (None, 224, 'test-target'),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True),        224,  "vgg16"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True), 224,  "mobilenet_v2"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True),    224,  "googlenet"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True),      227,  "alexnet"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True),     224,  "resnet18"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True),     224,  "resnet34"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True),     224,  "resnet50"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True),    224,  "resnet101"),
    # (torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True),    224,  "resnet152")
]

# set the model to inference mode 
for model, size, name in models:
    # model.eval()
    convert(model, name, size)
    
# import urllib 
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.ur qlretrieve(url, filename)
