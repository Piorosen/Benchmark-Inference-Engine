import torch
import torch.onnx 

#Function to Convert to ONNX 
def convert(model, name, size): 
    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, size, size, requires_grad=True)  
    
    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "neural_model/onnx/" + name + ".onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names
    ) 
        #  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
        #                         'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

models = [ 
    (torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True),        224,  "vgg16"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True), 224,  "mobilenet_v2"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True),    224,  "googlenet"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True),      227,  "alexnet"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True),     224,  "resnet18"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True),     224,  "resnet34"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True),     224,  "resnet50"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True),    224,  "resnet101"),
    (torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True),    224,  "resnet152")
]

# set the model to inference mode 
for model, size, name in models:
    model.eval()
    convert(model, name, size)
    
# import urllib 
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.ur qlretrieve(url, filename)
