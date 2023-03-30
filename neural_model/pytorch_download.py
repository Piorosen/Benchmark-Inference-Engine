import torch
vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
mbn2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
goog = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

# set the model to inference mode 
vgg.eval() 
mbn2.eval() 
goog.eval() 
alex.eval() 

# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

import torch.onnx 

#Function to Convert to ONNX 
def convert(model, name, size): 
    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, size, size, requires_grad=True)  
    
    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         name + ".onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

convert(vgg, "vgg16", 224)
convert(goog, "googlenet", 224)
convert(mbn2, "mobilenet", 224)
convert(alex, "alexnet", 227)
