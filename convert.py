import torch

# Load your PyTorch model
model = torch.load("model_v2.pth", map_location=torch.device('cpu'))
model.eval() 

import torch.onnx

dummy_input = torch.randn(1, 784)  # 1D input with 784 features
onnx_path = "model.onnx"

torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, 
                  input_names=["input"], output_names=["output"])