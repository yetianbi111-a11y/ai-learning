import torch

print("PyTorch版本:")
print(torch.__version__)

print("\nCUDA是否可用:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("\nGPU型号:")
    print(torch.cuda.get_device_name(0))