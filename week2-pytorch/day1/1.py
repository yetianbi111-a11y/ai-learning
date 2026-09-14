import torch
"""
tensor1 = torch.Tensor(3, 2)
print(tensor1)
print(tensor1.shape)
print(tensor1.dtype)

tensor2 = torch.IntTensor([[1,2],[3,4]])
print(tensor2)
print(tensor2.shape)
print(tensor2.dtype)


"""
"""
#按区间
tensor3 = torch.arange(0, 10) #arrange(start, end, step)
tensor4 = torch.linspace(0, 10, 5) #linspace(start, end, ammount)
print(tensor3)
print(tensor4)
"""

#按值填充
tensor5 = torch.zeros(3,2,4) #zeros(size)
print(tensor5)