import torch
import numpy as np
"""
#张量类型修改
tensor1 = torch.randint(0, 10, (3,4))
print(tensor1,tensor1.type())
tensor1 = tensor1.type(torch.float32)
print(tensor1,tensor1.type())

#torch.numpy()张量转numpy
tensor2 = torch.randint(0, 10, (3,4))
numpy_array = tensor2.numpy()
print(numpy_array, type(numpy_array))
"""

"""张量计算
tensor1.add()#不改变元素
tensor1.add_()#改变元素
tensor1.sub()
tensor1.sub_()
tensor1.pow()
tensor1.pow_()
"""

#hadamard积
tensor3 = torch.tensor([[1,2],[3,4]])
tensor4 = torch.tensor([[5,6],[7,8]])
print(tensor3 * tensor4) #对应元素相乘
print(tensor3.mul(tensor4)) 
print(tensor3.matmul(tensor4)) #矩阵乘法 tensor3.mm(tensor4) #矩阵乘法,tensor3 @ tensor4 #矩阵乘法