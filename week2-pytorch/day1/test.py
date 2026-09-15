import torch

"""
#固定随机数种子
torch.manual_seed(1)
tensor1 = torch.randint(0, 10, (3, 4))
print(tensor1)
print(tensor1.shape)
print(tensor1.dtype)
print(tensor1.ndim)
"""
"""
#reshapre
tensor2 =torch.arange(24)
print(tensor2)
tensor3 = tensor2.reshape(2, 3, 4)
print(tensor3)
"""
"""
#增加维度
tensor4 =torch.rand([3,224,244])
print(tensor4.shape)
print(tensor4.ndim)
print(tensor4.unsqueeze(dim=0).shape)
"""
torch.manual_seed(1)
tensor5 = torch.randint(0, 10, (3, 3))
tensor6 = torch.randint(0, 10, (3, 3))
print(tensor5) 
print(tensor6)
print(tensor5 +tensor6) #对应元素相加
print(tensor5*tensor6) #对应元素相乘
print(torch.matmul(tensor5,tensor6)) #矩阵乘法