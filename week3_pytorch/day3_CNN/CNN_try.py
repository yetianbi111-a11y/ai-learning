#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
#shuru: (N, C_in, H_in, W_in) -> output: (N, C_out, H_out, W_out) padding:填充幅度
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread("week3_pytorch/day3_CNN/duck.png")
print(img.shape)

input = torch.tensor(img).permute(2,0,1).float() # (C,H,W) -> (1,C,H,W)

conv = nn.Conv2d(in_channels=4,out_channels=3,kernel_size=9,stride=3,padding=0,bias=False)

output = conv(input)
print(output.shape)

output_img = torch.clamp(output.int(),0,255).permute(1,2,0).detach().numpy()# (N,C,H,W) -> (H,W,C)
print(output_img.shape)

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(output_img)
plt.show() 



