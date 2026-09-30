import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 1. 定义一个普通卷积层，用于学习偏移量 (offsets)
        # 输出通道数为 2 * kernel_size * kernel_size，因为每个采样点需要 (x, y) 两个偏移值
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # 2. 定义可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, x):
        # 前向传播时，先通过 offset_conv 生成偏移量
        offsets = self.offset_conv(x)
        # 然后将输入和偏移量一起传入 deform_conv 进行计算
        output = self.deform_conv(x, offsets)
        return output

# --- 使用示例 ---
# 创建一个输入张量 (batch_size=1, channels=3, height=64, width=64)
input_tensor = torch.randn(1, 3, 64, 64)

# 实例化可变形卷积模块
dcn = DeformableConv2d(in_channels=3, out_channels=16, kernel_size=3)

# 执行前向传播
output_tensor = dcn(input_tensor)

print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")