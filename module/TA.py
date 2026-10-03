import torch
from torch import nn

class Tlayer(nn.Module):
    '''
    Temporal-wise Attention Layer
    '''

    def __init__(self, timeWindows, reduction=5, dimension=3):
        super(Tlayer, self).__init__()
        if dimension == 3:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.temporal_excitation = nn.Sequential(
            nn.Linear(timeWindows, int(timeWindows // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(timeWindows // reduction), timeWindows),
            nn.Sigmoid()
        )

    def forward(self, input):
        t, b, c, h, w = input.size()  # 获取输入的尺寸

        temp = self.avg_pool(input.view(b * t, c, h, w))  # 适当展平输入
        # print(temp.shape) 
        temp = temp.view(b, t, -1).mean(dim=2)  # 在通道维度取平均
        # print(temp.shape)  # 输出形状用于调试
        
        y = self.temporal_excitation(temp)  # 传入时间注意力网
        y = y.view(t, b, 1, 1, 1)  # 适当调整形状以便于乘法
        # print(input.shape)
        y = torch.mul(input, y)  # 逐元素相乘
        # print(y.shape)

        return y
