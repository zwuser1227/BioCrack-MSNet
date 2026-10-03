import torch
import torch.nn as nn
from module.M_LIF import *
from module.LIF import *
from module.TA import *
import torch
import torch.nn as nn

class StackedSeparableConv(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers=3, kernel_size=3, stride=1, padding=1):
        super(StackedSeparableConv, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DepthwiseSeparableConv(in_channels, hidden_size, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(hidden_size))  # 批归一化
            layers.append(nn.ReLU(inplace=True))  # 非线性激活函数
            in_channels = hidden_size  # 更新输入通道大小
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 深度可分离卷积定义
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义 SNN 分割网络
class SNNCrackSegmentation(nn.Module):
    def __init__(self, input_channels=3, hidden_size=16, num_classes=1, time_windows=5):
        super(SNNCrackSegmentation, self).__init__()
        self.time_windows = time_windows

        # 卷积特征提取
        # self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)  # 3-16
        # self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)  # 16-32
        self.conv1 = DepthwiseSeparableConv(input_channels, hidden_size)  # 堆叠 3 层
        self.conv2 = DepthwiseSeparableConv(hidden_size, hidden_size)  # 堆叠 3 层
        # self.conv3 = nn.Conv2d(hidden_size, hidden_size, dilation=1, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Temporal-wise attention layer
        # self.tlayer = Tlayer(time_windows, reduction=5, dimension=4)  # 启用时间维度注意力层
        # 使用 ConvAttLIF 处理空间特征的脉冲神经元层
        self.att_lif_layer = ConvAttLIF(inputSize=hidden_size,
                                        hiddenSize=hidden_size,
                                        kernel_size=3,
                                        spikeActFun=nn.Sigmoid(),
                                        padding=1)
        self.conv = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=1)

        # 解码层：将特征图恢复到原图尺寸
        # self.final_conv = nn.Conv2d(hidden_size*2, hidden_size, kernel_size=1)

    def forward(self, x):
        # 假设输入 x 的形状为 [batch_size, time_windows, channels, height, width]
        batch_size, time_steps, channels, height, width = x.size()
        # 初始化一个列表来存储每个时间步的输出
        outputs = []
        outputs_no_weight = []
        for step in range(time_steps):
            # 提取当前时间步的输入
            x_step = x[:, step, :, :, :]  # 形状: [batch_size, channels, height, width]
            # 卷积层特征提取
            x_step1 =torch.relu(self.conv1(x_step))
            x_step2 = self.pool(torch.relu(self.conv2(x_step1)))

            x_noweight = x_step2

            x_weight = x_step2.sigmoid().mean()
            # print(x_weight.shape)
            x_step2 = x_step2*x_weight

            outputs.append(x_step2)  # 存储每个时间步的输出
            outputs_no_weight.append(x_noweight)  # 存储每个时间步的输出
            # outputs.append(x_step2)  # 存储每个时间步的输出

        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=0)  # 根据需要调整 dim
        if isinstance(outputs_no_weight, list):
            outputs_no_weight = torch.stack(outputs_no_weight, dim=0)  # 根据需要调整 dim
        # outputs = self.tlayer(outputs)  # 确保这里不需要额外增加维度
        # 脉冲神经元层
        outputs = self.att_lif_layer(outputs)
        outputs = outputs.permute(1, 0, 2, 3, 4)
        outputs = self.conv(outputs)
        # print(outputs.shape) 
        outputs_o = outputs.squeeze(0)
        # outputs_o = outputs.mean(dim=0)  # 在时间步上取平均，结合所有时间步的结果
        # print(outputs_o.shape) 
        outputs = outputs.view(outputs.size(1), outputs.size(0), outputs.size(2), outputs.size(3), outputs.size(4))
        # print(outputs.shape) 
        # 解码过程：恢复到原始图像尺寸
        # 堆叠时间步的输出
        # outputs_no_weight = torch.tensor(outputs_no_weight)
        # print(type(outputs_no_weight))
        # b, t, c, h, w = outputs_no_weight.size()
        # outputs_no_weight = outputs_no_weight.reshape(b * t, c, h, w)
        # print(outputs_no_weight.shape)  # 检查每个元素的类型

        # outputs = self.final_conv(outputs)
        outputs_no_weight = outputs_no_weight.view(x.size(0), x.size(1), outputs.size(2), outputs.size(3), outputs.size(4))
        return outputs+outputs_no_weight, outputs_o+x_step2

class SNN_C_encode(nn.Module):
    def __init__(self, input_channels=3, hidden_size=16, num_classes=1, time_windows=5):
        super(SNN_C_encode, self).__init__()
        self.SNNC1 = SNNCrackSegmentation(input_channels, hidden_size, time_windows)
        self.SNNC2 = SNNCrackSegmentation(hidden_size, hidden_size*2, time_windows)
        self.SNNC3 = SNNCrackSegmentation(hidden_size*2, hidden_size*4, time_windows)
        self.SNNC4 = SNNCrackSegmentation(hidden_size*4, hidden_size*8, time_windows)
    def forward(self, x):
        SNNC1, SNNC1_o = self.SNNC1(x)
        SNNC2, SNNC2_o = self.SNNC2(SNNC1)
        SNNC3, SNNC3_o = self.SNNC3(SNNC2)
        SNNC4, SNNC4_o = self.SNNC4(SNNC3)
        return SNNC1_o, SNNC2_o, SNNC3_o, SNNC4_o
class CFattention2(nn.Module):
    def __init__(self, in_channels, out_channels, filter_ratio=0.1):
        super(CFattention2, self).__init__()
        # 空洞卷积+上采样分支
        self.dconv1 = nn.Conv2d(in_channels[1], out_channels,3,1,1)
        self.upconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        
        # 空洞卷积分支（保持分辨率）
        self.dconv2 = nn.Conv2d(in_channels[0], out_channels,3,1,1)
        
        # 频域注意力生成模块
        self.freq_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 高频滤波器参数
        # self.filter_ratio = filter_ratio
          # 可学习的频率阈值参数
        self.threshold_h = nn.Parameter(torch.tensor(0.1))
        self.threshold_w = nn.Parameter(torch.tensor(0.1))
        # self.skip = nn.Conv2d(out_channels , out_channels, kernel_size=3, padding=1)

    def frequency_domain_attention(self, x):
        # FFT转换
        x_fft = torch.fft.rfft2(x, norm='backward')
        
        # 创建高频掩码
        _, _, h, w = x_fft.shape
        mask = torch.ones_like(x_fft)
        
        # 计算中心区域边界
        cutoff_h = int(h * self.threshold_h)
        cutoff_w = int(w * self.threshold_w)
        
        # 中心区域置零（保留高频）
        mask[..., h//2-cutoff_h:h//2+cutoff_h, w//2-cutoff_w:w//2+cutoff_w] = 0
        
        # 应用高通滤波
        high_freq = x_fft * mask
        
        # 逆变换回空间域
        x_high = torch.fft.irfft2(high_freq, s=x.shape[-2:], norm='backward')
        
        # 生成注意力权重
        return self.freq_conv(x_high)

    def forward(self, *inputs):
        x1, x2 = inputs
        # 分支1处理：空洞卷积+上采样
        x2 = self.dconv1(x2)
        x2_up = self.upconv(x2)
        
        # 分支2处理：空洞卷积保持分辨率
        x1 = self.dconv2(x1)
        
        # 从x1提取高频注意力权重
        attention = self.frequency_domain_attention(x2_up)
        
        # 注意力加权融合
        x1_att = x1 * attention
        return x2_up + x1_att
        # return x1_att


import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFrequencyFilter(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFrequencyFilter, self).__init__()
        # 可学习的频率阈值参数
        self.threshold_h = nn.Parameter(torch.tensor(0.1))
        self.threshold_w = nn.Parameter(torch.tensor(0.1))
        
        # 可学习的频率增强参数
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        # FFT转换
        x_fft = torch.fft.rfft2(x, norm='backward')
        b, c, h, w = x_fft.shape
        
        # 使用sigmoid确保阈值在0-0.5之间
        ratio_h = 0.5 * torch.sigmoid(self.threshold_h)
        ratio_w = 0.5 * torch.sigmoid(self.threshold_w)
        
        # 计算截止频率
        cutoff_h = int(h * ratio_h)
        cutoff_w = int(w * ratio_w)
        
        # 创建可学习的频率掩码
        mask = torch.ones_like(x_fft)
        if cutoff_h > 0 and cutoff_w > 0:
            mask[..., h//2-cutoff_h:h//2+cutoff_h, w//2-cutoff_w:w//2+cutoff_w] = 0
        
        # 应用可学习的高通滤波
        high_freq = x_fft * mask * self.alpha + x_fft * self.beta
        
        # 逆变换回空间域
        x_high = torch.fft.irfft2(high_freq, s=x.shape[-2:], norm='backward')
        
        return x_high

class CFattention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFattention, self).__init__()
        
        # 分支1：空洞卷积+上采样
        self.dconv1 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        
        # 分支2：空洞卷积保持分辨率
        self.dconv2 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应频率滤波
        self.adaptive_filter = AdaptiveFrequencyFilter(out_channels)
        
        # 频域注意力路径
        self.freq_path = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力路径
        self.spatial_path = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力路径
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 输出归一化
        self.final_norm = nn.BatchNorm2d(out_channels)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # 分支1处理
        x2 = self.dconv1(x2)
        x2_up = self.upconv(x2)
        
        # 分支2处理
        x1 = self.dconv2(x1)
        
        # 自适应频率滤波
        x_freq = self.adaptive_filter(x2_up)
        
        # 频域注意力
        freq_att = self.freq_path(x_freq)
        
        # 空间注意力
        spatial_att = self.spatial_path(x2_up)
        
        # 通道注意力
        channel_att = self.channel_path(x2_up)
        
        # 组合注意力
        combined_att = freq_att * spatial_att * channel_att
        
        # 注意力加权融合 + 残差连接
        x1_att = x1 * combined_att + x1
        
        # 特征融合
        out = x2_up + x1_att
        
        return self.final_norm(out)

class SNN_C_Decode(nn.Module):
    def __init__(self,  hidden_size=32, num_classes=1):
        super(SNN_C_Decode, self).__init__()
        self.CFattention1 = CFattention((hidden_size, hidden_size* 2), hidden_size)
        self.CFattention2 = CFattention((hidden_size*2, hidden_size*4), hidden_size*2)
        self.CFattention3 = CFattention((hidden_size*4, hidden_size*8), hidden_size*4)
        
        self.CFattention11 = CFattention((hidden_size, hidden_size*2), hidden_size)
        self.CFattention22 = CFattention((hidden_size*2, hidden_size*4), hidden_size*2)

        self.CFattention111 = CFattention((hidden_size, hidden_size*2), hidden_size)
        
        # self.upconv0 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=2, stride=2)
        # self.upconv1 = nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=2, stride=2)
        # self.upconv2 = nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=2, stride=2)

        # self.upconv00 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=2, stride=2)
        # self.upconv11 = nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=2, stride=2)

        # self.upconv000 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=2, stride=2)

        self.upconv0000 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1)


    def forward(self, *input):
        CFattention1 = self.CFattention1(input[0], input[1])
        CFattention2 = self.CFattention2(input[1], input[2])
        CFattention3 = self.CFattention3(input[2], input[3])

        CFattention11 = self.CFattention11(CFattention1, CFattention2)
        CFattention22 = self.CFattention22(CFattention2, CFattention3)

        CFattention111 = self.CFattention111(CFattention11, CFattention22)

        # SNNC1 = self.upconv0(input[1])
        # SNNC2 = self.upconv1(input[2])
        # SNNC3 = self.upconv2(input[3])

        # SNNC1_1 = SNNC1 + input[0]
        # SNNC1_2 = SNNC2 + input[1]
        # SNNC1_3 = SNNC3 + input[2]

        # SNNC11 = self.upconv00(SNNC1_2)
        # SNNC12 = self.upconv11(SNNC1_3)

        # SNNC1_11 = SNNC1_1 + SNNC11
        # SNNC1_12 = SNNC1_2 + SNNC12

        # SNNC111 = self.upconv000(SNNC1_12)

        # SNNC1_111 = SNNC1_11 + SNNC111
        # out = self.upconv0000(SNNC1_111)
        out = self.upconv0000(CFattention111)
        outs = self.final_conv(out)

        return outs

class SNN_C(nn.Module):
    def __init__(self, input_channels=3, hidden_size=16, num_classes=1, time_windows=5):
        super(SNN_C, self).__init__()
        self.encode = SNN_C_encode(input_channels, hidden_size, num_classes, time_windows)
        self.decode = SNN_C_Decode(hidden_size, num_classes,)
    def forward(self, x):
        encode = self.encode(x)
        decode = self.decode(*encode).sigmoid()
        return decode