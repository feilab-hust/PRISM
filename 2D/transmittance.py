import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils_res import Conv


class SEBlock(nn.Module):
    """通道注意力模块"""

    def __init__(self, num_features):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x):
        b, c, h, w = x.size()
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class BasicConv2d(nn.Module):
    """一个基本的2D卷积层，后面跟ReLU激活"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = Conv(out_channels, out_channels, k=kernel_size, groups=out_channels)

    def forward(self, x):
        return self.conv(self.pw(x))


class ResidualBlockWithSE(nn.Module):
    """
    带SEBlock的残差块，包含两个卷积层。
    输入和输出通道数相同。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = BasicConv2d(channels, channels, kernel_size=3)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=3)
        self.se_block = SEBlock(channels)
        self.relu_after_add = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se_block(out)  # 通道注意力在残差路径上
        out = out + residual  # 残差连接
        return self.relu_after_add(out)  # 残差连接后的激活


class Transmittance(nn.Module):
    """
    轻量级透射率估计网络，替代RestorationNetwork3d_subback2中的Unet3D (self.back_block)。
    输入是初级编码器的特征图，输出是与输入特征通道数相同的特征图。
    该网络的输出将通过外部的 nn.Conv3d(num_features, 1, ...) (self.back_conv)
    和 nn.Sigmoid() 层，最终得到透射率图。
    """

    def __init__(self, in_channels, base_features=24, num_residual_blocks=2):
        """
        Args:
            in_channels (int): 输入特征的通道数 (即RestorationNetwork3d_subback2中的num_features)。
            base_features (int): 网络内部处理特征的通道数，用于控制参数量，应小于等于in_channels。
                                 建议设置为in_channels的因子，如in_channels=64时，base_features=16, 32等。
            num_residual_blocks (int): 堆叠的残差块数量，增加深度。
        """
        super().__init__()

        # 将输入特征通道转换为内部处理的base_features通道
        self.initial_mapping = nn.Conv2d(in_channels, base_features, kernel_size=1)

        # 堆叠多个带SE的残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlockWithSE(base_features) for _ in range(num_residual_blocks)
        ])

        # 将内部处理的base_features通道转换回in_channels，以匹配原始Unet3D的输出约定
        self.final_mapping = nn.Conv2d(base_features, in_channels, kernel_size=1)
        self.relu_final = nn.ReLU(inplace=True)  # 最终输出前的激活

    def forward(self, x):
        # 初始特征映射
        x = F.relu(self.initial_mapping(x))

        # 顺序通过残差块
        for block in self.residual_blocks:
            x = block(x)

        # 最终映射回与输入通道数相同的特征图
        x = self.relu_final(self.final_mapping(x))

        return x