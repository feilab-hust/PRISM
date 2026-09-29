import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils_res import Conv


class Conv2(nn.Module):
    def __init__(self, dim_i, dim_o, k, bias=True, device=None):
        super().__init__()
        p = (k - 1) // 2
        self.pw = nn.Conv3d(dim_i, dim_o, 1, bias=bias, device=device)
        self.dw = Conv(dim_o, dim_o, k, groups=dim_o, bias=bias, device=device)
        # self.dw = nn.Conv3d(dim_o, dim_o, kernel_size=k, stride=1, padding=p, groups=dim_o, bias=bias, device=device)

    def forward(self, x):
        return self.dw(self.pw(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = Conv2(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def build_feature_extractor(in_channels, out_channels, num_blocks):
    layers = []
    for _ in range(num_blocks):
        layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        in_channels = out_channels  # 更新输入通道数
    return nn.Sequential(*layers)


def build_downsample_block(in_channels, out_channels):
    return ConvBlock(in_channels, out_channels, kernel_size=2, stride=2, padding=0)


class SEBlock(nn.Module):
    """通道注意力模块"""
    def __init__(self, num_features, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(num_features, num_features // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_features // reduction, num_features),
        #     nn.Sigmoid()
        # )
        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x):
        b, c, d, h, w = x.size()
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1, 1)
        return x * weights


class CBAM(nn.Module):
    """混合通道-空间注意力模块"""

    def __init__(self, num_features, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.sigmoid = nn.Sigmoid()
        # 通道注意力
        self.channel_att1 = nn.Sequential(
            # nn.Conv3d(num_features, num_features // reduction, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(num_features // reduction, num_features, 1)
            nn.Conv3d(num_features, num_features, 1)
        )
        self.channel_att2 = nn.Sequential(
            # nn.Conv3d(num_features, num_features // reduction, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(num_features // reduction, num_features, 1)
            nn.Conv3d(num_features, num_features, 1)
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            # nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2),
            Conv2(2, 1, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        channel_weights = self.sigmoid(self.channel_att1(x1) + self.channel_att2(x2))
        x_channel = x * channel_weights

        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_weights = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x_spatial = x_channel * spatial_weights

        return x_spatial


class ResidualGroup(nn.Module):
    def __init__(self, num_features, num_blocks, use_cbam=True):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock(num_features) for _ in range(num_blocks)]
        )
        # self.conv = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.conv = Conv2(num_features, num_features, k=3)
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(num_features)

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x = self.conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x + residual


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualChannelAttentionBlock, self).__init__()
        # self.conv1 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.conv1 = Conv2(num_features, num_features, k=3)
        self.conv2 = Conv2(num_features, num_features, k=3)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(num_features)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.ca(x)
        return x + residual


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # self.fc1 = nn.Conv3d(num_features, num_features // reduction, kernel_size=1, padding=0)
        # self.fc2 = nn.Conv3d(num_features // reduction, num_features, kernel_size=1, padding=0)
        self.fc = nn.Conv3d(num_features, num_features, kernel_size=1, padding=0)
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        # avg_out = self.relu(self.fc1(avg_out))
        # avg_out = self.sigmoid(self.fc2(avg_out))
        avg_out = self.fc(avg_out)
        return x * avg_out


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor=1, scale_axial=1):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
        self.scale_axial = scale_axial

    def forward(self, x):
        b, c, d, h, w = x.shape
        upscale = self.upscale_factor
        scale_axial = self.scale_axial
        new_c = c // (upscale ** 2 * scale_axial)

        # Reshape and permute to upscale h and w dimensions
        x = x.view(b, new_c, scale_axial, upscale, upscale, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, new_c, d*scale_axial, h * upscale, w * upscale)

        return x


class UnPixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor=1, scale_axial=1):
        super(UnPixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
        self.scale_axial = scale_axial

    def forward(self, x):
        b, c, d, h, w = x.shape
        upscale = self.upscale_factor
        scale_axial = self.scale_axial
        new_c = c * upscale ** 2 * scale_axial

        # Reshape and permute to upscale h and w dimensions
        x = x.view(b, c, d//scale_axial, scale_axial, h//upscale, upscale, w//upscale, upscale)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, new_c, d//scale_axial, h//upscale, w//upscale)

        return x


# class ConvTranspose(nn.Module):
#     def __init__(self, num_features, scale, scale_axial):
#         super(ConvTranspose, self).__init__()
#         self.scale = scale
#         self.scale_axial = scale_axial
#
#         # 使用转置卷积进行空间尺度的上采样
#         self.conv_transpose = nn.ConvTranspose3d(
#             num_features,
#             num_features,
#             kernel_size=2,
#             stride=(scale_axial, scale, scale),
#             padding=0
#         )
#
#     def forward(self, x):
#         # 对输入进行转置卷积上采样
#         x = self.conv_transpose(x)
#         return x


class ConvTranspose(nn.Module):
    def __init__(self, num_features, scale, scale_axial):
        super(ConvTranspose, self).__init__()
        self.scale = scale
        self.scale_axial = scale_axial

        # 使用转置卷积进行空间尺度的上采样
        self.conv_transpose = nn.ConvTranspose3d(
            int(num_features*scale**2*scale_axial),
            num_features,
            kernel_size=3,
            stride=(scale_axial, scale, scale),
            padding=1,
            output_padding=(scale_axial-1, scale-1, scale-1)
        )

    def forward(self, x):
        # 对输入进行转置卷积上采样
        x = self.conv_transpose(x)
        return x


class PrimaryEncoder(nn.Module):
    """初级编码器，用于提取初步特征"""
    def __init__(self, in_channels=1, num_features=32):
        super(PrimaryEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        # self.activate = nn.ReLU(inplace=True)
        self.activate = nn.GELU()
        self.se_block = SEBlock(num_features)

    def forward(self, x):
        residual = x
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = self.conv3(x)
        x += residual  # 添加残差连接
        x = self.activate(x)  # 残差后的激活函数
        x = self.se_block(x)
        return x


class DeepEncoder(nn.Module):
    """深度编码器，由通道注意力残差组组成"""
    def __init__(self, num_features, num_groups=5, num_blocks=5, use_cbam=True):
        """
        :param num_features: 通道数
        :param num_groups: 通道注意力残差组数量
        :param num_blocks: 每个残差组中的残差块数量
        """
        super(DeepEncoder, self).__init__()
        self.residual_groups = nn.ModuleList(
            [ResidualGroup(num_features, num_blocks, use_cbam=use_cbam) for _ in range(num_groups)]
        )
        # self.final_conv = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)
        self.final_conv = Conv2(num_features, num_features, k=3)

    def forward(self, x):
        """
        :param x: 输入特征
        :return: 输出深度特征
        """
        for i, group in enumerate(self.residual_groups):
            x = group(x)
        x = self.final_conv(x)
        return x


class Decoder(nn.Module):
    """解码器，结合横向和轴向上采样"""
    def __init__(self, num_features, scale, scale_axial, shuffle_flag=True):
        super(Decoder, self).__init__()
        self.scale = scale
        self.scale_axial = scale_axial

        # self.conv3d = nn.Conv3d(num_features, int(num_features * (scale ** 2) * scale_axial), kernel_size=1)
        self.conv3d = nn.Conv3d(num_features, int(scale ** 2 * scale_axial), kernel_size=1)
        if shuffle_flag:
            self.upsample = PixelShuffle3D(scale, int(scale_axial))
        else:
            self.upsample = ConvTranspose(num_features, scale, int(scale_axial))

    def forward(self, x):
        x = self.conv3d(x)
        x = self.upsample(x)
        return x


class FusionReconstruction(nn.Module):
    """图像融合重建模块，生成中间图像"""
    def __init__(self, num_features=32, out_channels=1, flag="concat"):
        super(FusionReconstruction, self).__init__()
        if (flag=="concat"):
            self.reconstructor = nn.Sequential(
                nn.Conv3d(2 * num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)
            )
        elif (flag=="add"):
            self.reconstructor = nn.Sequential(
                nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)
            )
        self.flag = flag

    def forward(self, feature_x, feature_y):
        """
        输入两个特征，生成中间图像
        :param feature_x: 初级编码器提取的特征 (x方向)
        :param feature_y: 初级编码器提取的特征 (y方向)
        :return: 中间图像
        """
        # 融合特征
        if (self.flag == "concat"):
            fused_features = torch.cat((feature_x, feature_y), dim=1)
        elif (self.flag == "add"):
            fused_features = feature_x + feature_y
        # 重建中间图像
        intermediate_image = self.reconstructor(fused_features)
        return intermediate_image
