import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils import DeepEncoder, Decoder
from network_utils_res import DeepEncoder_AxialTransUNet, Conv, DeepEncoder_Trans
from utils import fft_conv2d, FourierLowpass2D
from Transmittance import Transmittance


class RestorationNetwork3d_subback2(nn.Module):
    def __init__(self, scale=1, train_flag=True, num_features=64, back_features=32, num_groups=4, num_blocks=4, use_cbam=False, back_flag=True, activate="tanh",
                 back_ratio=0.2, freq_ratio_low=0.2, freq_ratio_high=0.075, freq_ratio2=0.5, attenuation_slope=0.3, model_type="CNN", shuffle_flag=True, padding_size=(0, 0, 0),
                 eps=1e-8, LN_flag=True, max_drop_path_rate=0.):
        super(RestorationNetwork3d_subback2, self).__init__()
        self.num_features = num_features
        self.primary_encoder = nn.Conv2d(1, num_features, kernel_size=3, padding=1, stride=1)
        if model_type=="CNN":
            self.deep_encoder = DeepEncoder(num_features=num_features, num_groups=num_groups, num_blocks=num_blocks, use_cbam=use_cbam)
        elif model_type=="AxialTrans":
            self.deep_encoder = DeepEncoder_AxialTransUNet(dim=num_features, LN_flag=LN_flag, max_drop_path_rate=max_drop_path_rate)
        if(scale > 1):
            self.decoder = Decoder(num_features=num_features, scale=scale, shuffle_flag=shuffle_flag)
            self.conv_out = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_out = nn.Conv2d(num_features, 1, kernel_size=1)

        self.back_block = Transmittance(num_features, back_features)
        self.back_conv = nn.Conv2d(num_features, 1, kernel_size=1)
        self.back_filter = FourierLowpass2D(cutoff_ratio=back_ratio, mode='low')

        self.activate = activate
        self.down_sample = nn.AvgPool2d(kernel_size=scale, stride=scale)
        self.low_filter = FourierLowpass2D(cutoff_ratio=freq_ratio_low, mode='low')
        self.high_filter = FourierLowpass2D(cutoff_ratio=freq_ratio_high, mode='high', cutoff_ratio2=freq_ratio2, attenuation_slope=attenuation_slope)
        self.padding_size = padding_size
        self.flag = train_flag
        self.scale = scale
        self.back_flag = back_flag
        self.eps = eps

        if not back_flag:
            for param in self.back_block.parameters():
                param.requires_grad = False
            for param in self.back_conv.parameters():
                param.requires_grad = False

    def forward(self, blurred_x, psf_x):
        h, w = blurred_x.shape[-2:]

        blurred_x_high = self.high_filter(blurred_x)

        scale_factor = torch.max(blurred_x) / torch.max(blurred_x_high)
        blurred_x_high = blurred_x_high * scale_factor

        feature = self.primary_encoder(blurred_x)
        feature_deep = self.deep_encoder(feature)
        feature_deep = feature + feature_deep
        if self.scale > 1:
            feature_deep = self.decoder(feature_deep)
        high_res_img = self.conv_out(feature_deep)
        if self.activate == "relu":
            high_res_img  = F.relu(high_res_img)
        elif self.activate == "gelu":
            high_res_img = F.gelu(high_res_img)
        elif self.activate == "sigmoid":
            high_res_img = torch.sigmoid(high_res_img)
        else:
            high_res_img = (1 + torch.tanh(high_res_img)) / 2

        if self.flag:
            # 点扩散函数模糊
            predicted_blur = fft_conv2d(high_res_img, psf_x)
            if self.scale > 1:
                predicted_blur = self.down_sample(predicted_blur)
            if self.back_flag:
                predicted_back = self.back_filter(blurred_x)
                predicted_back = predicted_back * torch.max(blurred_x) / (torch.max(predicted_back) + self.eps)
                transmittance = torch.sigmoid(self.back_conv(self.back_block(feature)))
                predicted_raw = transmittance * predicted_blur + (1.0 - transmittance) * predicted_back
                # transmittance = torch.zeros_like(predicted_blur, device=predicted_blur.device)
                # predicted_raw = predicted_blur + 0.3 * predicted_back
            else:
                if (self.padding_size != (0, 0)):
                    h = h - 2 * self.padding_size[0]
                    w = w - 2 * self.padding_size[1]
                    crop_h = slice(self.padding_size[0], self.padding_size[0] + h)
                    crop_w = slice(self.padding_size[1], self.padding_size[1] + w)
                    predicted_blur = predicted_blur[..., crop_h, crop_w]
                    crop_h = slice(self.scale * self.padding_size[0], self.scale * (self.padding_size[0] + h))
                    crop_w = slice(self.scale * self.padding_size[1], self.scale * (self.padding_size[1] + w))
                    high_res_img = high_res_img[..., crop_h, crop_w]
                return predicted_blur, high_res_img

            predicted_raw_high = self.high_filter(predicted_blur)
            # predicted_raw_high = self.high_filter(self.down_sample(high_res_img))
            scale_factor = torch.max(predicted_blur) / torch.max(predicted_raw_high)
            predicted_raw_high = predicted_raw_high * scale_factor

            if (self.padding_size != (0, 0)):
                h = h - 2 * self.padding_size[0]
                w = w - 2 * self.padding_size[1]
                crop_h = slice(self.padding_size[0], self.padding_size[0] + h)
                crop_w = slice(self.padding_size[1], self.padding_size[1] + w)
                predicted_raw = predicted_raw[..., crop_h, crop_w]
                predicted_raw_high = predicted_raw_high[..., crop_h, crop_w]
                blurred_x_high = blurred_x_high[..., crop_h, crop_w]
                predicted_blur = predicted_blur[..., crop_h, crop_w]
                if self.back_flag:
                    transmittance = transmittance[..., crop_h, crop_w]
        else:
            predicted_raw = 0
            predicted_raw_high = 0
            blurred_x_high = 0
            predicted_blur = 0
            transmittance = 0
            if (self.padding_size != (0, 0)):
                h = h - 2 * self.padding_size[0]
                w = w - 2 * self.padding_size[1]

        if (self.padding_size != (0, 0)):
            crop_h = slice(self.scale * self.padding_size[0], self.scale * (self.padding_size[0] + h))
            crop_w = slice(self.scale * self.padding_size[1], self.scale * (self.padding_size[1] + w))
            high_res_img = high_res_img[..., crop_h, crop_w]

        return predicted_raw, predicted_raw_high, high_res_img, blurred_x_high, predicted_blur, transmittance


class RestorationNetwork3d_Inference(nn.Module):
    def __init__(self, scale=1, num_features=64, num_groups=4, num_blocks=4, use_cbam=True, activate="tanh",
                model_type="CNN", shuffle_flag=True, padding_size=(0, 0, 0),LN_flag=True):
        super(RestorationNetwork3d_Inference, self).__init__()
        self.num_features = num_features
        self.scale = scale
        self.padding_size = padding_size

        # 图像重建部分 (与训练模型保持一致的初始化)
        self.primary_encoder = nn.Conv2d(1, num_features, kernel_size=3, padding=1, stride=1)

        if model_type == "CNN":
            self.deep_encoder = DeepEncoder(num_features=num_features, num_groups=num_groups, num_blocks=num_blocks, use_cbam=use_cbam)
        elif model_type=="AxialTrans":
            self.deep_encoder = DeepEncoder_AxialTransUNet(dim=num_features, LN_flag=LN_flag)

        if scale > 1:
            self.decoder = Decoder(num_features=num_features, scale=scale, shuffle_flag=shuffle_flag)
            self.conv_out = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_out = nn.Conv2d(num_features, 1, kernel_size=1)

        # 激活函数也保留
        self.activate = activate

    def forward(self, blurred_x):
        """
        推理阶段的前向传播。
        Args:
            blurred_x (torch.Tensor): 模糊的输入图像 [B, C, H, W]
        Returns:
            torch.Tensor: 重建后的高分辨率图像 [B, C, H, W]
        """

        # 核心图像重建路径
        feature = self.primary_encoder(blurred_x)
        feature_deep = self.deep_encoder(feature)
        feature_deep = feature + feature_deep

        if self.scale > 1:
            feature_deep = self.decoder(feature_deep)

        high_res_img = self.conv_out(feature_deep)
        if self.activate == "relu":
            high_res_img = F.relu(high_res_img)
        elif self.activate == "gelu":
            high_res_img = F.gelu(high_res_img)
        elif self.activate == "sigmoid":
            high_res_img = torch.sigmoid(high_res_img)
        else:
            high_res_img = (1 + torch.tanh(high_res_img)) / 2

        if self.padding_size != (0, 0):
            original_h, original_w = blurred_x.shape[-2:]

            # 计算裁剪后的原始有效尺寸
            cropped_h = original_h - 2 * self.padding_size[0]
            cropped_w = original_w - 2 * self.padding_size[1]

            crop_h_start = self.scale * self.padding_size[0]
            crop_h_end = self.scale * (self.padding_size[0] + cropped_h)

            crop_w_start = self.scale * self.padding_size[1]
            crop_w_end = self.scale * (self.padding_size[1] + cropped_w)

            # 确保索引在有效范围内
            crop_h_end = min(crop_h_end, high_res_img.shape[-2])
            crop_w_end = min(crop_w_end, high_res_img.shape[-1])

            high_res_img = high_res_img[..., crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return high_res_img
