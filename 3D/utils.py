import warnings
import random
from math import exp
import numpy as np
from tifffile import imread, imwrite

import torch
from torch import nn
import torch.nn.functional as F
from torch.fft import fftn, fft2
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import kornia.filters as kornia_filters


# 定义函数，将张量归一化到 [0, 1] 范围
def normalize_to_01(tensor):
    """
    归一化张量到 [0,1]，仅在 H, W 或 D, H, W 维度进行归一化，不影响 B, C 维度。
    """
    dims = tuple(range(2, tensor.ndim))  # 计算 H, W 或 D, H, W 维度的最小最大值

    min_val = tensor.amin(dim=dims, keepdim=True)  # 计算最小值
    max_val = tensor.amax(dim=dims, keepdim=True)  # 计算最大值

    return (tensor - min_val) / (max_val - min_val + 1e-6)  # 避免除零


# SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


class TVLoss(nn.Module):
    def __init__(self, use_l2=True):
        """
        TV Loss
        Args:
            use_l2 (bool): True for L2 norm (torch.pow), False for L1 norm (torch.abs).
        """
        super(TVLoss, self).__init__()
        self.use_l2 = use_l2

    def forward(self, y_pred):
        # Compute TV losses for each dimension
        if self.use_l2:
            x_diff = torch.pow(y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:], 2).mean()
            y_diff = torch.pow(y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :], 2).mean()
            z_diff = torch.pow(y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :], 2).mean()
        else:
            x_diff = torch.abs(y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]).mean()
            y_diff = torch.abs(y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]).mean()
            z_diff = torch.abs(y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]).mean()

        # Return the total TV loss
        return x_diff + y_diff, z_diff


class HessianLoss(nn.Module):
    def __init__(self, use_l2=True):
        """
        Hessian Loss
        Args:
            use_l2 (bool): True for L2 norm (torch.pow), False for L1 norm (torch.abs).
        """
        super(HessianLoss, self).__init__()
        self.use_l2 = use_l2

    def forward(self, y_pred):
        # First-order differences
        z_diff = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]
        y_diff = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]
        x_diff = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]

        # Second-order differences (Hessian components)
        if self.use_l2:
            zz_diff = torch.pow(z_diff[:, :, :-1, :, :] - z_diff[:, :, 1:, :, :], 2).mean()
            yy_diff = torch.pow(y_diff[:, :, :, :-1, :] - y_diff[:, :, :, 1:, :], 2).mean()
            xx_diff = torch.pow(x_diff[:, :, :, :, :-1] - x_diff[:, :, :, :, 1:], 2).mean()
            xy_diff = torch.pow(x_diff[:, :, :, :-1, :] - x_diff[:, :, :, 1:, :], 2).mean()
            yz_diff = torch.pow(y_diff[:, :, :-1, :, :] - y_diff[:, :, 1:, :, :], 2).mean()
            xz_diff = torch.pow(x_diff[:, :, :-1, :, :] - x_diff[:, :, 1:, :, :], 2).mean()
        else:
            zz_diff = torch.abs(z_diff[:, :, :-1, :, :] - z_diff[:, :, 1:, :, :]).mean()
            yy_diff = torch.abs(y_diff[:, :, :, :-1, :] - y_diff[:, :, :, 1:, :]).mean()
            xx_diff = torch.abs(x_diff[:, :, :, :, :-1] - x_diff[:, :, :, :, 1:]).mean()
            xy_diff = torch.abs(x_diff[:, :, :, :-1, :] - x_diff[:, :, :, 1:, :]).mean()
            yz_diff = torch.abs(y_diff[:, :, :-1, :, :] - y_diff[:, :, 1:, :, :]).mean()
            xz_diff = torch.abs(x_diff[:, :, :-1, :, :] - x_diff[:, :, 1:, :, :]).mean()

        # Return the sum of all second-order differences
        return zz_diff + yy_diff + xx_diff + 2 * xy_diff + 2 * yz_diff + 2 * xz_diff
        # return yy_diff + xx_diff + 2 * xy_diff


class L1SparsityLoss(nn.Module):
    def __init__(self):
        super(L1SparsityLoss, self).__init__()

    def forward(self, y_pred):
        y_pred = torch.abs(y_pred)
        y_pred_normalized = y_pred / (y_pred.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
        return torch.mean(y_pred_normalized)


class FourierMagnitudeLoss(nn.Module):
    def __init__(self, alpha = 4.0):
        super(FourierMagnitudeLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred):
        # 形状：[batch, channels, depth, height, width]
        batch_size, channels, depth, height, width = y_pred.shape

        # 初始化总损失
        total_loss = 0.0

        # 遍历深度方向的每个切片
        for z in range(depth):
            # 取出 z 方向的切片 [batch, channels, height, width]
            slice_2d = y_pred[:, :, z, :, :]

            # 对每个 2D 切片执行 2D FFT
            slice_fft = torch.fft.fftshift(torch.fft.fft2(slice_2d))

            # 计算频谱模
            magnitude = torch.abs(slice_fft)
            magnitude = torch.log(magnitude + 1 + 1e-8)

            # 归一化（按切片的最大值）
            # slice_magnitude_normalized = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            slice_magnitude_normalized = normalize_to_01(magnitude)
            # slice_magnitude_normalized = (torch.sigmoid(self.alpha * slice_magnitude_normalized) - 0.5) / (1/(1 + exp(-self.alpha)) - 0.5)

            # 计算损失 (1 - 归一化频谱模均值)
            slice_loss = torch.mean(1 - slice_magnitude_normalized)

            # 累加损失
            total_loss += slice_loss

        # 对所有切片的损失求平均
        total_loss /= depth

        return total_loss


class FourierMagnitudeLoss3D(nn.Module):
    def __init__(self):
        super(FourierMagnitudeLoss3D, self).__init__()

    def forward(self, y_pred):
        spectrum = torch.fft.fftshift(fftn(y_pred, dim = (-3, -2, -1)), dim = (-3, -2, -1))
        magnitude = torch.log(torch.abs(spectrum) + 1 + 1e-8)
        magnitude = normalize_to_01(magnitude)

        total_loss = torch.mean(1 - magnitude)

        return total_loss


class BackgroundSparsityLoss(nn.Module):
    def __init__(self,
                 window_size=7,
                 var_threshold=0.1,
                 penalty_type='exp',
                 eps=1e-8):
        """
        局部背景强度约束正则项
        Args:
            window_size (int): 计算局部统计量的窗口尺寸（奇数）
            var_threshold (float): 判定背景区域的方差阈值
            penalty_type (str): 惩罚类型 'exp'|'quadratic'
            eps (float): 数值稳定性常数
        """
        super().__init__()
        self.window_size = window_size
        self.var_threshold = var_threshold
        self.penalty_type = penalty_type
        self.eps = eps
        self.pool = nn.AvgPool3d(window_size, stride=1, padding=window_size // 2)

    def forward(self, y_pred):
        """
        Args:
            y_pred: 预测的高分辨率图像 [B,C,D,H,W]
        Returns:
            loss: 正则项损失值
        """
        # 计算局部均值和方差
        local_mean = self.pool(y_pred)
        local_std = torch.sqrt(self.pool(y_pred ** 2) - local_mean ** 2)

        # 生成背景区域掩码（低方差区域）
        background_mask = torch.sigmoid(
            (self.var_threshold - local_std) * 100  # 陡峭过渡
        )  # [0,1] 越接近1表示越可能是背景

        # 计算背景区域强度偏移量
        # background_offset = y_pred - local_mean.detach()  # 阻止均值参与梯度计算
        background_offset = y_pred

        # 选择惩罚函数
        if self.penalty_type == 'exp':
            penalty = torch.exp(F.relu(background_offset)) - 1.0
        elif self.penalty_type == 'quadratic':
            penalty = F.relu(background_offset) ** 2
        else:
            raise ValueError(f"Unsupported penalty type: {self.penalty_type}")

        # 加权平均损失
        weighted_penalty = penalty * background_mask
        valid_pixels = background_mask.sum() + self.eps
        return weighted_penalty.sum() / valid_pixels


class MinPool2d(nn.Module):
    """可导二维最小池化层"""

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        # 通过取反实现最小池化
        x_neg = -x
        output, _ = self.maxpool(x_neg)
        return -output


class MinPool3d(nn.Module):
    """可导三维最小池化层"""

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool = nn.MaxPool3d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        # 通过取反实现最小池化
        x_neg = -x
        output, _ = self.maxpool(x_neg)
        return -output


class DarkChannelPrior3D(nn.Module):
    def __init__(self,
                 window_size,
                 alpha=0.5,
                 eps=1e-8,
                 mode='3d'):
        """
        基于最小池化的暗通道先验正则项
        Args:
            window_size : 池化窗口尺寸
            alpha (float): 惩罚指数系数
            eps (float): 数值稳定性常数
            mode (str): '2d'对每个切片独立处理，'3d'三维处理
        """
        super().__init__()
        self.window_size = window_size
        self.alpha = alpha
        self.eps = eps
        self.mode = mode

        # 最小池化层
        if mode == '2d':
            # self.pool = MinPool2d(window_size, stride=1, padding=window_size // 2)
            self.pool = MinPool2d(window_size, stride=1, padding=0)
            # self.pool = MinkAveragePool2d(kernel_size=window_size, stride=1, padding=0, k_percent=15)
        else:
            # if len(window_size) == 1:
            #     padding = window_size // 2
            # else:
            #     padding = (window_size[0]//2, window_size[1]//2, window_size[2]//2)
            padding = 0
            self.pool = MinPool3d(window_size, stride=1, padding=padding)
            # self.pool = MinkAveragePool3d(kernel_size=window_size, stride=1, padding=padding, k_percent=15)

    def forward(self, y_pred):
        """
        Args:
            y_pred: 预测的高分辨率图像 [B,C,D,H,W]
        Returns:
            loss: 正则项损失值
        """
        # 提取暗通道
        y_pred = torch.abs(y_pred)
        if self.mode == '2d':
            B, C, D, H, W = y_pred.shape
            # 按切片处理
            y_pred_2d = y_pred.reshape(B * C, D, H, W)  # [B*C, D, H, W]
            dark = self.pool(y_pred_2d)  # [B*C, D, ~, ~]
            # dark = dark.view(B, C, D, H, W)  # 恢复形状
        else:
            dark = self.pool(y_pred)  # 3D处理

        dark += self.eps

        # 计算归一化暗通道损失
        # max_val = y_pred.amax(dim=(2, 3, 4), keepdim=True) + self.eps  # 防止除零
        # dark_norm = dark / max_val
        dark_norm = dark
        return torch.mean(torch.pow(dark_norm, self.alpha))


class MinkAveragePool3d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, k_percent=15):
        """
        三维最小k%平均值池化：对每个池化区域，取最小的k%元素求平均
        Args:
            kernel_size: 池化核大小 (int或tuple)
            stride: 步长 (int或tuple，默认=kernel_size)
            padding: 填充 (int或tuple，默认=0)
            k_percent: 取最小元素的百分比 (0-100]，默认50（取最小50%）
        """
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.k_percent = max(1e-6, min(100.0, k_percent))  # 限制范围

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        kd, kh, kw = self.kernel_size
        window_size = kd * kh * kw
        count = max(1, int(window_size * self.k_percent / 100))  # 至少取1个元素

        # 1. 展开池化区域：[B, C, D_out, H_out, W_out, kd*kh*kw]
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1
        )  # [B, C*kd*kh*kw, D_out*H_out*W_out]
        x_unfold = x_unfold.reshape(B, C, window_size, -1)  # [B, C, window_size, N]，N=D_out*H_out*W_out
        x_unfold = x_unfold.permute(0, 1, 3, 2)  # [B, C, N, window_size]

        # 2. 升序排序，取前count个最小元素
        x_sorted, _ = torch.sort(x_unfold, dim=-1, descending=False)
        x_mink = x_sorted[..., :count]  # [B, C, N, count]

        # 3. 计算平均值并reshape回目标形状
        x_avg = x_mink.mean(dim=-1)  # [B, C, N]
        D_out = (D + 2 * self.padding[0] - kd) // self.stride[0] + 1
        H_out = (H + 2 * self.padding[1] - kh) // self.stride[1] + 1
        W_out = (W + 2 * self.padding[2] - kw) // self.stride[2] + 1
        x_out = x_avg.reshape(B, C, D_out, H_out, W_out)

        return x_out


class MinkAveragePool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, k_percent=15):
        """
        二维最小k%平均值池化：对每个池化区域，取最小的k%元素求平均
        Args:
            kernel_size: 池化核大小 (int或tuple)
            stride: 步长 (int或tuple，默认=kernel_size)
            padding: 填充 (int或tuple，默认=0)
            k_percent: 取最小元素的百分比 (0-100]，默认50（取最小50%）
        """
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.k_percent = max(1e-6, min(100.0, k_percent))  # 限制范围

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        window_size = kh * kw
        count = max(1, int(window_size * self.k_percent / 100))  # 至少取1个元素

        # 1. 展开池化区域：[B, C, H_out, W_out, kh*kw]
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1
        )  # [B, C*kh*kw, H_out*W_out]
        x_unfold = x_unfold.reshape(B, C, window_size, -1)  # [B, C, window_size, N]，N=H_out*W_out
        x_unfold = x_unfold.permute(0, 1, 3, 2)  # [B, C, N, window_size]

        # 2. 升序排序，取前count个最小元素
        x_sorted, _ = torch.sort(x_unfold, dim=-1, descending=False)
        x_mink = x_sorted[..., :count]  # [B, C, N, count]

        # 3. 计算平均值并reshape回目标形状
        x_avg = x_mink.mean(dim=-1)  # [B, C, N]
        H_out = (H + 2 * self.padding[0] - kh) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - kw) // self.stride[1] + 1
        x_out = x_avg.reshape(B, C, H_out, W_out)

        return x_out


def generate_gaussian_kernel_3d(size, sigma_x, sigma_y, sigma_z):
    """生成具有不同x, y, z维度方差的三维高斯核"""
    x, y, z = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[2]), np.linspace(-1, 1, size[0]))
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2) + z**2 / (2 * sigma_z**2)))
    gauss_kernel /= np.sum(gauss_kernel)  # 归一化
    return gauss_kernel


def generate_gaussian_kernel_3d_tensor(size, sigma_x, sigma_y, sigma_z):
    """生成具有不同x, y, z维度方差的三维高斯核"""
    x, y, z = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[2]), np.linspace(-1, 1, size[0]))
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2) + z**2 / (2 * sigma_z**2)))
    gauss_kernel /= np.sum(gauss_kernel)  # 归一化
    gauss_kernel = np.transpose(gauss_kernel, [2, 0, 1])
    gauss_kernel = torch.tensor(gauss_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return gauss_kernel


def add_noise(x, noise=0.1):
    x_noise = torch.normal(x, random.uniform(0, noise))
    return x_noise


def add_back(x, kernal, alpha=1):
    d1, h1, w1 = kernal.shape[-3:]
    x1 = F.pad(x, (
        (w1 - 1) // 2, (w1 - 1) // 2, (h1 - 1) // 2, (h1 - 1) // 2, (d1 - 1) // 2, (d1 - 1) // 2), 'reflect')
    predicted_back = F.conv3d(x1, kernal, padding=0)
    x_addback = x + alpha * predicted_back
    return x_addback


def my_ifftshift(x, dims):
    """
    模拟 np.fft.ifftshift 的行为：
    对于每个维度，如果维度大小为 n，
      - 如果 n 为偶数，则向左循环移动 n//2 个位置；
      - 如果 n 为奇数，则向左移动 (n+1)//2 个位置。
    """
    for d in dims:
        n = x.shape[d]
        shift = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
        x = torch.roll(x, shifts=shift, dims=d)
    return x


def fft_conv3d(image, kernel, padding_mode='reflect'):
    """
    基于FFT的三维卷积实现，修正非2的整数幂尺寸输入时的伪影问题。
    通过预先对 kernel 进行 ifftshift 操作，实现核中心与零频对应，
    从而在逆FFT后无需额外 roll 即可还原出与 F.conv3d 相同的结果。

    :param image: 输入图像 [B, C, D, H, W]
    :param kernel: 卷积核 [1, 1, kD, kH, kW]
    :param padding_mode: 边缘填充方式
    :return: 卷积结果 [B, C, D, H, W]
    """
    B, C, D, H, W = image.shape
    kD, kH, kW = kernel.shape[-3:]

    # 计算理论卷积输出尺寸（即线性卷积尺寸：N + k - 1）
    target_size = (D + kD - 1, H + kH - 1, W + kW - 1)

    # 为了提高 FFT 计算速度，将每个轴的尺寸扩展到最近的2的幂次
    # fft_size = tuple([2 ** int(torch.ceil(torch.log2(torch.tensor(s, dtype=torch.float32))).item())
    #                   for s in target_size])
    fft_size = tuple([s for s in target_size])

    # --------------------
    # 对 kernel 进行填充，使其尺寸为 fft_size，然后预先做 ifftshift 以对齐核中心
    pad_w_kernel = (fft_size[2] - kW) // 2
    pad_h_kernel = (fft_size[1] - kH) // 2
    pad_d_kernel = (fft_size[0] - kD) // 2
    kernel_padded = F.pad(kernel,
                          (pad_w_kernel, fft_size[2] - kW - pad_w_kernel,
                           pad_h_kernel, fft_size[1] - kH - pad_h_kernel,
                           pad_d_kernel, fft_size[0] - kD - pad_d_kernel))
    # 预先调整 kernel，使得其中心移到左上角（零频处）
    kernel_padded = my_ifftshift(kernel_padded, dims=(-3, -2, -1))

    # --------------------
    # 对图像进行填充，使其尺寸为 fft_size（采用指定的边缘填充方式）
    pad_w_img = (fft_size[2] - W) // 2
    pad_h_img = (fft_size[1] - H) // 2
    pad_d_img = (fft_size[0] - D) // 2
    image_padded = F.pad(image,
                         (pad_w_img, fft_size[2] - W - pad_w_img,
                          pad_h_img, fft_size[1] - H - pad_h_img,
                          pad_d_img, fft_size[0] - D - pad_d_img),
                         mode=padding_mode)

    # --------------------
    # FFT 变换
    image_fft = torch.fft.rfftn(image_padded, s=fft_size, dim=(-3, -2, -1))
    kernel_fft = torch.fft.rfftn(kernel_padded, s=fft_size, dim=(-3, -2, -1))

    # 频域相乘
    result_fft = image_fft * kernel_fft

    # 逆 FFT 得到卷积结果（此时已经是线性卷积的正确排列）
    result = torch.fft.irfftn(result_fft, s=fft_size, dim=(-3, -2, -1))

    # --------------------
    # 裁剪出有效区域：恢复为原图尺寸（假定 F.conv3d 的输出范围）
    crop_d = slice(pad_d_img, pad_d_img + D)
    crop_h = slice(pad_h_img, pad_h_img + H)
    crop_w = slice(pad_w_img, pad_w_img + W)
    result = result[..., crop_d, crop_h, crop_w]

    return result.real


class Fourier3DLowpass(nn.Module):
    def __init__(self,
                 cutoff_ratio_z=0.1,
                 cutoff_ratio_xy=0.1,
                 filter_type='gaussian',
                 mode='low',
                 padding_ratio=0.1,  # 新增：填充尺寸占图像尺寸的比例
                 epsilon=1e-8):
        """
        三维各向异性高/低通滤波模块（支持轴向和横向不同截止频率）
        Args:
            cutoff_ratio_z (float): 轴向归一化截止频率比例 (0-1)
            cutoff_ratio_xy (float): 横向归一化截止频率比例 (0-1)
            filter_type (str): 滤波器类型 ['ideal', 'gaussian']
            mode (str): 'low' 为低通，'high' 为高通
            padding_ratio (float): 填充尺寸占原始图像尺寸的比例 (例如 0.1 表示填充 10% 的边缘)。
            epsilon (float): 数值稳定性常数
        """
        super().__init__()
        self.cutoff_ratio_z = cutoff_ratio_z
        self.cutoff_ratio_xy = cutoff_ratio_xy
        self.filter_type = filter_type
        self.mode = mode
        self.padding_ratio = padding_ratio  # 保存填充比例
        self.epsilon = epsilon

    def _create_3d_filter(self, D, H, W, device):
        """生成三维各向异性低通滤波器"""
        # 生成归一化频率坐标 (范围[-0.5, 0.5])
        z = torch.linspace(-0.5, 0.5, D, device=device)
        y = torch.linspace(-0.5, 0.5, H, device=device)
        x = torch.linspace(-0.5, 0.5, W, device=device)

        # 构建三维网格
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

        # 计算各向异性频率距离
        distance_z = torch.abs(zz)  # 轴向距离
        distance_xy = torch.sqrt(yy ** 2 + xx ** 2)  # 横向距离

        # 计算各向截止频率
        cutoff_z = self.cutoff_ratio_z * 0.5  # 轴向最大距离为0.5
        cutoff_xy = self.cutoff_ratio_xy * 0.5 * torch.sqrt(torch.tensor(2.0))  # 横向最大距离为0.5√2

        # 生成滤波器掩膜
        if self.filter_type == 'ideal':
            mask_z = (distance_z <= cutoff_z).float()
            mask_xy = (distance_xy <= cutoff_xy).float()
            mask = mask_z * mask_xy  # 组合轴向和横向掩膜
        elif self.filter_type == 'gaussian':
            sigma_z = cutoff_z / (np.sqrt(-2 * np.log(0.5))) + self.epsilon  # 3σ覆盖主要频率成分
            sigma_xy = cutoff_xy / (np.sqrt(-2 * np.log(0.5))) + self.epsilon
            mask_z = torch.exp(-(distance_z ** 2) / (2 * sigma_z ** 2))
            mask_xy = torch.exp(-(distance_xy ** 2) / (2 * sigma_xy ** 2))
            mask = mask_z * mask_xy  # 组合轴向和横向掩膜
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")

        return mask

    def forward(self, x):
        """
        输入: [B, C, D, H, W]
        输出: 低频背景 [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        device = x.device

        pad_d = max(1, int(D * self.padding_ratio))
        pad_h = max(1, int(H * self.padding_ratio))
        pad_w = max(1, int(W * self.padding_ratio))

        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')
        D_padded, H_padded, W_padded = x_padded.shape[-3:]

        # 三维傅里叶变换
        x_fft = torch.fft.fftn(x_padded, dim=(-3, -2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))

        # 生成滤波器
        mask = self._create_3d_filter(D_padded, H_padded, W_padded, x.device)
        # imwrite(r"J:\zzh\deep_deconv\new_stack2\20250416\cached_data(Actx1)\simu\high_mask.tif", mask.cpu().numpy())

        # 应用滤波
        if self.mode == 'low':
            filtered_fft = x_fft_shift * mask
        elif self.mode == 'high':
            # 高通滤波器 = 原始频谱乘以 (1 - mask)
            high_mask = 1 - mask
            # imwrite(r"J:\zzh\deep_deconv\new_stack2\20250416\cached_data(Actx1)\simu\high_mask.tif", high_mask.cpu().numpy())
            filtered_fft = x_fft_shift * high_mask
        else:
            raise ValueError("Mode must be 'low' or 'high'")

        # 逆傅里叶变换
        filtered = torch.fft.ifftn(
            torch.fft.ifftshift(filtered_fft, dim=(-3, -2, -1)),
            dim=(-3, -2, -1)
        ).real
        # filtered = torch.abs(torch.fft.ifftn(
        #     torch.fft.ifftshift(filtered_fft, dim=(-3, -2, -1)),
        #     dim=(-3, -2, -1)
        # ))

        cropped_z_start = pad_d
        cropped_z_end = pad_d + D
        cropped_y_start = pad_h
        cropped_y_end = pad_h + H
        cropped_x_start = pad_w
        cropped_x_end = pad_w + W

        filtered = filtered[
                           :, :,
                           cropped_z_start:cropped_z_end,
                           cropped_y_start:cropped_y_end,
                           cropped_x_start:cropped_x_end
                           ]


        return filtered.clamp(min=self.epsilon)


# class Fourier3DLowpass(nn.Module):
#     def __init__(self,
#                  cutoff_ratio_z=0.1,
#                  cutoff_ratio_xy=0.1,
#                  filter_type='gaussian',
#                  mode='low',
#                  # 新增用于高通模式下高频衰减的参数
#                  cutoff_ratio2_z=None,  # 轴向高频衰减起始的截止频率比例
#                  cutoff_ratio2_xy=None,  # 横向高频衰减起始的截止频率比例
#                  attenuation_slope_z=0.3,  # 轴向衰减的陡峭程度（Sigmoid的k值相关）
#                  attenuation_slope_xy=0.3,  # 横向衰减的陡峭程度
#                  padding_ratio=0.1,  # 新增：填充尺寸占图像尺寸的比例
#                  epsilon=1e-8):
#         """
#         三维各向异性高/低通滤波模块（支持轴向和横向不同截止频率）
#         Args:
#             cutoff_ratio_z (float): 轴向归一化截止频率比例 (0-1)
#             cutoff_ratio_xy (float): 横向归一化截止频率比例 (0-1)
#             filter_type (str): 滤波器类型 ['ideal', 'gaussian']
#             mode (str): 'low' 为低通，'high' 为高通
#             cutoff_ratio2_z (float, optional): 仅在 mode='high' 时有效。
#                                               轴向高频衰减的起始频率比例。
#                                               应大于 cutoff_ratio_z。
#             cutoff_ratio2_xy (float, optional): 仅在 mode='high' 时有效。
#                                                横向高频衰减的起始频率比例。
#                                                应大于 cutoff_ratio_xy。
#             attenuation_slope_z (float): 仅在 mode='high' 且启用 cutoff_ratio2_z/xy 时有效。
#                                        轴向高频衰减的陡峭程度。值越大，衰减越陡峭。
#             attenuation_slope_xy (float): 仅在 mode='high' 且启用 cutoff_ratio2_z/xy 时有效。
#                                         横向高频衰减的陡峭程度。值越大，衰减越陡峭。
#             padding_ratio (float): 填充尺寸占原始图像尺寸的比例 (例如 0.1 表示填充 10% 的边缘)。
#             epsilon (float): 数值稳定性常数
#         """
#         super().__init__()
#         self.cutoff_ratio_z = cutoff_ratio_z
#         self.cutoff_ratio_xy = cutoff_ratio_xy
#         self.filter_type = filter_type
#         self.mode = mode
#         self.epsilon = epsilon
#
#         self.cutoff_ratio2_z = cutoff_ratio2_z
#         self.cutoff_ratio2_xy = cutoff_ratio2_xy
#         self.attenuation_slope_z = attenuation_slope_z
#         self.attenuation_slope_xy = attenuation_slope_xy
#         self.padding_ratio = padding_ratio  # 保存填充比例
#
#         # 验证高通衰减参数的合理性
#         if self.mode == 'high':
#             if self.cutoff_ratio2_z is not None and self.cutoff_ratio2_z < self.cutoff_ratio_z:
#                 raise ValueError("高通模式下，cutoff_ratio2_z 必须大于或等于 cutoff_ratio_z")
#             if self.cutoff_ratio2_xy is not None and self.cutoff_ratio2_xy < self.cutoff_ratio_xy:
#                 raise ValueError("高通模式下，cutoff_ratio2_xy 必须大于或等于 cutoff_ratio_xy")
#
#     def _create_3d_filter(self, D_padded, H_padded, W_padded, device):
#         """
#         生成三维各向异性低通滤波器。
#         这里的 D, H, W 是指填充后的尺寸。
#         """
#         # 生成归一化频率坐标 (范围[-0.5, 0.5])
#         z = torch.linspace(-0.5, 0.5, D_padded, device=device)
#         y = torch.linspace(-0.5, 0.5, H_padded, device=device)
#         x = torch.linspace(-0.5, 0.5, W_padded, device=device)
#
#         # 构建三维网格
#         zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
#
#         # 计算各向异性频率距离
#         distance_z = torch.abs(zz)  # 轴向距离
#         distance_xy = torch.sqrt(yy ** 2 + xx ** 2)  # 横向距离
#
#         # 计算各向截止频率 (相对于 [-0.5, 0.5] 范围内的最大距离)
#         D0_z = self.cutoff_ratio_z * 0.5
#         D0_xy = self.cutoff_ratio_xy * (np.sqrt(0.5 ** 2 + 0.5 ** 2))
#
#         # 生成滤波器掩膜
#         if self.filter_type == 'ideal':
#             mask_z_base = (distance_z <= D0_z).float()
#             mask_xy_base = (distance_xy <= D0_xy).float()
#             mask = mask_z_base * mask_xy_base  # 组合轴向和横向掩膜
#         elif self.filter_type == 'gaussian':
#             sigma_z = D0_z / (np.sqrt(-2 * np.log(0.5))) + self.epsilon
#             sigma_xy = D0_xy / (np.sqrt(-2 * np.log(0.5))) + self.epsilon
#             mask_z_base = torch.exp(-(distance_z ** 2) / (2 * sigma_z ** 2))
#             mask_xy_base = torch.exp(-(distance_xy ** 2) / (2 * sigma_xy ** 2))
#             mask = mask_z_base * mask_xy_base  # 组合轴向和横向掩膜
#         else:
#             raise ValueError(f"Unsupported filter type: {self.filter_type}")
#
#         return mask
#
#     def forward(self, x):
#         """
#         输入: [B, C, D, H, W]
#         输出: 滤波后的图像 [B, C, D, H, W]
#         """
#         B, C, D, H, W = x.shape
#         device = x.device
#
#         # --- 1. 计算填充尺寸并进行填充 ---
#         # 填充尺寸应为整数，且至少为1
#         pad_d = max(1, int(D * self.padding_ratio))
#         pad_h = max(1, int(H * self.padding_ratio))
#         pad_w = max(1, int(W * self.padding_ratio))
#
#         # F.pad 参数顺序：(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
#         # 这里对应 (W_left, W_right, H_top, H_bottom, D_front, D_back)
#         # mode='reflect' 是常见的选择，可以减少边缘的突变，有助于傅里叶变换
#         x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')
#
#         D_padded, H_padded, W_padded = x_padded.shape[-3:]  # 获取填充后的尺寸
#
#         # --- 2. 傅里叶变换 (在填充后的图像上进行) ---
#         x_fft = torch.fft.fftn(x_padded, dim=(-3, -2, -1))
#         x_fft_shift = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))
#
#         # --- 3. 生成滤波器 (基于填充后的尺寸) ---
#         base_lowpass_mask = self._create_3d_filter(D_padded, H_padded, W_padded, x.device)
#
#         # --- 4. 应用滤波 ---
#         if self.mode == 'low':
#             filtered_fft = x_fft_shift * base_lowpass_mask
#         elif self.mode == 'high':
#             high_mask = 1 - base_lowpass_mask
#
#             attenuation_mask = torch.ones_like(high_mask)
#
#             if self.cutoff_ratio2_z is not None or self.cutoff_ratio2_xy is not None:
#                 # 获取频率坐标 (使用填充后的尺寸)
#                 z = torch.linspace(-0.5, 0.5, D_padded, device=device)
#                 y = torch.linspace(-0.5, 0.5, H_padded, device=device)
#                 x = torch.linspace(-0.5, 0.5, W_padded, device=device)
#                 zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
#
#                 distance_z = torch.abs(zz)
#                 distance_xy = torch.sqrt(yy ** 2 + xx ** 2)
#
#                 if self.cutoff_ratio2_z is not None:
#                     D1_z = self.cutoff_ratio2_z * 0.5
#                     k_z = 10 / self.attenuation_slope_z
#                     sigmoid_attenuation_z = 1 / (1 + torch.exp(k_z * (distance_z - D1_z)))
#                     mask_z_attenuation = torch.where(
#                         distance_z <= D1_z,
#                         torch.ones_like(distance_z),
#                         sigmoid_attenuation_z
#                     )
#                 else:
#                     mask_z_attenuation = torch.ones_like(distance_z)
#
#                 if self.cutoff_ratio2_xy is not None:
#                     D1_xy = self.cutoff_ratio2_xy * (np.sqrt(0.5 ** 2 + 0.5 ** 2))
#                     k_xy = 10 / self.attenuation_slope_xy
#                     sigmoid_attenuation_xy = 1 / (1 + torch.exp(k_xy * (distance_xy - D1_xy)))
#                     mask_xy_attenuation = torch.where(
#                         distance_xy <= D1_xy,
#                         torch.ones_like(distance_xy),
#                         sigmoid_attenuation_xy
#                     )
#                 else:
#                     mask_xy_attenuation = torch.ones_like(distance_xy)
#
#                 attenuation_mask = mask_z_attenuation * mask_xy_attenuation
#
#             high_mask = high_mask * attenuation_mask
#             imwrite(r"J:\zzh\deep_deconv\new_stack2\20250416\cached_data(Actx1)\simu\high_mask.tif", high_mask.cpu().numpy())
#             filtered_fft = x_fft_shift * high_mask
#
#         else:
#             raise ValueError("Mode must be 'low' or 'high'")
#
#         # --- 5. 逆傅里叶变换 ---
#         filtered_padded = torch.fft.ifftn(
#             torch.fft.ifftshift(filtered_fft, dim=(-3, -2, -1)),
#             dim=(-3, -2, -1)
#         ).real
#
#         # --- 6. 裁剪回原始尺寸 ---
#         # 计算裁剪的起始和结束索引
#         # 由于我们是两边都pad了pad_d/h/w，所以裁剪时需要从pad_d/h/w开始，到原始尺寸+pad_d/h/w结束
#         cropped_z_start = pad_d
#         cropped_z_end = pad_d + D
#         cropped_y_start = pad_h
#         cropped_y_end = pad_h + H
#         cropped_x_start = pad_w
#         cropped_x_end = pad_w + W
#
#         filtered_cropped = filtered_padded[
#                            :, :,
#                            cropped_z_start:cropped_z_end,
#                            cropped_y_start:cropped_y_end,
#                            cropped_x_start:cropped_x_end
#                            ]
#
#         return filtered_cropped.clamp(min=self.epsilon)


# class FourierLowpass3D(nn.Module):
#     def __init__(self, cutoff_ratio=0.1, filter_type='gaussian', mode='low', epsilon=1e-8):
#         """
#         三维图像二维切片低通/高通滤波模块
#         Args:
#             cutoff_ratio (float): 截止频率比例 (0-1)
#             filter_type (str): 滤波器类型 ['ideal', 'gaussian']
#             mode (str): 'low' 表示低通滤波，'high' 表示高通滤波
#             epsilon (float): 数值稳定性常数
#         """
#         super().__init__()
#         self.cutoff_ratio = cutoff_ratio
#         self.filter_type = filter_type
#         self.mode = mode  # 'low' 或 'high'
#         self.epsilon = epsilon
#
#     def _create_filter(self, H, W, device):
#         """创建二维低通滤波器"""
#         # 生成频率坐标网格，范围 [-0.5, 0.5]
#         y = torch.linspace(-0.5, 0.5, H, device=device)
#         x = torch.linspace(-0.5, 0.5, W, device=device)
#         yy, xx = torch.meshgrid(y, x, indexing='ij')
#         # 计算距离矩阵
#         distance = torch.sqrt(xx ** 2 + yy ** 2)
#         max_distance = distance.max()
#
#         if self.filter_type == 'ideal':
#             mask = (distance <= self.cutoff_ratio * max_distance).float()
#         elif self.filter_type == 'gaussian':
#             sigma = self.cutoff_ratio * max_distance * 2  # 2倍截止频率作为 sigma
#             mask = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
#         else:
#             raise ValueError(f"Unsupported filter type: {self.filter_type}")
#         return mask
#
#     def forward(self, x):
#         """
#         输入: x, 形状为 [B, C, D, H, W]
#         输出: 滤波结果 [B, C, D, H, W]，若 mode 为 'low' 返回低频图像，
#               若 mode 为 'high' 返回高频图像。
#         主要修改：
#             1. 在傅里叶变换前对 x 在 H 和 W 方向进行反射填充；
#             2. 进行滤波后裁剪掉填充区域。
#         """
#         B, C, D, H, W = x.shape
#         device = x.device
#
#         # 设定反射填充大小，这里以 H、W 各取 10% 的尺寸为例（可根据需要调整）
#         pad_h = int(H * 0.1)
#         pad_w = int(W * 0.1)
#
#         # 对 x 在 H 和 W 方向进行反射填充。对于5D张量，F.pad的参数顺序为:
#         # (padW_left, padW_right, padH_top, padH_bottom, padD_front, padD_back)
#         # 这里只在 H 和 W 方向填充
#         x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode='reflect')
#         H_pad = H + 2 * pad_h
#         W_pad = W + 2 * pad_w
#
#         # 将填充后的3D图像看作多个二维切片，每个切片进行二维傅里叶变换
#         x_2d = x_pad.reshape(-1, H_pad, W_pad)  # [B * C * D, H_pad, W_pad]
#         x_fft = torch.fft.fft2(x_2d)
#         x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
#
#         # 创建滤波器（低通滤波器），基于填充后的尺寸
#         mask = self._create_filter(H_pad, W_pad, device)  # shape: [H_pad, W_pad]
#
#         # 如果选择高通模式，则取掩膜的互补
#         if self.mode == 'high':
#             mask = 1 - mask
#         elif self.mode != 'low':
#             raise ValueError("Mode must be 'low' or 'high'.")
#
#         # # 可选：保存滤波器用于调试
#         # tifffile.imwrite(r"J:\zzh\deep_deconv\new_stack2\20250306\cached_data(微管4)\simu\mask.tif",
#         #                  mask.cpu().numpy())
#
#         # 应用滤波器
#         filtered_fft = x_fft_shift * mask
#         # 逆傅里叶变换
#         filtered = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft, dim=(-2, -1)), dim=(-2, -1)).real
#         # 裁剪掉填充部分，恢复原始 H, W
#         filtered_cropped = filtered[:, pad_h:pad_h+H, pad_w:pad_w+W]
#         # 恢复为原始形状 [B, C, D, H, W]
#         filtered_3d = filtered_cropped.view(B, C, D, H, W)
#         # return filtered_3d
#         return filtered_3d.clamp(min=self.epsilon)


class FourierLowpass3D(nn.Module):
    def __init__(self, cutoff_ratio=0.1, filter_type='gaussian', mode='low',
                 cutoff_ratio2=None,
                 attenuation_slope=0.1,
                 epsilon=1e-8):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.filter_type = filter_type
        self.mode = mode
        self.epsilon = epsilon

        self.cutoff_ratio2 = cutoff_ratio2
        self.attenuation_slope = attenuation_slope

        if self.mode == 'high' and self.cutoff_ratio2 is not None:
            if self.cutoff_ratio2 <= self.cutoff_ratio:
                raise ValueError("对于高通滤波的高频衰减，cutoff_ratio2 必须大于 cutoff_ratio。")

    def _create_filter(self, H, W, device):
        """创建二维滤波器掩膜"""
        y = torch.linspace(-0.5, 0.5, H, device=device)
        x = torch.linspace(-0.5, 0.5, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        distance = torch.sqrt(xx ** 2 + yy ** 2)
        max_distance = distance.max()

        D0 = self.cutoff_ratio * max_distance + self.epsilon

        if self.filter_type == 'ideal':
            base_mask = (distance <= D0).float()
        elif self.filter_type == 'gaussian':
            sigma_base = D0 / (np.sqrt(-2 * np.log(0.5)))
            # sigma_base = D0 * 0.5
            base_mask = torch.exp(-(distance ** 2) / (2 * sigma_base ** 2))
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")

        if self.mode == 'high':
            final_mask = 1 - base_mask

            if self.cutoff_ratio2 is not None:
                D1 = self.cutoff_ratio2 * max_distance + self.epsilon
                attenuation_sigma = (max_distance - D1) * self.attenuation_slope
                if attenuation_sigma == 0:
                    attenuation_sigma = self.epsilon

                mask_attenuation = torch.exp(-((distance - D1) ** 2) / (2 * attenuation_sigma ** 2))
                mask_attenuation = torch.where(distance <= D1, torch.ones_like(distance), mask_attenuation)

                final_mask = final_mask * mask_attenuation

        elif self.mode == 'low':
            final_mask = base_mask
        else:
            raise ValueError("Mode must be 'low' or 'high'.")

        # imwrite(r"J:\zzh\deep_deconv\new_stack2\20250416\cached_data(Actx1)\simu\high_mask.tif", final_mask.cpu().numpy())
        return final_mask

    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device

        pad_h = int(H * 0.2)
        pad_w = int(W * 0.2)

        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode='reflect')
        H_pad = H + 2 * pad_h
        W_pad = W + 2 * pad_w

        x_2d = x_pad.view(-1, H_pad, W_pad)

        x_fft = torch.fft.fft2(x_2d, dim=(-2, -1))
        # 保持此处的 fftshift，确保 x_fft_shift 的零频率在中心
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # _create_filter 现在返回的 mask 也是零频率在中心
        mask = self._create_filter(H_pad, W_pad, device)

        filtered_fft = x_fft_shift * mask

        # 逆傅里叶变换前，将零频率移回角落
        filtered = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft, dim=(-2, -1)), dim=(-2, -1)).real

        filtered_cropped = filtered[:, pad_h:pad_h + H, pad_w:pad_w + W]

        filtered_3d = filtered_cropped.view(B, C, D, H, W)

        # return filtered_3d.clamp(min=self.epsilon)
        return filtered_3d.clamp(min=0)


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, mse_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, input, target):
        return self.l1_weight * self.l1(input, target) + self.mse_weight * self.mse(input, target)


def augment_3d_batch(input_tensor):
    """
    对一个 [batch_num, 1, channel, D, H, W] 维度的三维图像张量进行数据增强。
    每个三维图像（batch_num 中的一个样本）都会独立进行以下随机增强：
    1. 沿 H 和 W 维度进行 0°, 90°, 180°, 270° 随机旋转。
    2. 以一定概率进行水平翻转。
    3. 以一定概率进行垂直翻转。
    4. 以一定概率进行沿 Z 轴（D 维度）翻转。

    Args:
        input_tensor (torch.Tensor): 输入张量，维度为 [batch_num, 1, channel, D, H, W]。
                                    其中 channel 预期为 1。

    Returns:
        torch.Tensor: 增强后的张量，维度与输入张量相同。
    """
    if input_tensor.dim() != 6:
        raise ValueError(f"Input tensor must have 6 dimensions [batch_num, 1, channel, D, H, W], but got {input_tensor.dim()}")
    if input_tensor.shape[1] != 1 or input_tensor.shape[2] != 1:
        print("Warning: Expected batch_size=1 and channel=1. Processing with given dimensions.")

    augmented_tensors = []

    # 遍历 batch_num 维度
    for i in range(input_tensor.shape[0]):
        # 提取当前三维图像：[1, 1, 1, D, H, W]
        single_3d_image = input_tensor[i:i+1, :, :, :, :, :]
        # 移除 batch_size 和 channel 维度，使其成为 [1, D, H, W] 便于处理
        single_3d_image_squeeze = single_3d_image.squeeze(0).squeeze(0) # 变为 [1, D, H, W]

        # 1. 随机旋转 (0°, 90°, 180°, 270°)
        # 对 H 和 W 维度进行旋转，这意味着对 D 维度的每一张 2D slice 进行相同的旋转
        # TF.rotate 需要 [..., H, W] 或 [H, W] 的输入
        angle = random.choice([0, 90, 180, 270])
        rotated_image = TF.rotate(single_3d_image_squeeze, angle)

        # 2. 以一定概率进行水平翻转
        if random.random() > 0.5: # 50% 概率
            rotated_image = TF.hflip(rotated_image)

        # 3. 以一定概率进行垂直翻转
        if random.random() > 0.5: # 50% 概率
            rotated_image = TF.vflip(rotated_image)

        # 4. 以一定概率进行沿 Z 轴（D 维度）翻转
        if random.random() > 0.5: # 50% 概率
            # 沿 D 维度翻转
            rotated_image = torch.flip(rotated_image, dims=[1]) # dims=[1] 对应 D 维度

        # 将处理后的三维图像恢复到原始的 [1, 1, 1, D, H, W] 格式
        augmented_single_3d_image = rotated_image.unsqueeze(0).unsqueeze(0)
        augmented_tensors.append(augmented_single_3d_image)

    # 将所有增强后的三维图像拼接回原始的 batch_num 维度
    return torch.cat(augmented_tensors, dim=0)


class MedianFilter3D(nn.Module):
    """
    对 [batch_size, 1, D, H, W] 维度的三维图像张量中的每一个二维切片
    (即 H x W 维度) 进行中值滤波。

    Args:
        kernel_size (int or tuple): 中值滤波核的大小。
                                    如果为整数，则核为 (kernel_size, kernel_size)。
                                    如果为元组 (kh, kw)，则分别指定高度和宽度。
        padding (str or int or tuple, optional): 填充方式。
                                                 'same' 会在输入周围填充，使输出大小与输入大小相同。
                                                 如果为整数，则在所有边填充相同数量的零。
                                                 如果为元组 (pad_h, pad_w)，则分别指定高度和宽度的填充。
                                                 默认为 'same'。
    """

    def __init__(self, kernel_size, padding='same'):
        super().__init__()
        # kornia 的 median_blur 要求 kernel_size 为 tuple (k_h, k_w)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError("kernel_size must be an int or a tuple of two ints.")

        self.padding_mode = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，维度为 [batch_size, 1, D, H, W]。

        Returns:
            torch.Tensor: 中值滤波后的张量，维度与输入相同。
        """
        if x.dim() != 5:
            raise ValueError(f"Input tensor must have 5 dimensions [batch_size, 1, D, H, W], but got {x.dim()}")
        if x.shape[1] != 1:
            print(f"Warning: Expected 1 channel, but got {x.shape[1]}. Median filter will be applied per channel.")

        batch_size, num_channels, D, H, W = x.shape

        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(batch_size * D, num_channels, H, W)

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError("kornia.filters.median_blur requires odd kernel_size for 'same' output shape.")

        filtered_x_reshaped = kornia_filters.median_blur(x_reshaped, self.kernel_size)

        # 将结果恢复到原始形状 [batch_size, 1, D, H, W]
        filtered_x = filtered_x_reshaped.reshape(batch_size, D, num_channels, H, W).permute(0, 2, 1, 3, 4)

        return filtered_x


def _create_3d_split_sigmoid_lowpass_mask(D, H, W,
                                          cutoff_ratio_axial, transition_width_axial,
                                          cutoff_ratio_lateral, transition_width_lateral,
                                          device, epsilon=1e-8):
    """
    创建三维 Sigmoid 低通滤波掩码，支持轴向和横向独立截止。
    当距离小于截止频率时为1，当大于截止频率时进行较大坡度的衰减（类似Sigmoid函数）。

    Args:
        D (int): 深度维度大小。
        H (int): 高度维度大小。
        W (int): 宽度维度大小。
        cutoff_ratio_axial (float): 轴向 (Z) 截止频率的相对比例 (0-0.5)。
        transition_width_axial (float): 轴向 Sigmoid 函数的过渡宽度。
        cutoff_ratio_lateral (float): 横向 (Y, X) 截止频率的相对比例 (0-0.5)。
        transition_width_lateral (float): 横向 Sigmoid 函数的过渡宽度。
        device (torch.device): 创建掩码的设备。
        epsilon (float): 数值稳定性常数。
    Returns:
        torch.Tensor: 三维 Sigmoid 低通滤波掩码。
    """
    # 创建频率坐标 (中心为 0)
    d_coords = torch.linspace(-0.5, 0.5, D, device=device)
    h_coords = torch.linspace(-0.5, 0.5, H, device=device)
    w_coords = torch.linspace(-0.5, 0.5, W, device=device)

    # 扩展为三维网格
    ddd, hhh, www = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')

    # --- 轴向 (Z) 距离和掩码 ---
    distance_axial = torch.abs(ddd)  # 轴向距离只需考虑绝对值
    max_distance_axial = distance_axial.max() + epsilon
    D0_axial = cutoff_ratio_axial * max_distance_axial
    k_axial = 10 / transition_width_axial  # 控制陡峭程度

    # 轴向 Sigmoid 衰减掩码
    # 当 distance_axial <= D0_axial 时，mask_axial_attenuation 为 1
    # 当 distance_axial > D0_axial 时，开始 Sigmoid 衰减
    mask_axial_attenuation = torch.where(
        distance_axial <= D0_axial,
        torch.ones_like(distance_axial),
        1 / (1 + torch.exp(k_axial * (distance_axial - D0_axial)))
    )

    # --- 横向 (Y, X) 距离和掩码 ---
    distance_lateral = torch.sqrt(hhh ** 2 + www ** 2)  # 横向径向距离
    max_distance_lateral = distance_lateral.max() + epsilon
    D0_lateral = cutoff_ratio_lateral * max_distance_lateral
    k_lateral = 10 / transition_width_lateral  # 控制陡峭程度

    # 横向 Sigmoid 衰减掩码
    # 当 distance_lateral <= D0_lateral 时，mask_lateral_attenuation 为 1
    # 当 distance_lateral > D0_lateral 时，开始 Sigmoid 衰减
    mask_lateral_attenuation = torch.where(
        distance_lateral <= D0_lateral,
        torch.ones_like(distance_lateral),
        1 / (1 + torch.exp(k_lateral * (distance_lateral - D0_lateral)))
    )

    # 组合轴向和横向掩码 (取乘积，确保两者都通过才有效)
    final_mask = mask_axial_attenuation * mask_lateral_attenuation

    return final_mask
