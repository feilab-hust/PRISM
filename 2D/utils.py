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

    return (tensor - min_val) / (max_val - min_val + 1e-8)  # 避免除零


# SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


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
            x_diff = torch.pow(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:], 2).mean()
            y_diff = torch.pow(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :], 2).mean()
        else:
            x_diff = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]).mean()
            y_diff = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]).mean()

        # Return the total TV loss
        return x_diff + y_diff


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
        y_diff = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]
        x_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]

        # Second-order differences (Hessian components)
        if self.use_l2:
            yy_diff = torch.pow(y_diff[:, :, :-1, :] - y_diff[:, :, 1:, :], 2).mean()
            xx_diff = torch.pow(x_diff[:, :, :, :-1] - x_diff[:, :, :, 1:], 2).mean()
            xy_diff = torch.pow(x_diff[:, :, :-1, :] - x_diff[:, :, 1:, :], 2).mean()
        else:
            yy_diff = torch.abs(y_diff[:, :, :-1, :] - y_diff[:, :, 1:, :]).mean()
            xx_diff = torch.abs(x_diff[:, :, :, :-1] - x_diff[:, :, :, 1:]).mean()
            xy_diff = torch.abs(x_diff[:, :, :-1, :] - x_diff[:, :, 1:, :]).mean()

        # Return the sum of all second-order differences
        return yy_diff + xx_diff + 2 * xy_diff


class L1SparsityLoss(nn.Module):
    def __init__(self):
        super(L1SparsityLoss, self).__init__()

    def forward(self, y_pred):
        y_pred = torch.abs(y_pred)
        y_pred_normalized = y_pred / (y_pred.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return torch.mean(y_pred_normalized)


class FourierMagnitudeLoss(nn.Module):
    def __init__(self, alpha = 4.0):
        super(FourierMagnitudeLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred):
        slice_fft = torch.fft.fftshift(torch.fft.fft2(y_pred))

        # 计算频谱模
        magnitude = torch.abs(slice_fft)
        magnitude = torch.log(magnitude + 1 + 1e-8)

        # 归一化（按切片的最大值）
        # slice_magnitude_normalized = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        slice_magnitude_normalized = normalize_to_01(magnitude)
        # slice_magnitude_normalized = (torch.sigmoid(self.alpha * slice_magnitude_normalized) - 0.5) / (1/(1 + exp(-self.alpha)) - 0.5)

        # 计算损失 (1 - 归一化频谱模均值)
        loss = torch.mean(1 - slice_magnitude_normalized)

        return loss


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


class DarkChannelPrior2D(nn.Module):
    def __init__(self,
                 window_size,
                 alpha=0.5,
                 eps=1e-8):
        """
        基于最小池化的暗通道先验正则项
        Args:
            window_size : 池化窗口尺寸
            alpha (float): 惩罚指数系数
            eps (float): 数值稳定性常数
        """
        super().__init__()
        self.window_size = window_size
        self.alpha = alpha
        self.eps = eps

        self.pool = MinPool2d(window_size, stride=1, padding=0)

    def forward(self, y_pred):
        """
        Args:
            y_pred: 预测的高分辨率图像 [B,C,D,H,W]
        Returns:
            loss: 正则项损失值
        """
        # 提取暗通道
        y_pred = torch.abs(y_pred)
        dark = self.pool(y_pred)

        dark += self.eps

        # 计算归一化暗通道损失
        # max_val = y_pred.amax(dim=(2, 3, 4), keepdim=True) + self.eps  # 防止除零
        # dark_norm = dark / max_val
        dark_norm = dark
        return torch.mean(torch.pow(dark_norm, self.alpha))


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


def fft_conv2d(image, kernel, padding_mode='reflect'):
    """
    基于FFT的三维卷积实现，修正非2的整数幂尺寸输入时的伪影问题。
    通过预先对 kernel 进行 ifftshift 操作，实现核中心与零频对应，
    从而在逆FFT后无需额外 roll 即可还原出与 F.conv2d 相同的结果。

    :param image: 输入图像 [B, C, H, W]
    :param kernel: 卷积核 [1, 1, kH, kW]
    :param padding_mode: 边缘填充方式
    :return: 卷积结果 [B, C, H, W]
    """
    B, C, H, W = image.shape
    kH, kW = kernel.shape[-2:]

    # 计算理论卷积输出尺寸（即线性卷积尺寸：N + k - 1）
    target_size = (H + kH - 1, W + kW - 1)

    # 为了提高 FFT 计算速度，将每个轴的尺寸扩展到最近的2的幂次
    fft_size = tuple([2 ** int(torch.ceil(torch.log2(torch.tensor(s, dtype=torch.float32))).item())
                      for s in target_size])

    # --------------------
    # 对 kernel 进行填充，使其尺寸为 fft_size，然后预先做 ifftshift 以对齐核中心
    pad_w_kernel = (fft_size[1] - kW) // 2
    pad_h_kernel = (fft_size[0] - kH) // 2
    kernel_padded = F.pad(kernel,
                          (pad_w_kernel, fft_size[1] - kW - pad_w_kernel,
                           pad_h_kernel, fft_size[0] - kH - pad_h_kernel))
    # 预先调整 kernel，使得其中心移到左上角（零频处）
    kernel_padded = my_ifftshift(kernel_padded, dims=(-2, -1))

    # --------------------
    # 对图像进行填充，使其尺寸为 fft_size（采用指定的边缘填充方式）
    pad_w_img = (fft_size[1] - W) // 2
    pad_h_img = (fft_size[0] - H) // 2
    image_padded = F.pad(image,
                         (pad_w_img, fft_size[1] - W - pad_w_img,
                          pad_h_img, fft_size[0] - H - pad_h_img),
                         mode=padding_mode)

    # --------------------
    # FFT 变换
    image_fft = torch.fft.rfftn(image_padded, s=fft_size, dim=(-2, -1))
    kernel_fft = torch.fft.rfftn(kernel_padded, s=fft_size, dim=( -2, -1))

    # 频域相乘
    result_fft = image_fft * kernel_fft

    # 逆 FFT 得到卷积结果（此时已经是线性卷积的正确排列）
    result = torch.fft.irfftn(result_fft, s=fft_size, dim=(-2, -1))

    # --------------------
    # 裁剪出有效区域：恢复为原图尺寸（假定 F.conv2d 的输出范围）
    crop_h = slice(pad_h_img, pad_h_img + H)
    crop_w = slice(pad_w_img, pad_w_img + W)
    result = result[..., crop_h, crop_w]

    return result.real


class FourierLowpass2D(nn.Module):
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

        return final_mask

    def forward(self, x):
        B, C, H, W = x.shape
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

        filtered_3d = filtered_cropped.view(B, C, H, W)

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


def augment_2d_batch(input_tensor):
    """
    对一个 [batch_num, 1, channel, H, W] 维度的二维图像张量进行数据增强。
    每个二维图像（batch_num 中的一个样本）都会独立进行以下随机增强：
    1. 沿 H 和 W 维度进行 0°, 90°, 180°, 270° 随机旋转。
    2. 以一定概率进行水平翻转。
    3. 以一定概率进行垂直翻转。

    Args:
        input_tensor (torch.Tensor): 输入张量，维度为 [batch_num, 1, channel, H, W]。
                                    其中 channel 预期为 1。

    Returns:
        torch.Tensor: 增强后的张量，维度与输入张量相同。
    """
    if input_tensor.dim() != 5:
        raise ValueError(f"Input tensor must have 6 dimensions [batch_num, 1, channel, H, W], but got {input_tensor.dim()}")
    if input_tensor.shape[1] != 1 or input_tensor.shape[2] != 1:
        print("Warning: Expected batch_size=1 and channel=1. Processing with given dimensions.")

    augmented_tensors = []

    # 遍历 batch_num 维度
    for i in range(input_tensor.shape[0]):
        # 提取当前三维图像：[1, 1, 1, H, W]
        single_2d_image = input_tensor[i:i+1, :, :, :, :]
        # 移除 batch_size 和 channel 维度，使其成为 [1, H, W] 便于处理
        single_2d_image_squeeze = single_2d_image.squeeze(0).squeeze(0) # 变为 [1, H, W]

        # 1. 随机旋转 (0°, 90°, 180°, 270°)
        # 对 H 和 W 维度进行旋转，这意味着对每一张 2D slice 进行相同的旋转
        # TF.rotate 需要 [..., H, W] 或 [H, W] 的输入
        angle = random.choice([0, 90, 180, 270])
        rotated_image = TF.rotate(single_2d_image_squeeze, angle)

        # 2. 以一定概率进行水平翻转
        if random.random() > 0.5: # 50% 概率
            rotated_image = TF.hflip(rotated_image)

        # 3. 以一定概率进行垂直翻转
        if random.random() > 0.5: # 50% 概率
            rotated_image = TF.vflip(rotated_image)

        # 将处理后的三维图像恢复到原始的 [1, 1, 1, H, W] 格式
        augmented_single_2d_image = rotated_image.unsqueeze(0).unsqueeze(0)
        augmented_tensors.append(augmented_single_2d_image)

    # 将所有增强后的三维图像拼接回原始的 batch_num 维度
    return torch.cat(augmented_tensors, dim=0)


class MedianFilter3D(nn.Module):
    """
    对 [batch_size, 1, H, W] 维度的三维图像张量中的每一个二维切片
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
        if x.dim() != 4:
            raise ValueError(f"Input tensor must have 5 dimensions [batch_size, 1, H, W], but got {x.dim()}")
        if x.shape[1] != 1:
            print(f"Warning: Expected 1 channel, but got {x.shape[1]}. Median filter will be applied per channel.")

        batch_size, num_channels, H, W = x.shape

        x_reshaped = x

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError("kornia.filters.median_blur requires odd kernel_size for 'same' output shape.")

        filtered_x_reshaped = kornia_filters.median_blur(x_reshaped, self.kernel_size)

        # 将结果恢复到原始形状 [batch_size, 1, H, W]
        filtered_x = filtered_x_reshaped.reshape(batch_size, num_channels, H, W)

        return filtered_x


def _create_2d_split_sigmoid_lowpass_mask(H, W,
                                          cutoff_ratio_lateral, transition_width_lateral,
                                          device, epsilon=1e-8):
    """
    创建二维 Sigmoid 低通滤波掩码。
    当距离小于截止频率时为1，当大于截止频率时进行较大坡度的衰减（类似Sigmoid函数）。

    Args:
        H (int): 高度维度大小。
        W (int): 宽度维度大小。
        cutoff_ratio_lateral (float): 横向 (Y, X) 截止频率的相对比例 (0-0.5)。
        transition_width_lateral (float): 横向 Sigmoid 函数的过渡宽度。
        device (torch.device): 创建掩码的设备。
        epsilon (float): 数值稳定性常数。
    Returns:
        torch.Tensor: 二维 Sigmoid 低通滤波掩码。
    """
    # 创建频率坐标 (中心为 0)
    h_coords = torch.linspace(-0.5, 0.5, H, device=device)
    w_coords = torch.linspace(-0.5, 0.5, W, device=device)

    # 扩展为二维网格
    hhh, www = torch.meshgrid(h_coords, w_coords, indexing='ij')

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

    final_mask = mask_lateral_attenuation

    return final_mask
