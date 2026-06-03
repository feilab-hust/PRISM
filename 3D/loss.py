import torch
from torch import nn
import torch.nn.functional as F
from torch.fft import fftn, fft2
import torchvision.models as models

from utils import SSIM3D, TVLoss, HessianLoss, L1SparsityLoss, DarkChannelPrior3D, normalize_to_01, CombinedLoss, _create_3d_split_sigmoid_lowpass_mask


class CustomLoss_single(nn.Module):
    def __init__(self, spatial_weight=1, ssim_weight=0.04, fourier_weight=0.05, loss_type='both', fourier_flag=False,
                 fourier_magnitude_cutoff_ratio_axial=0.4,      # 轴向截止比例
                 fourier_magnitude_transition_width_axial=0.25, # 轴向 Sigmoid 过渡宽度
                 fourier_magnitude_cutoff_ratio_lateral=0.4,    # 横向截止比例
                 fourier_magnitude_transition_width_lateral=0.25, D=64, H=128, W=128, device=torch.device('cpu')):
        super(CustomLoss_single, self).__init__()
        self.spatial_weight = spatial_weight
        self.ssim_weight = ssim_weight
        self.fourier_weight= fourier_weight
        self.fourier_flag = fourier_flag
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = CombinedLoss()
        self.SSIM = SSIM3D()
        self.mask = _create_3d_split_sigmoid_lowpass_mask(D, H, W, fourier_magnitude_cutoff_ratio_axial, fourier_magnitude_transition_width_axial, fourier_magnitude_cutoff_ratio_lateral, fourier_magnitude_transition_width_lateral, device).unsqueeze(0).unsqueeze(0)

    def forward(self, predicted_blur_x, blurred_x):
        # 空间域损失
        spatial_loss_x = self.loss_fn(predicted_blur_x, blurred_x)

        # SSIM 损失
        ssim_loss_x = 1 - self.SSIM(predicted_blur_x, blurred_x)

        # 频谱损失
        if self.fourier_flag:
            fourier_loss_x = self.compute_3d_fourier_loss(predicted_blur_x, blurred_x)
        else:
            fourier_loss_x = torch.tensor(0)

        # 总损失 = 空间域损失 + 傅里叶域损失 + SSIM 损失
        total_loss = (self.spatial_weight * spatial_loss_x +
                      self.ssim_weight * ssim_loss_x +
                      self.fourier_weight * fourier_loss_x)

        return total_loss, spatial_loss_x, ssim_loss_x, fourier_loss_x

    def compute_3d_fourier_loss(self, pred, target):
        pred_fft = torch.fft.fftshift(torch.fft.fftn(pred, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        target_fft = torch.fft.fftshift(torch.fft.fftn(target, dim=(-3, -2, -1)), dim=(-3, -2, -1))

        magnitude_pred = torch.log(torch.abs(pred_fft) + 1 + 1e-8) * self.mask
        magnitude_target = torch.log(torch.abs(target_fft) + 1 + 1e-8) * self.mask

        return self.loss_fn(magnitude_pred, magnitude_target)


class FinalLoss_single(nn.Module):
    def __init__(self, spatial_weight=1, high_weight=1, ssim_weight=0.04, fourier_weight=0.05, fourier_weight_high=0.05, hessian_weight=10, tv_weight=(10, 10),
                 sparsity_weight=0.1, dark_weight=0.5, loss_type='L1', window_size=7, alpha=0.5, mode='2d', fourier_flag=False,
                 fourier_magnitude_cutoff_ratio_axial=0.4,  # 轴向截止比例
                 fourier_magnitude_transition_width_axial=0.25,  # 轴向 Sigmoid 过渡宽度
                 fourier_magnitude_cutoff_ratio_lateral=0.4,  # 横向截止比例
                 fourier_magnitude_transition_width_lateral=0.25, D=64, H=128, W=128, device=torch.device('cpu'),
                 fourier_magnitude_cutoff_ratio_axial2=0.75,  # 轴向截止比例
                 fourier_magnitude_transition_width_axial2=0.25,  # 轴向 Sigmoid 过渡宽度
                 fourier_magnitude_cutoff_ratio_lateral2=0.75,  # 横向截止比例
                 fourier_magnitude_transition_width_lateral2=0.25):
        super(FinalLoss_single, self).__init__()
        self.custom_loss = CustomLoss_single(spatial_weight, ssim_weight, fourier_weight, loss_type, fourier_flag,
                                             fourier_magnitude_cutoff_ratio_axial,  # 轴向截止比例
                                             fourier_magnitude_transition_width_axial,  # 轴向 Sigmoid 过渡宽度
                                             fourier_magnitude_cutoff_ratio_lateral,  # 横向截止比例
                                             fourier_magnitude_transition_width_lateral, D, H, W, device)
        self.custom_loss_high = CustomLoss_single(spatial_weight, ssim_weight, fourier_weight_high, loss_type, fourier_flag,
                                             fourier_magnitude_cutoff_ratio_axial2,  # 轴向截止比例
                                             fourier_magnitude_transition_width_axial2,  # 轴向 Sigmoid 过渡宽度
                                             fourier_magnitude_cutoff_ratio_lateral2,  # 横向截止比例
                                             fourier_magnitude_transition_width_lateral2, D, H, W, device)
        self.tv_loss = TVLoss()
        self.hessian_loss = HessianLoss()
        self.l1_sparsity_loss = L1SparsityLoss()
        self.dark_loss = DarkChannelPrior3D(window_size=window_size, alpha=alpha, mode=mode)

        self.spatial_weight = spatial_weight
        self.ssim_weight = ssim_weight
        self.fourier_weight = fourier_weight
        self.high_weight = high_weight
        self.hessian_weight = hessian_weight
        self.tv_weight = tv_weight
        self.sparsity_weight = sparsity_weight
        self.dark_weight = dark_weight

    def forward(self, predicted_raw, predicted_raw_high, blurred, blurred_high, y_pred, predicted_blur):
        total_loss, spatial_loss_low, ssim_loss_low, fourier_loss_low = self.custom_loss(predicted_raw, blurred)
        total_loss_high, spatial_loss_high, ssim_loss_high, fourier_loss_high = self.custom_loss_high(predicted_raw_high, blurred_high)
        total_loss += total_loss_high * self.high_weight
        total_loss_normal = total_loss.item()

        # 计算 TV 和 Hessian 损失
        tv_loss_xy, tv_loss_z = self.tv_loss(y_pred)

        hessian_loss = self.hessian_loss(y_pred)

        sparsity_loss = self.l1_sparsity_loss(y_pred)

        dark_loss = self.dark_loss(predicted_blur)

        # 计算总损失
        total_loss += (self.hessian_weight * hessian_loss +
                       self.tv_weight[0] * tv_loss_xy +
                       self.tv_weight[1] * tv_loss_z +
                       self.sparsity_weight * sparsity_loss +
                       self.dark_weight * dark_loss)

        total_loss_normal += (self.hessian_weight * hessian_loss.item() +
                              self.tv_weight[0] * tv_loss_xy.item() +
                              self.tv_weight[1] * tv_loss_z.item() +
                              self.sparsity_weight * sparsity_loss.item() +
                              self.dark_weight * dark_loss.item())

        return total_loss_normal, total_loss, spatial_loss_low, ssim_loss_low, fourier_loss_low, spatial_loss_high, ssim_loss_high, fourier_loss_high, tv_loss_xy, tv_loss_z, hessian_loss, sparsity_loss, dark_loss


class FinalLoss_single_blur(nn.Module):
    def __init__(self, spatial_weight=1, ssim_weight=0.04, fourier_weight=0.05, hessian_weight=10, tv_weight=(10, 10),
                 sparsity_weight=0.1, dark_weight=0.5, loss_type='L1', window_size=7, alpha=0.5, mode='2d', fourier_flag=False,
                 fourier_magnitude_cutoff_ratio_axial=0.4,  # 轴向截止比例
                 fourier_magnitude_transition_width_axial=0.25,  # 轴向 Sigmoid 过渡宽度
                 fourier_magnitude_cutoff_ratio_lateral=0.4,  # 横向截止比例
                 fourier_magnitude_transition_width_lateral=0.25, D=64, H=128, W=128, device=torch.device('cpu')):
        super(FinalLoss_single_blur, self).__init__()
        self.custom_loss = CustomLoss_single(spatial_weight, ssim_weight, fourier_weight, loss_type, fourier_flag,
                                             fourier_magnitude_cutoff_ratio_axial,  # 轴向截止比例
                                             fourier_magnitude_transition_width_axial,  # 轴向 Sigmoid 过渡宽度
                                             fourier_magnitude_cutoff_ratio_lateral,  # 横向截止比例
                                             fourier_magnitude_transition_width_lateral, D, H, W, device
                                             )
        self.tv_loss = TVLoss()
        self.hessian_loss = HessianLoss()
        self.l1_sparsity_loss = L1SparsityLoss()
        self.dark_loss = DarkChannelPrior3D(window_size=window_size, alpha=alpha, mode=mode)

        self.spatial_weight = spatial_weight
        self.ssim_weight = ssim_weight
        self.hessian_weight = hessian_weight
        self.tv_weight = tv_weight
        self.sparsity_weight = sparsity_weight
        self.dark_weight = dark_weight

    def forward(self, predicted_blur, blurred, y_pred):
        total_loss, spatial_loss, ssim_loss, fourier_loss = self.custom_loss(predicted_blur, blurred)
        total_loss_normal = total_loss.item()

        # 计算 TV 和 Hessian 损失
        tv_loss_xy, tv_loss_z = self.tv_loss(y_pred)

        hessian_loss = self.hessian_loss(y_pred)

        sparsity_loss = self.l1_sparsity_loss(y_pred)

        dark_loss = self.dark_loss(predicted_blur)

        # 计算总损失
        total_loss += (self.hessian_weight * hessian_loss +
                       self.tv_weight[0] * tv_loss_xy +
                       self.tv_weight[1] * tv_loss_z +
                       self.sparsity_weight * sparsity_loss +
                       self.dark_weight * dark_loss)

        total_loss_normal += (self.hessian_weight * hessian_loss.item() +
                              self.tv_weight[0] * tv_loss_xy.item() +
                              self.tv_weight[1] * tv_loss_z.item() +
                              self.sparsity_weight * sparsity_loss.item() +
                              self.dark_weight * dark_loss.item())

        return total_loss_normal, total_loss, spatial_loss, ssim_loss, fourier_loss, tv_loss_xy, tv_loss_z, hessian_loss, sparsity_loss, dark_loss
