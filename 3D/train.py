import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from data_utils import get_3d_dataloaders_single
from model import RestorationNetwork3d_subback2
from loss import FinalLoss_single, FinalLoss_single_blur
from utils import normalize_to_01, augment_3d_batch


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def preprocess_image_3d(image_path, device):
    image = np.float32(imread(image_path))
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
    image = image / torch.max(image)
    return image


def preprocess_to_tensor(model_pre, train_loader, psf_x, psf_y, data_final_save_path, scale=1, scale_axial=1, padding_size=(0, 0, 0)):
    nnnn = 1
    batch_num, batch_s, channel, depth, height, width = train_loader.shape
    train_loader = train_loader.view(batch_num*batch_s, 1, channel, depth, height, width)
    if padding_size != (0, 0, 0):
        train_loader0 = torch.zeros((batch_num*batch_s, 1, channel, depth+2*padding_size[0], height+2*padding_size[1], width+2*padding_size[2]), device=train_loader.device)
        for i in range(batch_num*batch_s):
            train_loader0[i] = F.pad(train_loader[i], (
                padding_size[2], padding_size[2], padding_size[1], padding_size[1], padding_size[0],
                padding_size[0]), 'reflect')
        train_loader = train_loader0
    # 设置模型为评估模式
    model_pre.eval()

    # 用于存储所有批次的输出
    all_outputs = []

    with torch.no_grad():
        for batch in train_loader:
            inputs = batch

            # 使用预训练模型处理数据
            if channel == 2:
                _, _, outputs = model_pre(inputs[:, 0, :, :, :].unsqueeze(1), inputs[:, 1, :, :, :].unsqueeze(1), psf_x, psf_y)  # 输出形状 [batch_size, 1, depth, height, width]
            else:
                if model_pre.back_flag:
                    _, _, outputs, _, blur_low, back = model_pre(inputs[:, 0, :, :, :].unsqueeze(1), psf_x, padding_mode=padding_mode)
                    imwrite(os.path.join(data_final_save_path, "back", str(nnnn) + ".tif"), back.cpu().numpy())
                else:
                    predicted_blur, outputs = model_pre(inputs[:, 0, :, :, :].unsqueeze(1), psf_x, padding_mode=padding_mode)
            nnnn += 1

            # 归一化至 [0, 1] 范围
            # outputs = normalize_to_01(outputs)
            # background = torch.quantile(outputs, q=0)
            # outputs = outputs - background
            # outputs[outputs<0] = 0
            # outputs = normalize_to_01(outputs)

            outputs = outputs.unsqueeze(0)

            # 收集到列表中
            all_outputs.append(outputs.cpu())  # 返回到 CPU 以便拼接

    # 拼接所有批次为一个张量
    result_tensor = torch.cat(all_outputs, dim=0)  # [batch_num, batch_size, 1, target_depth, target_height, target_width]
    result_tensor = result_tensor.view(batch_num, batch_s, 1, depth * scale_axial, height * scale, width * scale)
    return result_tensor


def train(model, batch_size, train_loader, psf_x, criterion, optimizer, device, padding_size=(0, 0, 0), back_flag=True):
    # 将训练数据打乱
    batch_num, batch_s, channel, depth, height, width = train_loader.shape
    train_loader2 = train_loader.view(batch_num * batch_s, 1, channel, depth, height, width)
    index = torch.randperm(train_loader2.shape[0])    # 随机洗牌
    train_loader2 = train_loader2[index].view(train_loader2.size())
    train_loader2 = augment_3d_batch(train_loader2)    # 数据增强
    train_loader2 = train_loader2.view(batch_num * batch_s // batch_size, batch_size, channel, depth, height, width)

    model.train()
    running_loss = 0.0
    for batch_idx, blurred_images in enumerate(train_loader2):
        blurred_x = blurred_images.to(device)  # 模糊图像
        if (padding_size != (0, 0, 0)):
            blurred_x_pad = F.pad(blurred_x, (
                padding_size[2], padding_size[2], padding_size[1], padding_size[1], padding_size[0],
                padding_size[0]),  'reflect')
        else:
            blurred_x_pad = blurred_x

        # 前向传播
        optimizer.zero_grad()
        if back_flag:
            predicted_raw, predicted_raw_high, high_res_img, blurred_x_high, predicted_blur, _ = model(blurred_x_pad, psf_x, padding_mode=padding_mode)
            loss_normal, loss, space, ssim, fourier_loss_low, space_high, ssim_high, fourier_loss_high, tv_xy, tv_z, hessian, sparsity, dark = criterion(
                predicted_raw, predicted_raw_high, blurred_x, blurred_x_high, high_res_img, predicted_blur)
        else:
            predicted_blur, high_res_img = model(blurred_x_pad, psf_x, padding_mode=padding_mode)
            loss_normal, loss, space, ssim, fourier_loss, tv_xy, tv_z, hessian, sparsity, dark = criterion(predicted_blur, blurred_x, high_res_img)


        if torch.isnan(loss):
            print(f"损失变为 NaN，训练中断。")
            if back_flag:
                print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss_low: {fourier_loss_low.item():.4f}, space_high: {space_high.item():.4f}, ssim_high: {ssim_high.item():.4f}, fourier_loss_high: {fourier_loss_high.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
            else:
                print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss: {fourier_loss.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
            return False # 退出训练

        # 反向传播和优化
        loss.backward()

        # 检查梯度是否为 NaN 或 Inf
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"梯度出现 NaN，在层 {name} 的梯度中检测到 NaN！")
                    if back_flag:
                        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss_low: {fourier_loss_low.item():.4f}, space_high: {space_high.item():.4f}, ssim_high: {ssim_high.item():.4f}, fourier_loss_high: {fourier_loss_high.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
                    else:
                        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss: {fourier_loss.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
                    return False  # 退出训练
                if torch.isinf(param.grad).any():
                    print(f"梯度出现 Inf，在层 {name} 的梯度中检测到 Inf！")
                    if back_flag:
                        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss_low: {fourier_loss_low.item():.4f}, space_high: {space_high.item():.4f}, ssim_high: {ssim_high.item():.4f}, fourier_loss_high: {fourier_loss_high.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
                    else:
                        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss: {fourier_loss.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
                    return False  # 退出训练

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        running_loss += loss_normal

        if batch_idx % 5 == 0:
            if back_flag:
                print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss_low: {fourier_loss_low.item():.4f}, space_high: {space_high.item():.4f}, ssim_high: {ssim_high.item():.4f}, fourier_loss_high: {fourier_loss_high.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
            else:
                print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss: {fourier_loss.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
    if back_flag:
        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss_low: {fourier_loss_low.item():.4f}, space_high: {space_high.item():.4f}, ssim_high: {ssim_high.item():.4f}, fourier_loss_high: {fourier_loss_high.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
    else:
        print(f"Batch {batch_idx}, Loss: {loss_normal:.4f}, space: {space.item():.4f}, ssim: {ssim.item():.4f}, fourier_loss: {fourier_loss.item():.4f}, hessian: {hessian.item():.6f}, tv_xy: {tv_xy.item():.6f}, tv_z: {tv_z.item():.6f}, sparsity: {sparsity.item():.4f}, dark: {dark.item():.4f}")
    return running_loss / len(train_loader2)

if __name__ == "__main__":

    mp.set_start_method('spawn')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("GPU可用！")

    ################################## 超参数 ###################################
    scale = 2  # 横向上采样倍率
    scale_axial = 2  # 轴向上采样倍率

    x_blur_dir = r"Z:\ZY_D\to ljc\20250627 Tom20\for_net2\act_TIM\泛化性实验\2"  # 原图像所在文件夹路径
    kernel_path = r"Z:\ZY_D\to ljc\20250627 Tom20\for_net2\sr_tublin.tif"  # psf路径
    data_cache_path = os.path.join(x_blur_dir, "cached_data.pt")  # 数据缓存文件的路径
    data_save_path = os.path.join(x_blur_dir, "cached_data")  # 裁剪图像块保存文件夹的路径
    data_final_save_path = os.path.join(data_save_path, 'final')  # 超分模型输出的裁剪图像块保存文件夹的路径
    best_weight_path = r"Z:\ZY_D\to ljc\20250627 Tom20\for_net2\act_TIM\泛化性实验\2\best_model.pth"  # 已有的最佳权重，读取进行微调
    model_path = r"Z:\ZY_D\to ljc\20250627 Tom20\for_net2\act_TIM\泛化性实验\2"  # 权重文件夹路径
    best_model_path = os.path.join(model_path, "best_model2.pth")  # 最佳权重保存路径
    final_model_path =os.path.join(model_path, "restoration_model2.pth")  # 最终权重保存路径
    loss_curve_path = os.path.join(model_path, "loss_curve2.png")  # 损失函数曲线保存路径

    batch_size = 2  # 训练批次大小
    crop_size = 96  # 横向裁剪大小
    crop_depth = 16  # 轴向裁剪大小
    num_per_img = 800  # 每个stack裁剪图像块数量
    top_percent = 0.1  # 裁剪图像块时选取标准差(或均值)前多少的
    save_flag = True  # 是否保存裁剪图像块

    num_features = 32
    back_features = 32
    num_groups = 2
    num_blocks = 4
    activate = "tanh"
    back_flag = True    # 跑两轮 第一轮是False 第二轮是True
    max_drop_path_rate = 0.075
    # window_size = (5, 5, 5)    # psf的外接矩形直径
    window_size = 7
    padding_mode = 'reflect'
    alpha = 1.0    # 一般是0.5
    if isinstance(window_size, tuple):
        mode = '3d'
    else:
        mode = '2d'
    back_ratio = (0.1, 0.1)
    freq_ratio_high = 0.10
    freq_ratio2 = 0.75
    attenuation_slope = 0.3
    fourier_flag = True

    ssim_weight = 1.0
    fourier_weight = 0.
    if back_flag:
        fourier_weight_high = 0.
        high_weight = 1.0
        hessian_weight = 3  # hessian损失的权重
        tv_weight = (3, 3)  # tv损失的权重
        sparsity_weight = 0.5  # 稀疏损失的权重
        dark_weight = 1.0
    else:
        hessian_weight = 5  # hessian损失的权重
        tv_weight = (5, 5)  # tv损失的权重
        sparsity_weight = 0.05  # 稀疏损失的权重
        dark_weight = 0.05
    loss_type = 'both'  # ‘L1’：MAE; 'L2'：MSE; else：Combine
    if back_flag:
        learning_rate = 10e-5  # 重建模块学习率
        learning_rate_back = 50e-5  # 背景估计模块学习率
        num_epochs = 100  # epoch大小
    else:
        learning_rate = 40e-5  # 重建模块学习率
        learning_rate_back = 0  # 背景估计模块学习率
        num_epochs = 50  # epoch大小
    save_final_flag = True  # 是否保存最终输出的图像裁剪块
    LN_flag = True
    model_type = "AxialTrans"
    # model_type = "CNN"
    shuffle_flag = True
    padding_size = (4, 4, 4)

    fourier_magnitude_cutoff_ratio_axial = 0.75  # 轴向截止比例
    fourier_magnitude_transition_width_axial = 0.25  # 轴向 Sigmoid 过渡宽度
    fourier_magnitude_cutoff_ratio_lateral = 0.75  # 横向截止比例
    fourier_magnitude_transition_width_lateral = 0.25

    fourier_magnitude_cutoff_ratio_axial2 = 0.75  # 轴向截止比例
    fourier_magnitude_transition_width_axial2 = 0.25  # 轴向 Sigmoid 过渡宽度
    fourier_magnitude_cutoff_ratio_lateral2 = 0.75  # 横向截止比例
    fourier_magnitude_transition_width_lateral2 = 0.25
    #############################################################################

    # 创建文件夹
    folder = os.path.exists(data_save_path)
    if not folder:
        os.makedirs(data_save_path)
        print('文件夹创建成功：', data_save_path)
    else:
        print('文件夹已经存在：', data_save_path)
    folder = os.path.exists(data_final_save_path)
    if not folder:
        os.makedirs(data_final_save_path)
        print('文件夹创建成功：', data_final_save_path)
    else:
        print('文件夹已经存在：', data_final_save_path)
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)
        print('文件夹创建成功：', model_path)
    else:
        print('文件夹已经存在：', model_path)
    folder = os.path.exists(os.path.join(data_final_save_path, "back"))
    if not folder:
        os.makedirs(os.path.join(data_final_save_path, "back"))
        print('文件夹创建成功：', os.path.join(data_final_save_path, "back"))
    else:
        print('文件夹已经存在：', os.path.join(data_final_save_path, "back"))

    # 读取psf
    kernel_blur = np.float32(imread(kernel_path))
    kernel_blur = kernel_blur / np.sum(kernel_blur)
    kernel_blur = torch.from_numpy(kernel_blur).type(torch.FloatTensor).to(device)
    if len(kernel_blur.shape) == 3:
        kernel_blur = kernel_blur.unsqueeze(0).unsqueeze(0)
    elif len(kernel_blur.shape) == 2:
        kernel_blur = kernel_blur.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    print("kernel_blur shape:", kernel_blur.shape)


    # 创建数据加载器
    if os.path.exists(data_cache_path):
        print(f"Loading data from {data_cache_path}")
        # 直接加载数据张量
        train_loader = torch.load(data_cache_path, weights_only=False)
        batch_num, batch_s, channel, depth, height, width = train_loader.shape
        train_loader = train_loader.view(batch_num * batch_s // batch_size, batch_size, channel, depth, height, width)
        train_loader = train_loader.to(device)
        # if not back_flag:
        train_loader = train_loader
        # 输出加载的数据形状
        print("Final batches shape:", train_loader.shape)
    else:
        train_loader = get_3d_dataloaders_single(x_blur_dir, batch_size=batch_size, crop_size=crop_size,
                                           crop_depth=crop_depth, top_percent=top_percent, num_per_img=num_per_img)
        train_loader = train_loader.to(device)
        batch_num, batch_s, channel, depth, height, width = train_loader.shape
        # 保存生成的数据张量
        torch.save(train_loader, data_cache_path)
        print(f"Data saved to {data_cache_path}")
        # if not back_flag:
        train_loader = train_loader
    print(torch.max(train_loader))

    img_name = [f for f in os.listdir(x_blur_dir) if f.lower().endswith(('.tif', '.tiff'))]
    num_img = len(img_name)
    if (save_flag):
        batch_num, batch_s, channel, depth, height, width = train_loader.shape
        train_loader1 = train_loader.view(batch_num * batch_s, 1, channel, depth, height, width)
        for i in range(int(num_img * num_per_img * top_percent)):
            img = train_loader1[i, :, 0, :, :, :].squeeze().cpu().numpy()
            imwrite(os.path.join(data_save_path, str(i + 1) + ".tif"), img)

    # 初始化模型
    model = RestorationNetwork3d_subback2(scale=scale, scale_axial=scale_axial, num_features=num_features, back_features=back_features,
                                         num_groups=num_groups, num_blocks=num_blocks, back_flag=back_flag, activate=activate,
                                         img_size=(crop_depth+2*padding_size[0], crop_size+2*padding_size[1], crop_size+2*padding_size[2]),
                                         back_ratio=back_ratio, freq_ratio_low=freq_ratio_high, freq_ratio_high=freq_ratio_high, freq_ratio2=freq_ratio2, attenuation_slope=attenuation_slope,
                                         train_flag=True, model_type=model_type, shuffle_flag=shuffle_flag,
                                         padding_size=padding_size, device=device, LN_flag=LN_flag, max_drop_path_rate=max_drop_path_rate).to(device)
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path))
        print(f"Model loading weight from {best_weight_path}")
    # model = torch.compile(model, backend="aot_eager")

    # 初始化损失函数
    if back_flag:
        criterion = FinalLoss_single(spatial_weight=1, high_weight=high_weight, ssim_weight=ssim_weight, fourier_weight=fourier_weight, fourier_weight_high=fourier_weight_high,
                                     hessian_weight=hessian_weight, tv_weight=tv_weight, sparsity_weight=sparsity_weight,
                                     dark_weight=dark_weight, loss_type=loss_type, window_size=window_size, alpha=alpha, mode=mode, fourier_flag=fourier_flag,
                                     fourier_magnitude_cutoff_ratio_axial=fourier_magnitude_cutoff_ratio_axial,
                                     fourier_magnitude_transition_width_axial=fourier_magnitude_transition_width_axial,
                                     fourier_magnitude_cutoff_ratio_lateral=fourier_magnitude_cutoff_ratio_lateral,
                                     fourier_magnitude_transition_width_lateral=fourier_magnitude_transition_width_lateral,
                                     D=crop_depth, H=crop_size, W=crop_size, device=device,
                                     fourier_magnitude_cutoff_ratio_axial2=fourier_magnitude_cutoff_ratio_axial2,
                                     fourier_magnitude_cutoff_ratio_lateral2=fourier_magnitude_cutoff_ratio_lateral2,
                                     fourier_magnitude_transition_width_axial2=fourier_magnitude_transition_width_axial2,
                                     fourier_magnitude_transition_width_lateral2=fourier_magnitude_transition_width_lateral2)
    else:
        criterion = FinalLoss_single_blur(spatial_weight=1, ssim_weight=ssim_weight, fourier_weight=fourier_weight, hessian_weight=hessian_weight, tv_weight=tv_weight,
                                          sparsity_weight=sparsity_weight, dark_weight=dark_weight, loss_type=loss_type, window_size=window_size, alpha=alpha, mode=mode, fourier_flag=fourier_flag,
                                          fourier_magnitude_cutoff_ratio_axial=fourier_magnitude_cutoff_ratio_axial,
                                          fourier_magnitude_cutoff_ratio_lateral=fourier_magnitude_cutoff_ratio_lateral,
                                          fourier_magnitude_transition_width_axial=fourier_magnitude_transition_width_axial,
                                          fourier_magnitude_transition_width_lateral=fourier_magnitude_transition_width_lateral, D=crop_depth, H=crop_size, W=crop_size, device=device)

    # 初始化优化器
    unet_params = list(model.back_block.parameters()) + list(model.back_conv.parameters())
    other_params = [
        param for param in model.parameters() if id(param) not in {id(p) for p in unet_params}]
    optimizer = optim.Adam([
        {'params': unet_params, 'lr': learning_rate_back},  # conv_feature 和 mlp 的参数
        {'params': other_params, 'lr': learning_rate}  # 其他模型参数
    ])
    scheduler = CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=num_epochs)

    # 训练超分反卷积模型
    best_loss = float('inf')
    train_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        avg_train_loss = train(model, batch_size, train_loader, kernel_blur, criterion, optimizer, device, padding_size=padding_size,back_flag=back_flag)
        if not avg_train_loss:
            raise ValueError("梯度出现NaN")

        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

        scheduler.step()

        current_lr_back = scheduler.get_last_lr()[0]
        current_lr_sr = scheduler.get_last_lr()[1]
        print(f"Current Back Module Learning Rate: {current_lr_back:.10f};"
              f"Current Sr Module Learning Rate: {current_lr_sr:.10f}")

    # 保存最终图像
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.flag = True
    train_loader_new = preprocess_to_tensor(model, train_loader, kernel_blur, kernel_blur, data_final_save_path=data_final_save_path, scale=scale, scale_axial=scale_axial, padding_size=padding_size).to(device)
    if (save_final_flag):
        batch_num, batch_s, channel, depth, height, width = train_loader_new.shape
        train_loader1 = train_loader_new.view(batch_num * batch_s, 1, channel, depth, height, width)
        num = train_loader1.shape[0]
        for i in range(num):
            img = train_loader1[i, :, 0, :, :, :].squeeze().cpu().numpy()
            imwrite(os.path.join(data_final_save_path, str(i + 1) + ".tif"), img)

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    loss_length = len(train_losses)

    plt.figure()
    plt.plot(range(1, loss_length + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_path)
    # plt.show()
    print(f"Loss curve saved at {loss_curve_path}")
