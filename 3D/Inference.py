import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from tifffile import imread, imwrite
import torch
import torch.nn.functional as F
import time

from model import RestorationNetwork3d_Inference


def inference_blockwise(num, model, image_x, block_size, block_depth, overlap, overlap_depth, scale_factor_xy,
                        scale_factor_z, padding_size=(0, 0, 0), n=0):
    """
    Args:
        num (int): 当前处理图像的索引，用于打印进度。
        model (nn.Module): 实例化后的 RestorationNetwork3d_Inference 模型。
        image_x (torch.Tensor): 输入的模糊图像，形状 [1, 1, D, H, W]。
        block_size (int): 横向分块尺寸。
        block_depth (int): 轴向分块深度。
        overlap (int): 横向重叠区域大小。
        overlap_depth (int): 轴向重叠区域深度。
        scale_factor_xy (int): 横向超分辨率倍率。
        scale_factor_z (int): 轴向超分辨率倍率。
        padding_size (tuple): 模型初始化时使用的填充尺寸 (d_pad, h_pad, w_pad)。
                              用于裁剪最终输出。
    Returns:
        torch.Tensor: 重建后的高分辨率图像，形状 [output_D, output_H, output_W]。
    """
    # 计算输入图像的原始尺寸
    image_x = image_x[:, :, :, :, :]
    depth, height, width = image_x.shape[-3:]

    depth_new = (block_depth - overlap_depth) * ((depth - block_depth) // (block_depth - overlap_depth) + 1) + block_depth
    image_x_new = torch.zeros((1, 1, depth_new, height, width))
    image_x_new[:, :, :depth, :, :] = image_x
    image_x_new[:, :, depth:, :, :] = torch.flip(image_x, dims=[2])[:, :, 1:(depth_new - depth + 1), :, :]
    image_x = image_x_new.to(image_x.device)
    depth_0 = depth
    depth = depth_new
    # 计算输出图像的尺寸 (考虑超分倍率)
    output_depth = int(depth * scale_factor_z)
    output_height = int(height * scale_factor_xy)
    output_width = int(width * scale_factor_xy)

    output = np.zeros((1, 1, output_depth, output_height, output_width))
    weight_map = np.zeros((1, 1, output_depth, output_height, output_width))

    # 遍历分块区域进行推理
    with torch.no_grad():
        for d in tqdm(range(0, depth, block_depth - overlap_depth), desc=f"Image {num} Z-axis"):
            for h in range(0, height, block_size - overlap):
                for w in range(0, width, block_size - overlap):
                    print(f"Processing block at {num}, D:{d / depth:.2%}, H:{h / height:.2%}, W:{w / width:.2%}")

                    # 当前块的范围，考虑到重叠区域
                    d_end = min(d + block_depth, depth)
                    h_end = min(h + block_size, height)
                    w_end = min(w + block_size, width)

                    # 提取当前块
                    block_x = image_x[:, :, d:d_end, h:h_end, w:w_end]
                    if (padding_size != (0, 0, 0)):
                        block_x = F.pad(block_x, (padding_size[2], padding_size[2], padding_size[1], padding_size[1], padding_size[0],
                            padding_size[0]), 'reflect')

                    normalized_block_x = block_x

                    high_res_img_block = model(normalized_block_x)

                    # 将块尺寸转换为横向和轴向的上采样后的尺寸
                    scaled_d_start = int(d * scale_factor_z)
                    scaled_d_end = int(d_end * scale_factor_z)
                    scaled_h_start = int(h * scale_factor_xy)
                    scaled_h_end = int(h_end * scale_factor_xy)
                    scaled_w_start = int(w * scale_factor_xy)
                    scaled_w_end = int(w_end * scale_factor_xy)

                    # 生成权重矩阵，中心区域权重较高，边缘区域权重较低
                    weight = torch.ones_like(high_res_img_block)
                    for i in range(weight.shape[2]):
                        z_weight = 1 - abs(i - (weight.shape[2] - 1) / 2) / ((weight.shape[2] + 1) / 2)
                        weight[:, :, i, :, :] *= z_weight
                    for j in range(weight.shape[3]):
                        y_weight = 1 - abs(j - (weight.shape[3] - 1) / 2) / ((weight.shape[3] + 1) / 2)
                        weight[:, :, :, j, :] *= y_weight
                    for k in range(weight.shape[4]):
                        x_weight = 1 - abs(k - (weight.shape[4] - 1) / 2) / ((weight.shape[4] + 1) / 2)
                        weight[:, :, :, :, k] *= x_weight

                    # 将当前块加入到输出图像
                    high_res_img_block = high_res_img_block.cpu().numpy()
                    weight = weight.cpu().numpy()
                    output[:, :, scaled_d_start:scaled_d_end, scaled_h_start:scaled_h_end,
                    scaled_w_start:scaled_w_end] += high_res_img_block * weight
                    weight_map[:, :, scaled_d_start:scaled_d_end, scaled_h_start:scaled_h_end,
                    scaled_w_start:scaled_w_end] += weight
                    if (w + block_size >= width):
                        break
                if (h + block_size >= height):
                    break
            if (d + block_depth >= depth):
                break
    output /= weight_map
    output = output[0, 0, :int(depth_0 * scale_factor_z), :, :]
    print(np.shape(output))
    if n > 0:
        output = output[n: -n, :, :]
    return output


def preprocess_image_3d(image_path, device, n=0):
    """
    对 3D 图像堆栈进行归一化，并根据参数 n 实现镜像扩充。

    Args:
        image_path (str): TIF 图像路径
        device (torch.device): 计算设备
        n (int): 需要在前后各扩充的 slice 数量
    """
    # 1. 读取图像 [D, H, W]
    image_np = np.float32(imread(image_path))
    image = torch.from_numpy(image_np).to(device)

    # 2. 扩充堆栈 (镜像对称填充)
    # 只有当 n > 0 且堆栈深度足够时才进行操作
    if n > 0:
        # 获取 1到n+1 个 slice 并反序
        # slice(1, n+1) -> [1, ..., n]
        # flip(0) 在深度(D)维度上翻转
        prefix = image[1 : n + 1].flip(dims=[0])

        # 获取后 n 个 slice 并反序
        # slice(-n-1, None) -> 最后 n 个
        suffix = image[-n - 1 : -1].flip(dims=[0])

        # 在 D 维度上拼接: [前缀, 原始, 后缀]
        image = torch.cat([prefix, image, suffix], dim=0)

    # 3. 归一化处理 (Min-Max Normalization)
    img_min = torch.min(image)
    img_max = torch.max(image)

    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)

    # 4. 调整维度以符合 PyTorch 3D 训练要求: [Batch, Channel, D, H, W]
    # 当前 image 形状为 [D, H, W]，增加 B 和 C 维度
    image = image.unsqueeze(0).unsqueeze(0)

    return image


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description="PRISM inference")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Folder containing input .tif/.tiff images or stacks")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for restored outputs")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--block_depth", type=int, default=24)
    parser.add_argument("--overlap_depth", type=int, default=4)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n = 0
    scale = 2  # 横向上采样倍率
    scale_axial = 1  # 轴向上采样倍率
    num_features = 32
    padding_size = (4, 8, 8)  # 与训练模型保持一致

    path_x = str(args.input_dir)
    weight_path = str(args.weight_path)
    save_path = str(args.output_dir)
    os.makedirs(save_path, exist_ok=True)

    block_size = args.block_size
    overlap = args.overlap
    block_depth = args.block_depth
    overlap_depth = args.overlap_depth

    # 创建输出文件夹
    os.makedirs(save_path, exist_ok=True)
    print(f"输出文件夹: {save_path}")

    # --- 加载推理模型并加载权重 ---
    model = RestorationNetwork3d_Inference(
        scale=scale,
        scale_axial=scale_axial,
        num_features=num_features,
        padding_size=padding_size,
    ).to(device)

    print(f"尝试从 {weight_path} 加载模型权重...")
    trained_state_dict = torch.load(weight_path, map_location=device, weights_only=True)

    # 过滤掉推理模型不需要的键
    inference_state_dict = {}
    for k, v in trained_state_dict.items():
        if k.startswith('primary_encoder.') or \
                k.startswith('deep_encoder.') or \
                k.startswith('decoder.') or \
                k.startswith('conv_out.') or \
                k.startswith('activate.'):
            inference_state_dict[k] = v

    # 加载过滤后的权重
    model.load_state_dict(inference_state_dict, strict=True)
    print("模型权重加载完成。")

    model.eval()

    # --- 逐块推理循环 ---
    img_names = [f for f in os.listdir(path_x) if f.lower().endswith(('.tif', '.tiff'))]
    num_length = len(img_names)

    for i in range(num_length):

        input_x_image_path = os.path.join(path_x, img_names[i])
        print(f"\n--- 处理图像: {img_names[i]} ({i + 1}/{num_length}) ---")
        blurred_x = preprocess_image_3d(input_x_image_path, device, n=n)

        start_time = time.time()
        # 执行带平滑的逐块推理
        high_res_img_np = inference_blockwise(
            i + 1,
            model,
            blurred_x,
            block_size,
            block_depth,
            overlap,
            overlap_depth,
            scale,
            scale_axial,
            padding_size=padding_size,
            n=n
        )

        end_time = time.time()
        run_time = end_time - start_time
        print(f"代码运行时间: {run_time:.4f} 秒")

        # 归一化到 [0, 1] 再转换为 uint16
        if np.max(high_res_img_np) - np.min(high_res_img_np) > 1e-8:
            high_res_img_np = (high_res_img_np - np.min(high_res_img_np)) / (np.max(high_res_img_np) - np.min(high_res_img_np))
        else:
            high_res_img_np = np.zeros_like(high_res_img_np)  # 如果图像是常数，直接设为0

        high_res_img_np = np.uint16(65535 * high_res_img_np)

        # 保存结果
        outpath = os.path.join(save_path, f"{os.path.splitext(img_names[i])[0]}.tif")
        imwrite(outpath, high_res_img_np)
        print(f"图像 {img_names[i]} 处理完成，结果已保存到 {outpath}")

    print("\n所有图像推理完成。")