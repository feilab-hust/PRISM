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


def inference_blockwise(num, model, image_x, block_size, overlap, scale_factor_xy, padding_size=(0, 0)):
    """
    Args:
        num (int): 当前处理图像的索引，用于打印进度。
        model (nn.Module): 实例化后的 RestorationNetwork3d_Inference 模型。
        image_x (torch.Tensor): 输入的模糊图像，形状 [1, 1, D, H, W]。
        block_size (int): 横向分块尺寸。
        overlap (int): 横向重叠区域大小。
        scale_factor_xy (int): 横向超分辨率倍率。
        padding_size (tuple): 模型初始化时使用的填充尺寸 (h_pad, w_pad)。
                              用于裁剪最终输出。
    Returns:
        torch.Tensor: 重建后的高分辨率图像，形状 [output_H, output_W]。
    """
    # 计算输入图像的原始尺寸
    height, width = image_x.shape[-2:]

    output_height = int(height * scale_factor_xy)
    output_width = int(width * scale_factor_xy)

    output = np.zeros((1, 1, output_height, output_width))
    weight_map = np.zeros((1, 1, output_height, output_width))

    # 遍历分块区域进行推理
    with torch.no_grad():
        for h in tqdm(range(0, height, block_size - overlap), desc=f"Image {num} X-axis"):
            for w in range(0, width, block_size - overlap):
                print(f"Processing block at {num}, H:{h / height:.2%}, W:{w / width:.2%}")

                # 当前块的范围，考虑到重叠区域
                h_end = min(h + block_size, height)
                w_end = min(w + block_size, width)

                # 提取当前块
                block_x = image_x[:, :, h:h_end, w:w_end]
                if (padding_size != (0, 0)):
                    block_x = F.pad(block_x, (padding_size[1], padding_size[1], padding_size[0], padding_size[0]), 'reflect')

                normalized_block_x = block_x

                high_res_img_block = model(normalized_block_x)

                # 将块尺寸转换为横向和轴向的上采样后的尺寸
                scaled_h_start = int(h * scale_factor_xy)
                scaled_h_end = int(h_end * scale_factor_xy)
                scaled_w_start = int(w * scale_factor_xy)
                scaled_w_end = int(w_end * scale_factor_xy)

                # 生成权重矩阵，中心区域权重较高，边缘区域权重较低
                weight = torch.ones_like(high_res_img_block)
                for j in range(weight.shape[2]):
                    y_weight = 1 - abs(j - (weight.shape[2] - 1) / 2) / ((weight.shape[2] + 1) / 2)
                    weight[:, :, j, :] *= y_weight
                for k in range(weight.shape[3]):
                    x_weight = 1 - abs(k - (weight.shape[3] - 1) / 2) / ((weight.shape[3] + 1) / 2)
                    weight[:, :, :, k] *= x_weight

                # 将当前块加入到输出图像
                high_res_img_block = high_res_img_block.cpu().numpy()
                weight = weight.cpu().numpy()
                output[:, :, scaled_h_start:scaled_h_end, scaled_w_start:scaled_w_end] += high_res_img_block * weight
                weight_map[:, :, scaled_h_start:scaled_h_end, scaled_w_start:scaled_w_end] += weight
                if (w + block_size >= width):
                    break
            if (h + block_size >= height):
                break
    output /= weight_map
    output = output[0, 0, :, :]
    return output


def preprocess_image_2d(image_path, device, idx=1.0):
    image = np.float32(imread(image_path))
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
    image = image / torch.max(image) * idx
    return image


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description="PRISM inference")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Folder containing input .tif/.tiff images or stacks")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for restored outputs")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale = 2  # 横向上采样倍率
    num_features = 32
    padding_size = (8, 8)  # 与训练模型保持一致

    path_x = str(args.input_dir)
    weight_path = str(args.weight_path)
    save_path = str(args.output_dir)
    os.makedirs(save_path, exist_ok=True)

    block_size = args.block_size
    overlap = args.overlap

    # 创建输出文件夹
    os.makedirs(save_path, exist_ok=True)
    print(f"输出文件夹: {save_path}")

    # --- 加载推理模型并加载权重 ---
    model = RestorationNetwork3d_Inference(
        scale=scale,
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
        start_time = time.time()

        input_x_image_path = os.path.join(path_x, img_names[i])
        print(f"\n--- 处理图像: {img_names[i]} ({i + 1}/{num_length}) ---")
        blurred_x = preprocess_image_2d(input_x_image_path, device)

        # 执行带平滑的逐块推理
        high_res_img_np = inference_blockwise(
            i + 1,
            model,
            blurred_x,
            block_size,
            overlap,
            scale,
            padding_size=padding_size
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