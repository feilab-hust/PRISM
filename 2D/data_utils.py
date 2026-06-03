# import os
# import numpy as np
# from PIL import Image
# from tifffile import imread, imwrite
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from utils import normalize_to_01
#
#
# class SingleDimBlur3DDataset_single(Dataset):
#     def __init__(self, x_blur_dir, crop_size=64, crop_depth=32, transform=None, num_per_img=200):
#         """
#         参数：
#             x_blur_dir (str): 存放x方向模糊图像的文件夹路径。
#             crop_size (int): 图像裁剪后的宽高尺寸。
#             crop_depth (int): 图像裁剪后的深度尺寸（z轴）。
#             transform (callable, optional): 可选的对样本应用的变换操作。
#             num_per_img (int): 每个堆栈生成的裁剪次数。
#         """
#         self.x_blur_dir = x_blur_dir
#         self.crop_size = crop_size
#         self.crop_depth = crop_depth
#         self.transform = transform
#         self.num_per_img = num_per_img
#         # 仅获取 .tif 或 .tiff 文件
#         self.image_names = sorted(
#             [f for f in os.listdir(x_blur_dir) if f.lower().endswith(('.tif', '.tiff'))]
#         )
#
#     def __len__(self):
#         return len(self.image_names) * self.num_per_img
#
#     def __getitem__(self, idx):
#         # 获取图像文件的索引
#         real_idx = idx // self.num_per_img
#         x_img_path = os.path.join(self.x_blur_dir, self.image_names[real_idx])
#
#         # 读取三维堆栈
#         z_stack = imread(x_img_path)[:, :, :].astype(np.float32)  # 假设原始图像为16位
#
#         # 随机选择起始 z 位置，确保不会越界
#         max_z_start = z_stack.shape[0] - self.crop_depth
#         z_start = np.random.randint(0, max_z_start + 1)
#
#         # 随机选择裁剪参数
#         slice_for_params = Image.fromarray(z_stack[z_start])  # 随机裁剪参数基于起始切片
#         i, j, h, w = transforms.RandomCrop.get_params(slice_for_params, output_size=(self.crop_size, self.crop_size))
#
#         # 裁剪整个深度范围内的切片
#         cropped_stack = []
#         for z in range(z_start, z_start + self.crop_depth):
#             slice_img = Image.fromarray(z_stack[z])
#             cropped_slice = transforms.functional.crop(slice_img, i, j, h, w)
#             cropped_stack.append(transforms.functional.to_tensor(cropped_slice))
#
#         # 将裁剪后的切片组合为张量并调整维度
#         x_image_3d = torch.stack(cropped_stack)
#         x_image_3d = x_image_3d.squeeze()  # 形状为 [depth, height, width]
#         x_image_3d = x_image_3d.unsqueeze(0)  # 添加通道维度，变为 [1, depth, height, width]
#
#         if self.transform:
#             x_image_3d = self.transform(x_image_3d)
#
#         return x_image_3d
#
#
# def get_3d_dataloaders_single(x_blur_dir, batch_size=8, crop_size=64, crop_depth=32, shuffle=False,
#                        top_percent=0.1, num_per_img=200):
#     dataset = SingleDimBlur3DDataset_single(x_blur_dir, crop_size=crop_size, crop_depth=crop_depth,
#                                      num_per_img=num_per_img)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=32)
#
#     # 用于存储所有图像的选定裁剪块
#     all_selected_patches = []
#
#     # 按图像分组处理
#     current_image_patches = []
#     for idx, batch in enumerate(dataloader):
#         print(idx+1)
#         current_image_patches.append(batch.squeeze(0))  # 去掉 batch_size=1 的维度
#         # 每 `num_per_img` 个 batch 属于同一图像
#         if (idx + 1) % num_per_img == 0:
#             # 计算每个裁剪块的标准差
#             values = [torch.std(patch[0, crop_depth // 4:3 * crop_depth // 4, crop_size // 4:3 * crop_size // 4, crop_size // 4:3 * crop_size // 4]).item() for patch in current_image_patches]
#             # 计算每个裁剪块的均值
#             # values = [torch.mean(patch[0, crop_depth // 4:3 * crop_depth // 4, crop_size // 4:3 * crop_size // 4, crop_size // 4:3 * crop_size // 4]).item() for patch in current_image_patches]
#             # 将裁剪块和标准差(或均值)组合
#             image_patches_with_std = list(zip(current_image_patches, values))
#             # 按标准差降序排列
#             image_patches_with_std.sort(key=lambda x: x[1], reverse=True)
#             # 筛选前 top_percent 的裁剪块
#             top_k = int(len(image_patches_with_std) * top_percent)
#             selected_patches = [patch[0] for patch in image_patches_with_std[:top_k]]
#             # 添加到总选定裁剪块列表
#             all_selected_patches.extend(selected_patches)
#             # 清空当前图像的裁剪块列表
#             current_image_patches = []
#
#     # 将所有选定的裁剪块拼接
#     all_selected_patches = torch.stack(all_selected_patches, dim=0)
#
#     for i in range(all_selected_patches.shape[0]):
#         all_selected_patches[i, 0, :, :, :] = all_selected_patches[i, 0, :, :, :] / torch.max(all_selected_patches[i, 0, :, :, :])
#         # all_selected_patches[i, 0, :, :, :] = normalize_to_01(all_selected_patches[i, 0, :, :, :].unsqueeze(1)).squeeze(1)
#
#     # 按 batch_size 将选定的裁剪块划分成多个批次
#     num_batches = all_selected_patches.shape[0] // batch_size
#     final_batches = torch.stack([
#         all_selected_patches[i * batch_size: (i + 1) * batch_size]
#         for i in range(num_batches)
#     ], dim=0)
#
#     print("Final batches shape:", final_batches.shape)  # 形状: [num_batches, batch_size, 1, depth, height, width]
#     return final_batches


import os
import numpy as np
import torch
from utils import normalize_to_01
from tifffile import imread
from torchvision.transforms.functional import crop as tv_crop


def get_2d_dataloaders_single(x_blur_dir, batch_size=8, crop_size=64, shuffle=False, top_percent=0.1, num_per_img=200):
    """
    高效地读取二维图像，进行随机裁剪、高方差筛选并生成批次。

    该函数直接在内存中执行所有操作，避免了 Dataset/DataLoader 的 I/O 瓶颈。

    输出形状: [num_batches, batch_size, 1, height, width]
    """
    # ------------------ 1. 文件 I/O 和初始化 ------------------

    # 仅获取 .tif 或 .tiff 文件
    image_names = sorted(
        [f for f in os.listdir(x_blur_dir) if f.lower().endswith(('.tif', '.tiff'))]
    )
    if not image_names:
        print(f"警告: 路径 {x_blur_dir} 中未找到 .tif 文件。")
        return torch.empty(0)

    # 用于存储所有图像的选定裁剪块
    all_selected_patches = []

    print(f"找到 {len(image_names)} 个图像文件，开始预处理...")

    # ------------------ 2. 按图像处理、裁剪和筛选 ------------------

    for img_idx, img_name in enumerate(image_names):
        x_img_path = os.path.join(x_blur_dir, img_name)

        # 优化 I/O: 一次性读取整个图像
        # 形状: [H_orig, W_orig]
        z_stack_np = imread(x_img_path).astype(np.float32)

        H_orig, W_orig = z_stack_np.shape

        if H_orig < crop_size or W_orig < crop_size:
            print(f"警告: 文件 {img_name} 尺寸 ({H_orig}x{W_orig}) 小于裁剪尺寸，跳过。")
            continue

        # 转换为 PyTorch 张量，并添加通道维度: [1, H, W]
        z_stack = torch.from_numpy(z_stack_np).unsqueeze(0).to(torch.float32)
        # z_stack = normalize_to_01(z_stack)
        z_stack /= torch.max(z_stack)

        current_image_patches = []

        # ------------------ 2.1 随机裁剪 ------------------
        for _ in range(num_per_img):
            # 随机选择 H/W 轴起始位置
            max_h_start = H_orig - crop_size
            max_w_start = W_orig - crop_size
            h_start = np.random.randint(0, max_h_start + 1)
            w_start = np.random.randint(0, max_w_start + 1)

            # 裁剪 (利用张量切片实现，比 PIL/transforms 快得多)
            # 裁剪后的形状: [1, crop_size, crop_size]
            cropped_stack = z_stack[:,
                            h_start: h_start + crop_size,
                            w_start: w_start + crop_size]

            current_image_patches.append(cropped_stack)

        # ------------------ 2.2 高方差筛选 ------------------
        if not current_image_patches:
            continue

        # 将所有裁剪块堆叠成一个张量 [num_per_img, 1, H, W]
        patches_tensor = torch.stack(current_image_patches, dim=0)

        # 定义内部子区域 (避免边界效应)
        h_q = crop_size // 4

        # 计算每个裁剪块内部子区域的标准差 (std) [num_per_img]
        # 筛选高方差的 patches (通常代表丰富的纹理和结构)
        values = patches_tensor[:, 0, h_q:3 * h_q, h_q:3 * h_q].std(dim=(-2, -1))
        # values = patches_tensor[:, 0, h_q:3 * h_q, h_q:3 * h_q].mean(dim=(-2, -1))

        # 组合 patches 和 values，按值降序排列
        image_patches_with_std = sorted(
            zip(current_image_patches, values.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        # 筛选前 top_percent 的裁剪块
        top_k = int(len(image_patches_with_std) * top_percent)
        selected_patches = [patch[0] for patch in image_patches_with_std[:top_k]]

        all_selected_patches.extend(selected_patches)

        print(f"文件 {img_idx + 1}/{len(image_names)} ({img_name}) 完成。选取 {len(selected_patches)} 个高方差 Patch。")

    # ------------------ 3. 归一化和批量化 ------------------

    if not all_selected_patches:
        print("未选中任何 Patch，请检查输入参数和文件。")
        return torch.empty(0)

    # 堆叠所有选定的裁剪块 [Total_Patches, 1, H, W]
    all_selected_patches_tensor = torch.stack(all_selected_patches, dim=0)
    # print(all_selected_patches_tensor.shape)

    # 归一化 (逐个 Patch 归一化)
    for i in range(all_selected_patches_tensor.shape[0]):
        # 形状: [1, H, W]
        patch_data = all_selected_patches_tensor[i, 0, :, :]
        # all_selected_patches_tensor[i, 0, :, :] = normalize_to_01(patch_data)
        all_selected_patches_tensor[i, 0, :, :] /= torch.max(all_selected_patches_tensor[i, 0, :, :])

    # ------------------ 4. 批量划分 ------------------

    if shuffle:
        # 在划分批次前，随机打乱所有选定的 Patch
        indices = torch.randperm(all_selected_patches_tensor.shape[0])
        all_selected_patches_tensor = all_selected_patches_tensor[indices]

    total_patches = all_selected_patches_tensor.shape[0]
    num_batches = total_patches // batch_size

    if num_batches == 0:
        print(f"警告: 选中的 Patch 总数 ({total_patches}) 不足以组成一个完整批次 (batch_size={batch_size})。")
        return torch.empty(0)

    # 划分成批次 [num_batches, batch_size, 1, height, width]
    final_batches = torch.stack([
        all_selected_patches_tensor[i * batch_size: (i + 1) * batch_size]
        for i in range(num_batches)
    ], dim=0)

    print(f"总 Patch 数: {total_patches}。最终生成批次数: {num_batches}。")
    print("Final batches shape:", final_batches.shape)

    return final_batches
