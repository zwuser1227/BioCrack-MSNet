import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import snn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 启用异常检测
torch.autograd.set_detect_anomaly(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 使用之前定义的序列生成器函数
# def generate_zoom_in_sequence(image_path, num_steps=5, zoom_factor=0.15):
#     original_image = cv2.imread(image_path)
#     if original_image is None:
#         raise FileNotFoundError(f"无法加载图像: {image_path}")

#     height, width = original_image.shape[:2]
#     zoom_sequence = [original_image]

#     for i in range(1, num_steps):
#         crop_size = (int(width * (1 - i * zoom_factor)), int(height * (1 - i * zoom_factor)))
#         crop_x = (width - crop_size[0]) // 2
#         crop_y = (height - crop_size[1]) // 2

#         if crop_size[0] <= 0 or crop_size[1] <= 0:
#             raise ValueError(f"裁剪后的图像尺寸无效: {crop_size}")

#         cropped_image = original_image[crop_y:crop_y + crop_size[1], crop_x:crop_x + crop_size[0]]
#         zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
#         zoom_sequence.append(zoomed_image)

#     return zoom_sequence
# def zoom_in_and_save(image_path, label_path, num_steps=5, zoom_factor=0.15):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"无法加载图像: {image_path}")
#     mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         raise FileNotFoundError(f"无法加载图像: {label_path}")

#     zoomed_images = [image]
#     h, w = image.shape[:2]

#     # # 创建输出文件夹，如果不存在
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)

#     # 找到裂缝的轮廓并计算中心
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         M = cv2.moments(largest_contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#         else:
#             cx, cy = w // 2, h // 2
#     else:
#         cx, cy = w // 2, h // 2

#     # 遍历每个缩放因子，生成对应的以裂缝为中心的图像
#     for i in range(1, num_steps):
#         # 计算放大后的裁剪区域大小
#         crop_size = (int(w * (1 - i * zoom_factor)), int(h * (1 - i * zoom_factor)))
#         # crop_size = (int(w / factor), int(h / factor))

#         # 动态调整裁剪区域，确保包含裂缝
#         x_start = max(cx - crop_size[0] // 2, 0)
#         y_start = max(cy - crop_size[1] // 2, 0)

#         # 扩展裁剪区域的大小，以避免裁剪掉裂缝
#         expanded_crop_size = (crop_size[0] * 2, crop_size[1] * 2)

#         # 确保裁剪区域不超出图像边界
#         x_start = max(cx - expanded_crop_size[0] // 2, 0)
#         y_start = max(cy - expanded_crop_size[1] // 2, 0)

#         # 确保裁剪区域不超过图像的边界
#         x_end = min(x_start + expanded_crop_size[0], w)
#         y_end = min(y_start + expanded_crop_size[1], h)

#         # 裁剪出以裂缝为中心的区域
#         cropped_image = image[y_start:y_end, x_start:x_end]

#         # 将裁剪后的区域缩放回原图大小
#         zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_CUBIC)

#         # 增强对比度
#         zoomed_image = cv2.convertScaleAbs(zoomed_image, alpha=1.5, beta=0)

#         # 保存放大后的裂缝区域
#         # zoomed_image_path = os.path.join(output_dir, f"zoomed_crack_{i}.png")
#         # cv2.imwrite(zoomed_image_path, zoomed_image)

#         # print(f"已保存裂缝区域: {zoomed_image_path}")

#         zoomed_images.append(zoomed_image)

#     # print(zoomed_images.shape)
#     return zoomed_images
import cv2
import numpy as np

def enhance_image(image):
    """
    对图像进行清晰度增强和数据增强处理。
    """
    # 增强对比度
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    # 锐化处理
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)

    # 添加随机噪声
    # noise = np.random.normal(0, 10, sharpened_image.shape).astype(np.uint8)
    # noisy_image = cv2.add(sharpened_image, noise)

    return sharpened_image

# def zoom_in_and_save(image_path, label_path, num_steps=5, zoom_factor=0.15):
#     # 加载图像和标签
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"无法加载图像: {image_path}")
#     mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         raise FileNotFoundError(f"无法加载图像: {label_path}")

#     zoomed_images = [image]
#     h, w = image.shape[:2]

#     # 找到裂缝的轮廓并计算中心
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         M = cv2.moments(largest_contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#         else:
#             cx, cy = w // 2, h // 2

#         # 计算裂缝的最小外接矩形
#         rect = cv2.minAreaRect(largest_contour)
#         (rect_w, rect_h) = rect[1]

#         # 检查裂缝是否占据图像长度或宽度的三分之二
#         if rect_w > 0.8 or rect_h > 0.8:
#             # 裂缝足够大，跳过放大处理，生成增强后的图像序列
#             for i in range(1, num_steps):
#                 enhanced_image = enhance_image(image)
#                 zoomed_images.append(enhanced_image)
#             return zoomed_images
#     else:
#         cx, cy = w // 2, h // 2
#     # 遍历每个缩放因子，生成对应的以裂缝为中心的图像
#     for i in range(1, num_steps):
#         # 计算放大后的裁剪区域大小
#         crop_size = (int(w * (1 - i * zoom_factor)), int(h * (1 - i * zoom_factor)))

#         # 动态调整裁剪区域，确保包含裂缝
#         expanded_crop_size = (crop_size[0] * 2, crop_size[1] * 2)
#         x_start = max(cx - expanded_crop_size[0] // 2, 0)
#         y_start = max(cy - expanded_crop_size[1] // 2, 0)
#         x_end = min(x_start + expanded_crop_size[0], w)
#         y_end = min(y_start + expanded_crop_size[1], h)

#         # 裁剪出以裂缝为中心的区域
#         cropped_image = image[y_start:y_end, x_start:x_end]

#         # 将裁剪后的区域缩放回原图大小
#         zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_CUBIC)

#         # 增强对比度
#         zoomed_image = cv2.convertScaleAbs(zoomed_image, alpha=1.5, beta=0)

#         # 保存放大后的图像
#         zoomed_images.append(zoomed_image)

#     return zoomed_images

def calculate_crack_size(mask):
    """计算裂缝区域的大小和位置"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        return w, h  # 返回宽度和高度
    return 0, 0  # 如果没有找到裂缝，返回0,0

def apply_light_transformations(image):
    """对图像应用轻微的变换"""
    # 随机亮度调整
    alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # 对比度调整
    beta = np.random.randint(-10, 10)  # 亮度调整
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 随机模糊
    if np.random.rand() > 0.5:
        ksize = np.random.choice([3, 5, 7])  # 随机选择一个模糊内核大小
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # 随机旋转
    if np.random.rand() > 0.5:
        angle = np.random.randint(-5, 5)  # 随机旋转角度
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image
def zoom_in_and_save(image_path, label_path, num_steps=5, zoom_factor=0.15):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法加载图像: {label_path}")

    zoomed_images = [image]
    h, w = image.shape[:2]

    # 计算裂缝区域的大小
    crack_width, crack_height = calculate_crack_size(mask)

    # 判断是否需要进行缩放
    if crack_width < w * 0.8 and crack_height < h * 0.8:
        # 裂缝的宽度或高度小于图像的 3/4 时，进行缩放
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = w // 2, h // 2
        else:
            cx, cy = w // 2, h // 2

        # 遍历每个缩放因子，生成对应的以裂缝为中心的图像
        for i in range(1, num_steps):
            crop_size = (int(w * (1 - i * zoom_factor)), int(h * (1 - i * zoom_factor)))
            x_start = max(cx - crop_size[0] // 2, 0)
            y_start = max(cy - crop_size[1] // 2, 0)
            expanded_crop_size = (crop_size[0] * 2, crop_size[1] * 2)
            x_start = max(cx - expanded_crop_size[0] // 2, 0)
            y_start = max(cy - expanded_crop_size[1] // 2, 0)
            x_end = min(x_start + expanded_crop_size[0], w)
            y_end = min(y_start + expanded_crop_size[1], h)

            cropped_image = image[y_start:y_end, x_start:x_end]
            zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_CUBIC)
            zoomed_image = cv2.convertScaleAbs(zoomed_image, alpha=1.5, beta=0)
            zoomed_images.append(zoomed_image)
    else:
        # 如果裂缝足够大，应用轻微的图像变换来生成序列
        for i in range(1, num_steps):
            transformed_image = enhance_image(image)  # 应用轻微变换
            zoomed_images.append(transformed_image)

    return zoomed_images

class CrackDataset(Dataset):
    def __init__(self, image_dir, label_dir, num_steps=5, zoom_factor=0.15):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.num_steps = num_steps
        self.zoom_factor = zoom_factor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.png'))
        # zoom_sequence = generate_zoom_in_sequence(image_path, self.num_steps, self.zoom_factor)
        zoom_sequence = zoom_in_and_save(image_path,label_path,self.num_steps, self.zoom_factor)
        
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # label = cv2.resize(label, (zoom_sequence[0].shape[1], zoom_sequence[0].shape[0]))

        zoom_sequence = np.array(zoom_sequence)
        zoom_sequence = torch.tensor(zoom_sequence).permute(0, 3, 1, 2) / 255.0
        # label = torch.tensor(label) / 255.0

        return zoom_sequence

# class CrackDataset(Dataset):
#     def __init__(self, image_dir, num_steps=5):
#         self.image_dir = image_dir
#         # self.label_dir = label_dir
#         self.image_files = os.listdir(image_dir)
#         self.num_steps = num_steps

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         # label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.jpg'))

#         # 加载原始图像和标签
#         image = cv2.imread(image_path)
#         # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#         # # 根据需要进行裁剪、缩放等基础处理
#         # image = cv2.resize(image, (256, 256))  # 示例，调整大小
#         # label = cv2.resize(label, (256, 256))

#         # 构建一个虚拟的时间序列（重复相同图像）
#         image_sequence = [image] * self.num_steps  # 这里创建相同图像的序列

#         # 转换为张量
#         image_sequence = np.array(image_sequence)
#         image_sequence = torch.tensor(image_sequence).permute(0, 3, 1, 2) / 255.0  # (steps, channels, height, width)
#         # label = torch.tensor(label) / 255.0

#         return image_sequence
# 定义模型加载函数
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)

# 定义测试函数
def test_model(model, dataloader, device, result_dir='results_uav_new1'):#results_DeepCrack
    model.eval()  # 设置模型为评估模式
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():  # 禁用梯度计算
        for i, image_sequences in enumerate(dataloader):
            image_sequences = image_sequences.float().to(device)

            outputs = model(image_sequences)
            # print(outputs.shape)
            outputs = outputs.squeeze(1).cpu().numpy()
            # print(outputs.shape)
            # output_image = outputs[-1]
            # 获取原始图像文件名
            original_image_file = dataloader.dataset.image_files[i]
            # print(outputs.shape)
            for j in range(outputs.shape[0]):
                output_image = outputs[j]

                # # 可视化输出
                # plt.figure(figsize=(5, 5))
                # plt.title('Predicted')
                # plt.imshow(output_image, cmap='gray')
                # plt.axis('off')

                # 使用原始文件名保存预测结果
                base_filename = os.path.splitext(original_image_file)[0]  # 去掉扩展名
                # output_file = os.path.join(result_dir, f'{base_filename}.png')

                cv2.imwrite(os.path.join(result_dir, '%s.png' % base_filename), output_image* 255 )
                print("Running test [%d/%d]" % (i + 1, len(dataloader)))    ### lk 2022.02.28

                # plt.savefig(output_file)
                # plt.close()

    print("Testing complete and results saved.")

# 使用示例
#UAV数据集
test_image_dir = '/data/zw2024/pro/dataset/crack_segmentation/test/images'
label_dir = '/data/zw2024/pro/dataset/crack_segmentation/test/labels'
#svrddd数据集
# test_image_dir = '/data/zw2024/pro/dataset/svrdd_pixel/test/images'
# label_dir = '/data/zw2024/pro/dataset/svrdd_pixel/test/gt'

# #Deepcrack数据集
# test_image_dir = '/data/zw2024/pro/dataset/DeepCrack/test_imgcopy'
# label_dir = '/data/zw2024/pro/dataset/DeepCrack/test_lab'
#EdmCrack600数据集
# test_image_dir = '/data/zw2024/pro/dataset/dataset-EdmCrack600/test/test_images_512'
# label_dir = '/data/zw2024/pro/dataset/dataset-EdmCrack600/test/test_lables_512'
# BJN260数据集
# test_image_dir = '/data/zw2024/pro/dataset/BJN260/test/test_img_copy'
# label_dir = '/data/zw2024/pro/dataset/BJN260/test/test_lable'

test_dataset = CrackDataset(test_image_dir,label_dir)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
import snn_new_728 # 确保将 SNNCrackSegmentation 定义在 snn_model.py 文件中
model = snn_new_728.SNN_C(input_channels=3, hidden_size=64, num_classes=1, time_windows=5)
model_path = 'uav_result_model/model_epoch_100.pth'#DeepCrack_result_model
load_model(model, model_path, device)

test_model(model, test_dataloader, device)
