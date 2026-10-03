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

    # # 添加随机噪声
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
#         if rect_w > 0.8 * w or rect_h > 0.8 * h:
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
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

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
    if crack_width < w * 0.7 and crack_height < h * 0.7:
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
# # 使用之前定义的序列生成器函数
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
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.png'))#UAV 标签是png
        # label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.jpg'))#DeeepCrack jpg
        # label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.png', '.png'))#Edm png
        # zoom_sequence = generate_zoom_in_sequence(image_path, self.num_steps, self.zoom_factor)
        zoom_sequence = zoom_in_and_save(image_path,label_path,self.num_steps, self.zoom_factor)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (zoom_sequence[0].shape[1], zoom_sequence[0].shape[0]))

        zoom_sequence = np.array(zoom_sequence)
        zoom_sequence = torch.tensor(zoom_sequence).permute(0, 3, 1, 2) / 255.0
        label = torch.tensor(label) / 255.0

        return zoom_sequence, label


# class CrackDataset(Dataset):
#     def __init__(self, image_dir, label_dir, num_steps=5):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.image_files = os.listdir(image_dir)
#         self.num_steps = num_steps

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.jpg'))

#         # 加载原始图像和标签
#         image = cv2.imread(image_path)
#         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#         # # 根据需要进行裁剪、缩放等基础处理
#         # image = cv2.resize(image, (256, 256))  # 示例，调整大小
#         # label = cv2.resize(label, (256, 256))

#         # 构建一个虚拟的时间序列（重复相同图像）
#         image_sequence = [image] * self.num_steps  # 这里创建相同图像的序列

#         # 转换为张量
#         image_sequence = np.array(image_sequence)
#         image_sequence = torch.tensor(image_sequence).permute(0, 3, 1, 2) / 255.0  # (steps, channels, height, width)
#         label = torch.tensor(label) / 255.0

#         return image_sequence, label

# 模型定义
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print(targets.shape)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
class bceLoss(nn.Module):
    def __init__(self):
        super(bceLoss, self).__init__()

    def forward(self, inputs, targets):
        # print(targets.shape)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        return nn.BCELoss()(inputs, targets)
class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.bce = bceLoss()
        self.dce = DiceLoss()
    def forward(self, inputs, targets):
        bceloss = self.bce(inputs, targets)
        diceloss = self.dce(inputs, targets)

        loss = 0.2*diceloss + 0.8*bceloss
        return loss
# 使用数据集和数据加载器
#UAV数据集
image_dir = '/data/zw2024/pro/dataset/crack_segmentation/train/images'
label_dir = '/data/zw2024/pro/dataset/crack_segmentation/train/labels'
#svrddd数据集
# image_dir = '/data/zw2024/pro/dataset/svrdd_pixel/train/images_four'
# label_dir = '/data/zw2024/pro/dataset/svrdd_pixel/train/gt_four'
# # #Deepcrack数据集
# image_dir = '/data/zw2024/pro/dataset/DeepCrack/train_img_four'
# label_dir = '/data/zw2024/pro/dataset/DeepCrack/train_lab_four'
#edmcrack600数据集
# image_dir = '/data/zw2024/pro/dataset/dataset-EdmCrack600/train/train_images_four_512'
# label_dir = '/data/zw2024/pro/dataset/dataset-EdmCrack600/train/train_lables_four_512'
#BJN260数据集
# image_dir = '/data/zw2024/pro/dataset/BJN260/train_img_four'
# label_dir = '/data/zw2024/pro/dataset/BJN260/train_lable_four'
#Rain365数据集
# image_dir = '/data/zw2024/pro/dataset/Rain365/train_four-orignal'
# label_dir = '/data/zw2024/pro/dataset/Rain365/train_four_lable-orignal'

dataset = CrackDataset(image_dir, label_dir, num_steps=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True,pin_memory=True)

# 使用 SNNCrackSegmentation 代替 SNN_vgg 模型
# import snn_ICME  # 确保将 SNNCrackSegmentation 定义在 snn_model.py 文件中
import snn_new_728
model = snn_new_728.SNN_C(input_channels=3, hidden_size=64, num_classes=1, time_windows=5)

# 损失函数和优化器
criterion = FusionLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 设备配置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')
# 创建保存模型权重的文件夹
os.makedirs('uav_result_model', exist_ok=True)

# 训练函数
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=100, print_every=100, save_every=5):
    model.to(device)  # 将模型移动到GPU

    loss_history = []  # 记录每个epoch的loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0  # 计数器，记录已经处理的batch数量

        for i, (image_sequences, labels) in enumerate(dataloader):
            # 将数据移动到GPU
            image_sequences = image_sequences.float().to(device)
            labels = labels.float().to(device)
            
            # 修改输入维度 [batch_size, steps, channels, height, width] -> [batch_size * steps, channels, height, width]
            # image_sequences = image_sequences.view(-1, image_sequences.size(2), image_sequences.size(3), image_sequences.size(4))
            # print(labels)
            # 打印最大值
            optimizer.zero_grad()
            # with torch.amp.autocast('cuda'):
            outputs = model(image_sequences)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()


            running_loss += loss.item() * image_sequences.size(0)
            batch_count += 1

            # 每处理指定数量的batch输出一次loss
            if batch_count % print_every == 0:
                avg_loss = running_loss / (print_every * dataloader.batch_size)
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_count}, Loss: {avg_loss:.4f}')
                running_loss = 0.0  # 重置运行loss

        # 在每个epoch结束时，输出一次总体loss
        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 每5个epoch保存一次模型权重
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join('uav_result_model', f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model weights saved to {checkpoint_path}')

    # 保存最终模型
    final_model_path = os.path.join('uav_result_model', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model weights saved to {final_model_path}')

    # 绘制损失曲线并保存
    plt.figure()
    plt.plot(range(num_epochs), loss_history, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid()
    plt.savefig('uav_loss_curve.png')
    plt.show()

# 开始训练
train_model(model, dataloader, criterion, optimizer, device)
