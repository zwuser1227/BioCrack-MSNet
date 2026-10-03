import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================
# 路径设置
# ==========================
pred_dir = r"E:/zw2023/project2023/CarNet-V1.0-main/svrdd_aug_CarNet/onescale_test_15e3"
gt_dir   = r"F:/dataset/SVRDD_Pixel/test/gt_255"

# ==========================
# Crack Density
# ==========================
def crack_density(mask):
    crack_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    return crack_pixels / total_pixels


# ==========================
# 主程序
# ==========================
cd_pred_list = []
cd_gt_list = []
errors = []

records = []

pred_files = sorted(os.listdir(pred_dir))

for name in tqdm(pred_files):

    pred_path = os.path.join(pred_dir, name)
    gt_path   = os.path.join(gt_dir, name)

    if not os.path.exists(gt_path):
        print(f"Missing GT: {name}")
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        print(f"Cannot read: {name}")
        continue

    # 二值化
    pred = (pred > 127).astype(np.uint8)
    gt   = (gt > 127).astype(np.uint8)

    # Crack Density
    cd_pred = crack_density(pred)
    cd_gt   = crack_density(gt)

    cd_pred_list.append(cd_pred)
    cd_gt_list.append(cd_gt)

    errors.append(cd_pred - cd_gt)

    records.append([name, cd_gt, cd_pred, cd_pred - cd_gt])


# ==========================
# 转 numpy
# ==========================
errors = np.array(errors)
cd_pred_list = np.array(cd_pred_list)
cd_gt_list = np.array(cd_gt_list)

# ==========================
# CD-MAE
# ==========================
cd_mae = np.mean(np.abs(errors))

# ==========================
# CD-RMSE
# ==========================
cd_rmse = np.sqrt(np.mean(errors ** 2))

# ==========================
# Pearson Correlation
# ==========================
pearson = np.corrcoef(cd_pred_list, cd_gt_list)[0, 1]

# ==========================
# 输出结果
# ==========================
print("=" * 60)
print(f"Number of Images : {len(errors)}")
print(f"CD-MAE          : {cd_mae:.6f}")
print(f"CD-RMSE         : {cd_rmse:.6f}")
print(f"Pearson r       : {pearson:.6f}")
print("=" * 60)


# ==========================
# 保存CSV（论文用）
# ==========================
df = pd.DataFrame(
    records,
    columns=["Image", "CD_GT", "CD_PRED", "ERROR"]
)

save_path = "CD_results.csv"
df.to_csv(save_path, index=False)

print(f"Results saved to: {save_path}")