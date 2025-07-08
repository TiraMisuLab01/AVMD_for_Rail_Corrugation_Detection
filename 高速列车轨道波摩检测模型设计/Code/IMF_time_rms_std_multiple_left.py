import glob
import numpy as np
import pandas as pd
from ICF_location import icf_location
from IMF import imf
"""
一次运行计算多个数据集的RMS,STD
"""
def calculate_rms_std(data, icf_location, imf, FS, alpha, tol):
    """
    :param data: 数据集
    :param icf_location: 最佳的VMD分解中心频率
    :param imf: 时域信号
    :param FS: 采样频率
    :param alpha: 平衡参数
    :param tol: 容忍度
    :return:
    """
    # 计算左轨
    left_icf = icf_location(data['Left'], FS, alpha)
    K_left = len(left_icf)
    imfs_left = imf(data['Left'], FS, alpha, tol, left_icf)
    rms_values_left = [np.sqrt(np.mean(np.abs(imf) ** 2)) for imf in imfs_left]
    std_values_left = [np.std(imf) for imf in imfs_left]
    return rms_values_left, std_values_left, K_left
# 文件名规律
# file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\混合数据\混合数据*.xlsx'
file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\正常区段\正常数据*.xlsx'  # 波摩检测阈值使用的正常路段的RMS+STD各自的平均值之和
# file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\波磨区段\波磨数据*.xlsx'
# 使用 glob 模块获取匹配的文件列表
file_names = glob.glob(file_pattern)
FS = 2000  # 采样频率
alpha = 4000  # 平衡参数
tol = 1e-7  # 收敛标准的容忍度；通常在1e-6左右
# 创建一个用于存储结果的空列表
results = []
rms = 0
std = 0
# 遍历文件名列表
for file_name in file_names:
    # 读取 Excel 文件
    df = pd.read_excel(file_name)
    # 提取数据
    data = {'Left': df['Left'].values}  # 假设 Excel 文件中有 'Left' 列
    # 计算计算左轨RMS和标准差
    rms_values_left, std_values_left, K_left = calculate_rms_std(data, icf_location, imf, FS, alpha, tol)
    # 只取第一个
    rms_values_left = rms_values_left[0]
    std_values_left = std_values_left[0]
    # 将结果存储到列表中
    result = {
        'File': file_name,
        'RMS_Left': rms_values_left,
        'STD_Left': std_values_left,
        'Total_IMFs_Left': K_left
    }
    rms += rms_values_left
    std += std_values_left
    results.append(result)
print(rms / 53 + std / 53)  # 波磨阈值
# 将结果列表转换为DataFrame
result_df = pd.DataFrame(results)
# 将DataFrame写入到Excel文件
result_df.to_excel('normal_result_left_output.xlsx', index=False)
# result_df.to_excel('bomo_result_left_output.xlsx', index=False)
