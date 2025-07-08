import glob
import numpy as np
import pandas as pd
# 不再需要 icf_location 和 原始的 imf
from vmdpy import VMD  # 导入 vmdpy 库中的 VMD 函数

"""
一次运行计算多个数据集的RMS,STD (固定K值 VMD版本 - 处理右轨数据)
"""


def calculate_rms_std_fixed_k(signal_data, K, alpha, tol, FS):
    """
    使用固定的K值执行VMD并计算每个模态的RMS和STD。
    (此函数与左轨版本相同，因为它处理的是通用的 signal_data)

    :param signal_data: 输入的时域信号 (numpy array)
    :param K: 固定的模态数
    :param alpha: VMD的平衡参数 (二次惩罚项)
    :param tol: VMD的收敛容忍度
    :param FS: 采样频率 (VMD本身不直接用，但保留)
    :return: 各模态的RMS列表, 各模态的STD列表
    """
    # VMD 参数设置
    tau = 0.  # 时间步长/噪声松弛因子 (通常设为0)
    DC = 0  # 是否包含直流分量 (0: 不包含)
    init = 1  # 初始化中心频率的方式 (1: 均匀初始化)

    # 执行 VMD 分解
    imfs, _, _ = VMD(signal_data, alpha, tau, K, DC, init, tol)

    # 计算每个模态的 RMS 和 STD
    rms_values = []
    std_values = []
    if imfs is not None and len(imfs) > 0:
        rms_values = [np.sqrt(np.mean(np.abs(mode) ** 2)) for mode in imfs]
        std_values = [np.std(mode) for mode in imfs]

    # 处理VMD分解失败或返回空模态的情况
    if not rms_values:
        print(f"警告: VMD未能成功分解信号，或者返回了空的模态列表。")
        rms_values = [np.nan] * K
        std_values = [np.nan] * K

    return rms_values, std_values


# === 主程序部分 ===

# 文件名规律 (请确保路径正确)
# file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\混合数据\混合数据*.xlsx'
file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\正常区段\正常数据*.xlsx'  # 波摩检测阈值使用的正常路段的RMS+STD各自的平均值之和
# file_pattern = 'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Dataset\波磨区段\波磨数据*.xlsx'

# 使用 glob 模块获取匹配的文件列表
file_names = glob.glob(file_pattern)
if not file_names:
    print(f"错误：在路径 '{file_pattern}' 下没有找到任何匹配的文件。请检查路径和文件名模式。")
    exit()

print(f"找到 {len(file_names)} 个文件进行处理。")

# --- 参数设置 ---
FS = 2000  # 采样频率 (Hz)
alpha = 4000  # VMD: 平衡参数
tol = 1e-7  # VMD: 收敛容忍度
fixed_K = 5  # <--- 在这里设置你想要的固定模态数 K

# 创建一个用于存储结果的空列表
results = []
total_rms_sum = 0  # 用于累加右轨第一个模态的RMS
total_std_sum = 0  # 用于累加右轨第一个模态的STD
processed_files = 0  # 计数成功处理的文件

# 遍历文件名列表
for file_name in file_names:
    print(f"正在处理文件: {file_name}")
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_name)

        # === 修改点：检查 'Right' 列是否存在 ===
        if 'Right' not in df.columns:
            print(f"警告：文件 {file_name} 中没有找到 'Right' 列，跳过此文件。")
            continue

            # === 修改点：提取 'Right' 列数据 ===
        right_signal = df['Right'].values.astype(float)

        # 检查信号是否有效
        if right_signal is None or len(right_signal) == 0 or np.isnan(right_signal).all():
            print(f"警告：文件 {file_name} 中的 'Right' 列数据无效或为空，跳过此文件。")
            continue

        # --- 调用固定K值的VMD计算函数 ---
        # === 修改点：传入 right_signal ===
        rms_values_right, std_values_right = calculate_rms_std_fixed_k(right_signal, fixed_K, alpha, tol, FS)

        # 检查返回结果是否有效
        if all(np.isnan(rms_values_right)) or not rms_values_right:
            print(f"警告：文件 {file_name} (右轨) 的VMD分解失败或未返回有效结果，跳过此文件的统计。")
            result = {
                'File': file_name,
                'RMS_Right': np.nan,  # 使用 Right
                'STD_Right': np.nan,  # 使用 Right
                'Fixed_K': fixed_K,
                'Error': 'VMD Failed'
            }
            results.append(result)
            continue

        # 只取第一个模态的RMS和STD
        # === 修改点：使用 right 的结果 ===
        first_mode_rms_right = rms_values_right[0] if len(rms_values_right) > 0 else np.nan
        first_mode_std_right = std_values_right[0] if len(std_values_right) > 0 else np.nan

        # 累加用于计算平均阈值 (仅在成功处理时累加)
        if not np.isnan(first_mode_rms_right) and not np.isnan(first_mode_std_right):
            total_rms_sum += first_mode_rms_right  # 使用 right 的 RMS
            total_std_sum += first_mode_std_right  # 使用 right 的 STD
            processed_files += 1
        else:
            print(f"警告：文件 {file_name} (右轨) 第一个模态的RMS/STD计算结果为NaN。")

        # 将结果存储到列表中
        # === 修改点：修改字典键名 ===
        result = {
            'File': file_name,
            'RMS_Right': first_mode_rms_right,  # 明确是右轨第一个模态
            'STD_Right': first_mode_std_right,  # 明确是右轨第一个模态
            'Fixed_K': fixed_K
            # 如果需要，也可以保存所有模态的结果
            # 'All_RMS_Right': rms_values_right,
            # 'All_STD_Right': std_values_right
        }
        results.append(result)

    except FileNotFoundError:
        print(f"错误：文件 {file_name} 未找到。")
    except Exception as e:
        print(f"处理文件 {file_name} 时发生错误: {e}")
        result = {
            'File': file_name,
            'RMS_Right': np.nan,
            'STD_Right': np.nan,
            'Fixed_K': fixed_K,
            'Error': str(e)
        }
        results.append(result)

# 计算并打印平均阈值 (基于成功处理的文件)
if processed_files > 0:
    average_threshold = (total_rms_sum / processed_files) + (total_std_sum / processed_files)
    # === 修改点：更新打印信息 ===
    print(
        f"\n基于 {processed_files} 个成功处理的文件 (右轨数据)，计算得到的平均阈值 (RMS_avg + STD_avg): {average_threshold}")
else:
    print("\n没有文件被成功处理，无法计算平均阈值。")

# 将结果列表转换为DataFrame
result_df = pd.DataFrame(results)

# 将DataFrame写入到Excel文件
# === 修改点：更新输出文件名 ===
# output_filename = f'fixed_k{fixed_K}_vmd_results_right.xlsx'
# 你也可以根据数据类型命名，例如:
output_filename = 'normal_result_right_fixed_k_output.xlsx'
# output_filename = 'bomo_result_right_fixed_k_output.xlsx'
try:
    result_df.to_excel(output_filename, index=False)
    print(f"\n结果已保存到文件: {output_filename}")
except Exception as e:
    print(f"\n错误：无法将结果写入Excel文件 {output_filename}。错误信息: {e}")



