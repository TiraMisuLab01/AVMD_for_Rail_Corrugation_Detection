import glob
import numpy as np
import pandas as pd
# from ICF_location import icf_location
# from IMF import imf
from vmdpy import VMD # 导入 vmdpy 库中的 VMD 函数

"""
一次运行计算多个数据集的RMS,STD
"""


def calculate_rms_std_fixed_k(signal_data, K, alpha, tol, FS):
    """
    使用固定的K值执行VMD并计算每个模态的RMS和STD。
    :param signal_data: 输入的时域信号 (numpy array)
    :param K: 固定的模态数
    :param alpha: VMD的平衡参数 (二次惩罚项)
    :param tol: VMD的收敛容忍度
    :param FS: 采样频率 (虽然VMD本身不直接用FS，但保留以防未来需要)
    :return: 各模态的RMS列表, 各模态的STD列表
    """
    # VMD 参数设置
    tau = 0.  # 时间步长/噪声松弛因子 (通常设为0)
    DC = 0  # 是否包含直流分量 (0: 不包含)
    init = 1  # 初始化中心频率的方式 (1: 均匀初始化)
    # 执行 VMD 分解
    # VMD 函数返回: imfs (模态), imf_hat (模态的频域表示), omega (中心频率历史)
    imfs, _, _ = VMD(signal_data, alpha, tau, K, DC, init, tol)
    # # --- 如果你的原始数据非常长，内存可能成为问题，可以考虑分段处理 ---
    # # --- 或者如果 VMD 实现支持，看是否有不需要存储所有历史记录的选项 ---

    # 计算每个模态的 RMS 和 STD
    rms_values = []
    std_values = []
    if imfs is not None and len(imfs) > 0:
        rms_values = [np.sqrt(np.mean(np.abs(mode) ** 2)) for mode in imfs]
        std_values = [np.std(mode) for mode in imfs]

    # 如果VMD未能成功分解（有时会发生，特别是信号质量差或参数不合适时）
    # 返回空列表或者根据需要处理
    if not rms_values:
        print(f"警告: VMD未能成功分解信号，或者返回了空的模态列表。")
        # 可以选择返回 K 个 NaN 值，或者空列表
        rms_values = [np.nan] * K
        std_values = [np.nan] * K
    return rms_values, std_values

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
print(len(file_names))
FS = 2000  # 采样频率
alpha = 4000  # 平衡参数
tol = 1e-7  # 收敛标准的容忍度；通常在1e-6左右
fixed_K = 1  # <--- 在这里设置你想要的固定模态数 K

# 创建一个用于存储结果的空列表
results = []
rms = 0
std = 0
# 创建一个用于存储结果的空列表
results = []
total_rms_sum = 0  # 用于累加第一个模态的RMS
total_std_sum = 0  # 用于累加第一个模态的STD
processed_files = 0  # 计数成功处理的文件
# 遍历文件名列表
for file_name in file_names:
    print(f"正在处理文件: {file_name}")
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_name)

        # 检查 'Left' 列是否存在
        if 'Left' not in df.columns:
            print(f"警告：文件 {file_name} 中没有找到 'Left' 列，跳过此文件。")
            continue  # 跳到下一个文件
        # 提取数据，并确保是 numpy array
        left_signal = df['Left'].values.astype(float)
        # 检查信号是否有效 (例如，非空，非NaN)
        if left_signal is None or len(left_signal) == 0 or np.isnan(left_signal).all():
            print(f"警告：文件 {file_name} 中的 'Left' 列数据无效或为空，跳过此文件。")
            continue

        # --- 调用固定K值的VMD计算函数 ---
        rms_values_left, std_values_left = calculate_rms_std_fixed_k(left_signal, fixed_K, alpha, tol, FS)
        # 检查返回结果是否有效 (例如，不是全部为 NaN)
        if all(np.isnan(rms_values_left)) or not rms_values_left:
            print(f"警告：文件 {file_name} 的VMD分解失败或未返回有效结果，跳过此文件的统计。")
            # 你可以选择是否将这个文件的失败记录也存入 result
            result = {
                'File': file_name,
                'RMS_Left': np.nan,  # 使用 Mode1 来明确是第一个模态
                'STD_Left': np.nan,
                'Fixed_K': fixed_K,
                'Error': 'VMD Failed'
            }
            results.append(result)
            continue  # 跳到下一个文件
        # 只取第一个模态的RMS和STD (根据你原始代码的逻辑)
        # 注意：这里假设你总是关心第一个模态。如果需要所有模态，需要修改存储方式。
        first_mode_rms_left = rms_values_left[0] if len(rms_values_left) > 0 else np.nan
        first_mode_std_left = std_values_left[0] if len(std_values_left) > 0 else np.nan

        # 累加用于计算平均阈值 (仅在成功处理时累加)
        if not np.isnan(first_mode_rms_left) and not np.isnan(first_mode_std_left):
            total_rms_sum += first_mode_rms_left
            total_std_sum += first_mode_std_left
            processed_files += 1
        else:
            print(f"警告：文件 {file_name} 第一个模态的RMS/STD计算结果为NaN。")
        # 将结果存储到列表中
        result = {
            'File': file_name,
            'RMS_Left': first_mode_rms_left,  # 明确是第一个模态
            'STD_Left': first_mode_std_left,  # 明确是第一个模态
            'Fixed_K': fixed_K  # 记录使用的固定K值
            # 如果需要，也可以保存所有模态的结果
            # 'All_RMS_Left': rms_values_left,
            # 'All_STD_Left': std_values_left
        }
        results.append(result)
    except FileNotFoundError:
        print(f"错误：文件 {file_name} 未找到。")
    except Exception as e:
        print(f"处理文件 {file_name} 时发生错误: {e}")
        # 考虑是否将错误信息也记录下来
        result = {
            'File': file_name,
            'RMS_Left': np.nan,
            'STD_Left': np.nan,
            'Fixed_K': fixed_K,
            'Error': str(e)
        }
        results.append(result)
# 计算并打印平均阈值 (基于成功处理的文件)
if processed_files > 0:
    average_threshold = (total_rms_sum / processed_files) + (total_std_sum / processed_files)
    print(f"\n基于 {processed_files} 个成功处理的文件，计算得到的平均阈值 (RMS_avg + STD_avg): {average_threshold}")
else:
    print("\n没有文件被成功处理，无法计算平均阈值。")
# 将结果列表转换为DataFrame
result_df = pd.DataFrame(results)
# 将DataFrame写入到Excel文件 (取消注释并修改为你想要的文件名)
# output_filename = f'fixed_k{fixed_K}_vmd_results_left.xlsx'
output_filename = 'normal_result_left_fixed_k_output.xlsx' # 或者根据数据类型命名
# output_filename = 'bomo_result_left_fixed_k_output.xlsx'  # 或者根据数据类型命名
try:
    result_df.to_excel(output_filename, index=False)
    print(f"\n结果已保存到文件: {output_filename}")
except Exception as e:
    print(f"\n错误：无法将结果写入Excel文件 {output_filename}。错误信息: {e}")
