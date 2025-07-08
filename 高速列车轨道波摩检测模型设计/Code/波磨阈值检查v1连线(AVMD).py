# 单   位： 苏州大学
# 作   者： 许圣
# 开发时间： 2025/4/13 12:46
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# 左轨
data_path1 = r'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Code\normal_result_left_output.xlsx'
data1 = pd.read_excel(data_path1)
rms1 = data1['RMS_Left']

data_path2 = r'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Code\bomo_result_left_output.xlsx'
data2 = pd.read_excel(data_path2)
rms2 = data2['RMS_Left']
# # 右轨
# data_path1 = r'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Code\normal_result_right_output.xlsx'
# data1 = pd.read_excel(data_path1)
# rms1 = data1['RMS_Right']
# data_path2 = r'E:\桌面文件存放\高速列车轨道波摩检测模型设计\Code\bomo_result_right_output.xlsx'
# data2 = pd.read_excel(data_path2)
# rms2 = data2['RMS_Right']

# 需要显示为蓝色的索引 (rms2 中的索引)
blue_indices = {0, 1, 2, 3, 4, 5, 6, 7, 32, 49, 50, 51} # 使用集合以便快速查找

meter1 = np.linspace(1000, 2000, len(rms1)) # 确保长度匹配
# 确保 rms2 至少有 2 个点才能计算间距
if len(rms2) >= 2:
    meter2 = np.linspace(2000, 3000, len(rms2))
    # 计算 meter2 中点与点之间的间距 (一个“单元”)
    shift_amount = meter2[1] - meter2[0]
    # 将 meter2 整体向右移动一个单元
    meter2 = meter2 + shift_amount
elif len(rms2) == 1:
     # 如果只有一个点，可以给一个默认的小偏移量，或者就放在 2000
     meter2 = np.array([2000.0 + 10.0]) # 比如偏移10m，或者不偏移
     shift_amount = 10.0 # 记录偏移量
else:
     meter2 = np.array([]) # rms2 为空
     shift_amount = 0
# meter2 = np.linspace(2000, 3000, len(rms2)) # 确保长度匹配
check = 2.0492137596112463  # 左轨波磨阈值，由`IMF_time_rms_std_multiple_left.py`计算正常路段的RMS与STD的平均值之和
# check = 1.620126212775014  # 右轨波磨阈值，由`IMF_time_rms_std_multiple_right.py`计算正常路段的RMS与STD的平均值之和

# # --- 绘图设置 ---
# plt.figure(figsize=(10, 5))
#
# # --- 绘制基础线条 ---
# plt.axhline(y=check, color='red', linestyle='--', label='阈值')
# plt.plot(meter1, rms1, label='正常区段', color='blue') # 绘制第一个数据集
#
# # --- 准备绘制第二个数据集 (rms2) ---
# # 创建完整的索引列表
# all_indices = list(range(len(rms2)))
# red_indices = {i for i in all_indices if i not in blue_indices} # 红色的索引集合
#
# # --- 修改后的绘制逻辑：遍历所有连接段 ---
# blue_label_added = False # 蓝色部分不需要单独标签，但保留以备万一
# red_label_added = False  # 标志位，确保“波磨路段”标签只添加一次
#
# for i in range(len(rms2) - 1):
#     idx1 = i
#     idx2 = i + 1
#
#     x_coords = [meter2[idx1], meter2[idx2]]
#     y_coords = [rms2[idx1], rms2[idx2]]
#
#     # 判断点 idx1 和 idx2 的颜色归属
#     is_idx1_blue = idx1 in blue_indices
#     is_idx2_blue = idx2 in blue_indices
#
#     # 根据规则确定连接线颜色
#     line_color = 'red' # 默认红色 (红接红, 蓝接红)
#     if is_idx1_blue and is_idx2_blue:
#         line_color = 'blue' # 蓝接蓝
#     elif not is_idx1_blue and is_idx2_blue:
#         line_color = 'blue' # 红接蓝
#
#     # 处理标签：只为第一个红色线段添加标签
#     current_segment_is_red = (not is_idx1_blue) or (not is_idx2_blue) # 只要有一个点是红，线段就算作红的一部分
#     segment_label = None
#     if line_color == 'red' and not red_label_added: # 仅当线是红色且标签未添加时
#          # 为了确保标签给的是一段连续的红色线段，而不是一个过渡线，
#          # 我们检查起始点是否为红色
#          if not is_idx1_blue:
#             segment_label = '波磨路段'
#             red_label_added = True
#
#     plt.plot(x_coords, y_coords, color=line_color, label=segment_label)
#
#
# # --- 计算和显示检测率部分 (移除手动图例句柄) ---
# red_point_count = len(red_indices)
# over_threshold_count = 0
# for idx in red_indices:
#     if rms2[idx] > check:
#         over_threshold_count += 1
#
# if red_point_count > 0:
#     detection_rate_percent = (over_threshold_count / red_point_count) * 100
#     print(f"红色点总数: {red_point_count}")
#     print(f"大于阈值的红色点数: {over_threshold_count}")
#     print(f"检测率: {detection_rate_percent / 100:.4f}")
#     # 在图中添加文本
#     plt.text(0.05, 0.75, f"波磨检测率: {detection_rate_percent:.2f}%",
#              transform=plt.gca().transAxes, fontsize=10, # 使用相对坐标定位文本
#              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)) # 加个背景框
# else:
#     print("没有红色点，无法计算检测率")
#     plt.text(0.05, 0.9, "无波磨数据点",
#              transform=plt.gca().transAxes, fontsize=10,
#              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
#
#
# # --- 设置图表属性 ---
# # plt.title('左侧轨道全区段信号故障模态检测')
# plt.title('左侧轨道全区段信号故障模态检测')
# plt.xlabel('路程 /m')
# plt.ylabel('故障模态均方根值 $m/s^2$')
# plt.ylim(0, 4.5)
# plt.grid(True)
#
# # --- 显示最终图例 ---
# # Matplotlib 会自动收集所有带 label 的绘图元素
# plt.legend()
# plt.show()

# --- 绘图设置 ---
plt.figure(figsize=(10, 5))
# --- 绘制基础线条 ---
plt.axhline(y=check, color='red', linestyle='--', label='阈值')
# 绘制第一个数据集 (1000-2000m)
# 只有在 rms1 有数据时才绘制
if len(rms1) > 0:
    plt.plot(meter1, rms1, color='blue', label='正常区段')
# --- !!! 新增：绘制连接线，如果两个数据集都存在 !!! ---
if len(rms1) > 0 and len(rms2) > 0:
    # 获取连接点坐标
    x1_connect = meter1[-1]
    y1_connect = rms1.iloc[-1]  # 使用 .iloc 访问 Pandas Series
    x2_connect = meter2[0]  # 使用偏移后的 meter2 的第一个点
    y2_connect = rms2.iloc[0]  # 使用 .iloc 访问 Pandas Series
    # 判断 rms2 起始点的颜色
    is_rms2_start_blue = 0 in blue_indices  # 检查索引 0 是否在蓝点集合中

    # 根据规则确定连接线颜色：蓝->蓝 = 蓝；蓝->红 = 红
    connection_color = 'blue' if is_rms2_start_blue else 'red'

    # 绘制连接线，不带标签
    plt.plot([x1_connect, x2_connect], [y1_connect, y2_connect], color=connection_color)
# --- 新增结束 ---
# --- 绘制第二个数据集的逻辑 ---
all_indices = list(range(len(rms2)))
red_indices = {i for i in all_indices if i not in blue_indices}  # 红色的索引集合
red_label_added = False  # 标志位，确保“波磨路段”标签只添加一次
# 循环绘制 rms2 内部的线段
for i in range(len(rms2) - 1):
    idx1 = i
    idx2 = i + 1
    x_coords = [meter2[idx1], meter2[idx2]]
    y_coords = [rms2.iloc[idx1], rms2.iloc[idx2]]  # 使用 .iloc
    # 判断点 idx1 和 idx2 的颜色归属
    is_idx1_blue = idx1 in blue_indices
    is_idx2_blue = idx2 in blue_indices
    # 根据规则确定连接线颜色
    line_color = 'red'  # 默认红色 (红接红, 蓝接红)
    if is_idx1_blue and is_idx2_blue:
        line_color = 'blue'  # 蓝接蓝
    elif not is_idx1_blue and is_idx2_blue:
        # 注意：红点连接到蓝点，线段本身是蓝色的
        line_color = 'blue'
        # else: # 蓝接红 或 红接红 都是红色，已是默认值
    # 处理标签：只为“第一个”红色线段添加标签
    segment_label = None
    if line_color == 'red' and not red_label_added:
        # 确保标签加在实际红色段落的开始
        # 条件：当前线段是红色的，并且它的起始点是红色的, 且标签未添加
        if not is_idx1_blue:  # 如果线段的起始点是红色的
            segment_label = '波磨路段'
            red_label_added = True
        # elif is_idx1_blue and not is_idx2_blue: # 如果是从蓝到红的过渡，也算波磨开始
        #     segment_label = '波磨路段'
        #     red_label_added = True
        # 上面这种会让蓝到红的第一条红线得到标签，如果想严格从红点开始，就只用 if not is_idx1_blue
    plt.plot(x_coords, y_coords, color=line_color, label=segment_label)
# --- 计算和显示检测率部分 ---
red_point_count = len(red_indices)
over_threshold_count = 0
for idx in red_indices:
    if rms2.iloc[idx] > check:
        over_threshold_count += 1
if red_point_count > 0:
    detection_rate_percent = (over_threshold_count / red_point_count) * 100
    print(f"红色点总数: {red_point_count}")
    print(f"大于阈值的红色点数: {over_threshold_count}")
    print(f"检测率: {detection_rate_percent / 100:.4f}")
    plt.text(0.05, 0.75, f"波磨检测率: {detection_rate_percent:.2f}%",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
else:
    print("没有红色点，无法计算检测率")
    plt.text(0.05, 0.9, "无波磨数据点",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
# --- 设置图表属性 ---
plt.title('左侧轨道全区段信号故障模态检测')
plt.xlabel('路程 /m')
plt.ylabel('故障模态均方根值 $m/s^2$')
plt.ylim(0, 4.5)
plt.grid(True)
# 调整x轴范围以适应偏移
if len(meter2) > 0:
    plt.xlim(right=meter2[-1] + shift_amount)  # 稍微扩大右边界
# --- 显示最终图例 ---
plt.legend()
plt.show()

