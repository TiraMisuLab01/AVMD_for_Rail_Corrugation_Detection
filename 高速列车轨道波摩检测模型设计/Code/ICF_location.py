# 导入必要的库
import numpy as np  # NumPy 用于数值计算
from scipy.io import loadmat  # 从 SciPy 导入 loadmat 函数，用于读取 MATLAB 文件

# 定义函数 ICF_location，用于计算信号的瞬时中心频率（ICF）的位置
def icf_location(sig_x, FS, Alpha):
    # sig_x: 输入信号
    # FS: 采样频率
    # Alpha: 一个常数，用于计算频率响应

    T = len(sig_x)  # 获取信号的长度
    t = np.arange(1, T+1) / T  # 生成一个归一化的时间向量
    freqs = t - 0.5 - 1/T  # 计算频率向量，范围从 -0.5 到 0.5

    # 计算信号的快速傅里叶变换（FFT）并进行频率移位
    f_hat = np.fft.fftshift(np.fft.fft(sig_x))

    # 保留正频率部分
    f_hat_plus = f_hat
    f_hat_plus[:T//2] = 0

    # 计算用于瞬时中心频率（ICF）计算的频率向量 omega_plus
    omega_plus = np.arange(0, 0.5-1/T+1, 5/FS)

    # 计算频率响应 u_hat_plus
    u_hat_plus = f_hat_plus / (1 + Alpha * (freqs - omega_plus[:, np.newaxis])**2)

    # 计算瞬时中心频率 omega_plus1
    omega_plus1 = np.sum(freqs[T//2+1:] * np.abs(u_hat_plus[:, T//2+1:]) ** 2, axis=1) / np.sum(np.abs(u_hat_plus[:, T//2+1:]) ** 2, axis=1)

    # 找到瞬时中心频率的位置
    ee = (omega_plus1 - omega_plus) * FS  # 计算频率偏移
    i = 1  # 初始化索引变量
    k = 1  # 初始化计数器
    I = []  # 初始化空列表，用于存储找到的 ICF 位置
    while i < len(ee):  # 遍历 ee
        if ee[i] > 0 and ee[i+1] < 0:  # 如果发现零点
            I.append(omega_plus[i])  # 将该位置添加到列表 I 中
            k += 1  # 计数器增加
        i += 1  # 索引变量增加
    ICF = np.array(I) * FS  # 将找到的位置转换为频率
    return ICF  # 返回瞬时中心频率的位置

# # 加载 .mat 文件中的信号数据
# mat_data = loadmat('sig_x.mat')  # 加载 .mat 文件
# sig_x = mat_data['ME1'][0, :]  # 从字典中提取信号数据
#
# # 将信号数据保存为 .npy 文件并重新加载
# np.save('sig_x.npy', sig_x)  # 保存信号数据为 .npy 文件
# sig_x = np.load('sig_x.npy')  # 重新加载 .npy 文件
#
# # 调用 ICF_location 函数并打印结果
# re = ICF_location(sig_x, 10000, 1000)  # 使用采样频率为 10000 Hz，常数 Alpha 为 1000 调用函数
# print(re)  # 打印计算得到的瞬时中心频率的位置
