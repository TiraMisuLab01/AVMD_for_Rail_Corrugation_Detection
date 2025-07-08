# 单   位： 苏州大学
# 作   者： 许圣
# 开发时间： 2024/7/31 下午5:43
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftshift

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# 定义函数 ICF_location，用于计算信号的瞬时中心频率（ICF）的位置
def icf_location(sig_x, FS, Alpha):
    # sig_x: 输入信号
    # FS: 采样频率
    # Alpha: 一个常数，用于计算频率响应

    T = len(sig_x)  # 获取信号的长度
    t = np.arange(1, T + 1) / T  # 生成一个归一化的时间向量
    freqs = t - 0.5 - 1 / T  # 计算频率向量，范围从 -0.5 到 0.5

    # 计算信号的快速傅里叶变换（FFT）并进行频率移位
    f_hat = np.fft.fftshift(np.fft.fft(sig_x))

    # 保留正频率部分
    f_hat_plus = f_hat
    f_hat_plus[:T // 2] = 0

    # 计算用于瞬时中心频率（ICF）计算的频率向量 omega_plus
    omega_plus = np.arange(0, 0.5 - 1 / T + 1, 5 / FS)

    # 计算频率响应 u_hat_plus
    u_hat_plus = f_hat_plus / (1 + Alpha * (freqs - omega_plus[:, np.newaxis]) ** 2)

    # 计算瞬时中心频率 omega_plus1
    omega_plus1 = np.sum(freqs[T // 2 + 1:] * np.abs(u_hat_plus[:, T // 2 + 1:]) ** 2, axis=1) / np.sum(
        np.abs(u_hat_plus[:, T // 2 + 1:]) ** 2, axis=1)

    # 找到瞬时中心频率的位置
    ee = (omega_plus1 - omega_plus) * FS  # 计算频率偏移
    i = 1  # 初始化索引变量
    k = 1  # 初始化计数器
    I = []  # 初始化空列表，用于存储找到的 ICF 位置
    while i < len(ee):  # 遍历 ee
        if ee[i] > 0 and ee[i + 1] < 0:  # 如果发现零点
            I.append(omega_plus[i])  # 将该位置添加到列表 I 中
            k += 1  # 计数器增加
        i += 1  # 索引变量增加
    ICF = np.array(I) * FS  # 将找到的位置转换为频率
    return ICF  # 返回瞬时中心频率的位置


# 信号分解
def imf(signal, FS, alpha, tol, ICF):
    K = len(ICF)  # 模态的数量
    T = len(signal)  # 信号长度

    # 信号镜像扩展
    f_mirror = np.concatenate([signal[T // 2:0:-1], signal, signal[T - 1:T // 2:-1]])
    f = f_mirror

    # 镜像信号的时域
    T = len(f)
    t = np.arange(1, T + 1) / T

    # 频谱域离散化
    freqs = t - 0.5 - 1 / T

    # 计算FFT并进行频率移位
    f_hat = fftshift(fft(f))
    f_hat_plus = f_hat
    f_hat_plus[:T // 2] = 0  # 保留正频率分量

    # 初始化模态频谱和中心频率
    N = 500
    u_hat_plus = np.zeros((N, T, K), dtype=complex)
    omega_plus = np.zeros((T, K))
    for k in range(K):
        omega_plus[:, k] = ICF[k] / FS

    # 其他初始化
    uDiff = tol + 2.2204e-16  # 更新步长初始化
    n = 0  # 循环计数器初始化

    # 主循环进行迭代更新
    while uDiff > tol and n < N:  # 没有收敛并且迭代次数未达到上限
        # 更新第一个模态累加器
        for k in range(1, K + 1):
            # 通过 Wiener 滤波器更新第一个模态的频谱
            u_hat_plus[n + 1, :, k - 1] = f_hat_plus / (1 + 2 * alpha * (freqs - omega_plus[n, k - 1]) ** 2)

        # 更新循环计数器
        n += 1

        # 检查是否收敛
        uDiff = 2.2204e-16  # 重置差值
        for i in range(1, K + 1):
            uDiff += (1 / T) * np.dot((u_hat_plus[n, :, i - 1] - u_hat_plus[n - 1, :, i - 1]),
                                      np.conjugate(u_hat_plus[n, :, i - 1] - u_hat_plus[n - 1, :, i - 1]))
        uDiff = abs(uDiff)  # 计算差值的绝对值

    N = min(N, n)
    omega = omega_plus[0:N, :]

    u_hat = np.zeros((T, K), dtype=complex)
    # 只取频谱的正频率部分用于重建
    u_hat[T // 2:T, :] = np.squeeze(u_hat_plus[N - 1, T // 2:T, :]).astype(complex)  # .reshape(-1, 1)
    # 取共轭使频谱对称，用于负频率部分的重建
    u_hat[T // 2 + 1:0:-1, :] = np.squeeze(np.conj(u_hat_plus[N - 1, T // 2:T, :]))  # .astype(complex).reshape(-1, 1)

    # 初始化模态数组
    u = np.zeros((K, len(t)))

    # 对每个模态进行逆傅里叶变换以重建时域信号
    for k in range(0, K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # u1 = u[:, 0:2 * T // 5] + u[:, 3 * T // 5 - 1:T - 1]

    try:
        u1 = u[:, 0:2 * T // 5 + 1] + u[:, 3 * T // 5 - 1:T - 1]
    except:
        u1 = u[:, 0:2 * T // 5] + u[:, 3 * T // 5 - 1:T - 1]
    else:
        u1 = u[:, 0:2 * T // 5 + 1] + u[:, 3 * T // 5 - 1:T - 1]
    return u1


# 查找最佳中心频率
def imf_rms_time(data1, data2, icf_location, imf, FS, alpha, tol):
    # 计算第一个数据集的IMF RMS值
    left1_icf = icf_location(data1['Left'], FS, alpha)
    print(f'左轨1最佳中心频率{left1_icf}')
    K1_left = len(left1_icf)
    imfs1_left = imf(data1['Left'], FS, alpha, tol, left1_icf)
    # 计算每个IMF分量的频域内的均方根（RMS）值
    rms1_values = [
        np.sqrt(np.mean(np.abs(imf) ** 2))  # 计算RMS值
        for imf in imfs1_left  # 对imfs1_left列表中的每个IMF分量执行以下操作
    ]

    print('左轨1均方根值')
    print(rms1_values)

    # 计算第二个数据集的IMF RMS值
    left2_icf = icf_location(data2['Left'], FS, alpha)
    print(f'左轨2最佳中心频率{left2_icf}')
    K2_left = len(left2_icf)
    imfs2_left = imf(data2['Left'], FS, alpha, tol, left2_icf)
    rms2_values = [np.sqrt(np.mean(np.abs(imf) ** 2)) for imf in imfs2_left]
    print('左轨2均方根值')
    print(rms2_values)

    # 确保两个数据集的IMF数量相同
    K_min = min(K1_left, K2_left)

    # 绘制柱状图进行比较
    plt.figure()  # 创建一个新的绘图窗口
    x = np.arange(K_min)  # 根据IMF的数量创建一个序列，用作柱状图的x轴坐标
    bar_width = 0.3  # 设置柱状图的宽度

    # 为第一个数据集绘制柱状图
    plt.bar(x, rms1_values[:K_min], width=bar_width, label='正常数据', color='blue')  # 绘制正常数据的RMS值的柱状图，颜色设置为蓝色

    # 为第二个数据集绘制柱状图
    plt.bar(x + bar_width, rms2_values[:K_min], width=bar_width, label='波磨数据',
            color='green')  # 绘制波磨数据的RMS值的柱状图，颜色设置为绿色

    # 计算两个数据集的RMS值之间的差异，并绘制差值的柱状图
    diff_values = [rms2 - rms1 for rms1, rms2 in zip(rms1_values, rms2_values)]  # 计算正常数据和波磨数据的RMS值之差
    print('差值')  # 打印“差值”这个词
    print(diff_values)  # 输出计算出的差值
    plt.bar(x + 2 * bar_width, diff_values[:K_min], width=bar_width, label='差值', color='red')  # 绘制差值的柱状图，颜色设置为红色

    # 设置图表的标签和标题
    plt.xlabel('IMF编号')  # 设置x轴标签为“IMF编号”
    plt.ylabel('RMS值')  # 设置y轴标签为“RMS值”
    plt.title('正常数据与波磨数据的IMF RMS比较')  # 设置图表标题为“正常数据与波磨数据的IMF RMS比较”
    plt.xticks(x + bar_width, range(1, K_min + 1))  # 设置x轴的刻度标签
    plt.legend()  # 显示图例


data_path = r'E:\桌面文件存放\Dataset\波磨区段\波磨数据1.xlsx'

data = pd.read_excel(data_path)
FS = 2000  # 采样频率
alpha = 4000  # 平衡参数
# tau = 0.0  # 对偶上升的时间步长（选择0表示容忍噪声）
# DC = 1  # 如果第一个模态保持在直流（0频）则为真
# init = 0  #    0 = 所有的omega从0开始
#                1 = 所有的omega均匀分布
#                2 = 所有的omega随机初始化
tol = 1e-7  # 收敛标准的容忍度；通常在1e-6左右

# 查找最佳中心频率


# 分别计算 "Left" 和 "Right" 列的 ICF
left_icf = icf_location(data['Left'], FS, alpha)
right_icf = icf_location(data['Right'], FS, alpha)
print(f'左轨最佳中心频率{left_icf}')
print(f'右轨最佳中心频率{right_icf}')

# 设置VMD算法的参数

K_left = len(left_icf)  # 使用最后一组模态的数量作为 K 的值
K_right = len(right_icf)

# 获取左轨信号
left_signal = data['Left']
# 获取右轨信号
right_signal = data['Right']

# 绘制原始信号时域图
# 左轨
f_left = left_signal.values
t = np.arange(0, len(data) / FS, 1 / FS)  # 使用NumPy库的arange函数创建一个时间轴 t，其中参数为起始时间0，终止时间为len(data) / FS，步长为1 / FS
plt.figure(figsize=(8, 5))
plt.plot(t, f_left)  # 使用Matplotlib的plot函数绘制图形，其中t是时间轴，f_left是左轨信号的幅值。它实际上是在时域上绘制了左轨原始信号的波形图
plt.title(f'左轨原始信号时域图')
plt.xlabel('时间 (s)')
plt.ylabel('幅值（$m/s^2$）')

# 计算左轨FFT
FFT = np.fft.fft(left_signal)
freqs = np.fft.fftfreq(len(left_signal), 1 / FS)

# 只取正频率部分
mask = freqs > 0  # 创建一个布尔数组，数组的每个元素都是判断对应频率是否大于0的布尔值。这个布尔数组被称为掩码
FFT = FFT[mask]  # 通过应用掩码，只选择频率大于0的部分，将数组进行切片操作，保留正频率部分
freqs = freqs[mask]

# 绘制左轨原始信号频域图
plt.figure(figsize=(10, 5))
plt.plot(freqs, np.abs(FFT) / (len(left_signal) / 2))  # 将幅值除以信号长度的一半
plt.title(f'左轨原始信号频谱图')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.grid(True)
plt.ylim(0, 3)  # 设置纵坐标范围为0到3

# 右轨
f_right = right_signal.values
t = np.arange(0, len(data) / FS, 1 / FS)  # 使用NumPy库的arange函数创建一个时间轴 t，其中参数为起始时间0，终止时间为len(data) / FS，步长为1 / FS
plt.figure(figsize=(8, 5))
plt.plot(t, f_right)  # 使用Matplotlib的plot函数绘制图形，其中t是时间轴，f_left是左轨信号的幅值。它实际上是在时域上绘制了左轨原始信号的波形图
plt.title(f'右轨原始信号时域图')
plt.xlabel('时间 (s)')
plt.ylabel('幅值（$m/s^2$）')

# 计算右轨FFT
FFT = np.fft.fft(right_signal)
freqs = np.fft.fftfreq(len(right_signal), 1 / FS)

# 只取正频率部分
mask = freqs > 0  # 创建一个布尔数组，数组的每个元素都是判断对应频率是否大于0的布尔值。这个布尔数组被称为掩码
FFT = FFT[mask]  # 通过应用掩码，只选择频率大于0的部分，将数组进行切片操作，保留正频率部分
freqs = freqs[mask]

# 绘制右轨原始信号频域图
plt.figure(figsize=(10, 5))
plt.plot(freqs, np.abs(FFT) / (len(right_signal) / 2))  # 将幅值除以信号长度的一半
plt.title(f'右轨原始信号频谱图')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.grid(True)
plt.ylim(0, 3)  # 设置纵坐标范围为0到3

ICF = left_icf  # 使用整个左轨ICF数组
imfs_left = imf(data['Left'], FS, alpha, tol, ICF)

# 绘制左轨IMF时域图
plt.figure(figsize=(10, 5))  # 创建一个新的图形窗口

for k in range(K_left):
    num_samples = len(imfs_left[k])

    # 创建一个时间数组，每个样本对应一个时间点
    time_array = np.linspace(0, num_samples / FS, num_samples)

    plt.subplot(K_left, 1, k + 1)
    plt.plot(time_array, imfs_left[k])  # 使用时间数组作为横坐标
    plt.title(f'左轨加速度分解模态 IMF {k + 1} 时域图')
    plt.xlabel('时间（s）')
    plt.ylabel('幅值 A/（$m/s^2$）')
    plt.grid(True)  # 加入网格线

# plt.tight_layout()  #  UserWarning: Tight layout not applied. tight_layout cannot make axes height small enough to
# accommodate all axes decorations.
plt.subplots_adjust(hspace=0.5)

ICF = right_icf  # 使用整个右轨ICF数组
imfs_right = imf(data['Right'], FS, alpha, tol, ICF)

# 绘制右轨IMF时域图
plt.figure(figsize=(12, 8))
for k in range(K_right):
    num_samples = len(imfs_right[k])

    # 创建一个时间数组，每个样本对应一个时间点
    time_array = np.linspace(0, num_samples / FS, num_samples)

    plt.subplot(K_right, 1, k + 1)
    plt.plot(time_array, imfs_right[k])  # 使用时间数组作为横坐标
    plt.title(f'右轨加速度分解模态 IMF {k + 1} 时域图')
    plt.xlabel('时间（s）')
    plt.ylabel('幅值 A/（$m/s^2$）')
    plt.grid(True)  # 加入网格线

plt.tight_layout()

# 绘制左轨IMF频域图
plt.figure(figsize=(12, 8))
for k in range(K_left):
    # 对IMF进行FFT
    FFT = np.fft.fft(imfs_left[k])
    # 计算FFT样本的频率
    freqs = np.fft.fftfreq(len(imfs_left[k]), 1 / FS)

    # 只取正频率部分
    mask = freqs > 0
    FFT = FFT[mask]
    freqs = freqs[mask]

    # 绘制IMF的频域图
    plt.subplot(K_left, 1, k + 1)
    plt.plot(freqs, np.abs(FFT) / (len(left_signal) / 2))  # 将幅值除以数据点数
    plt.title(f'左轨加速度分解模态 IMF {k + 1} 频域图')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅值')
    plt.grid(True)  # 加入网格线
    plt.ylim(0, 3)  # 限制纵坐标为0~3

plt.tight_layout()

# 绘制右轨IMF频域图
plt.figure(figsize=(10, 5))
for k in range(K_right):
    # 对IMF进行FFT
    FFT = np.fft.fft(imfs_right[k])
    # 计算FFT样本的频率
    freqs = np.fft.fftfreq(len(imfs_right[k]), 1 / FS)

    # 只取正频率部分
    mask = freqs > 0
    FFT = FFT[mask]
    freqs = freqs[mask]

    # 绘制IMF的频域图
    plt.subplot(K_right, 1, k + 1)
    plt.plot(freqs, np.abs(FFT) / (len(right_signal) / 2))
    plt.title(f'右轨加速度分解模态 IMF {k + 1} 频域图')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅值')
    plt.grid(True)  # 加入网格线

plt.tight_layout()
plt.show()
