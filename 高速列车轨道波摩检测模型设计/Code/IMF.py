import matplotlib
import numpy as np
from scipy.fft import fft, fftshift
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
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
    # 切法2，杠铃
    # u1 = u[:, T // 4:3 * T // 4]
    # uk = [u1[i, :] for i in range(K)]
    # return uk
