# -*- coding: utf-8 -*-
"""
创建于 2019年2月20日 19:24:58
作者：Vinícius Rezende Carvalho
"""
import numpy as np
def VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    变分模态分解
    由 Vinícius Rezende Carvalho 用Python实现 - vrcarva@gmail.com
    代码基于Dominique Zosso的MATLAB代码，可在以下网址找到：
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    原始论文：
    Dragomiretskiy, K. 和 Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    输入和参数：
    ---------------------
    f       - 要分解的时域信号（1维）
    alpha   - 数据保真度约束的平衡参数
    tau     - 对偶上升的时间步长（选择0表示容忍噪声）
    K       - 要恢复的模态数量
    DC      - 如果第一个模态是并保持在直流（0频）则为真
    init    - 0 = 所有的omega从0开始
               1 = 所有的omega均匀分布
               2 = 所有的omega随机初始化
    tol     - 收敛标准的容忍度；通常在1e-6左右
    输出：
    -------
    u       - 分解得到的模态集合
    u_hat   - 模态的光谱
    omega   - 估计的模式中心频率
    """
    if len(f) % 2:
        f = f[:-1]
    # 输入信号的周期和采样频率
    fs = 1. / len(f)
    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))
    # 镜像信号的时域从0到T
    T = len(fMirr)
    t = np.arange(1, T + 1) / T  # np.arrange()用于生成在给定的开始值和结束值之间均匀分布的数值
    # 频谱域离散化
    freqs = t - 0.5 - (1 / T)
    # 最大迭代次数（如果还没有收敛，那么它也不会继续迭代）
    Niter = 400
    # 未来的泛化：每个模态的个体alpha
    Alpha = alpha * np.ones(K)
    # 构造并居中 f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)  # 复制 f_hat
    f_hat_plus[:T // 2] = 0
    # omega_k的初始化
    omega_plus = np.zeros([Niter, K])
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0
    # 如果DC模式被强加，将其omega设为0
    if DC:
        omega_plus[0, 0] = 0
    # 从空的对偶变量开始
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)
    # 其他初始化
    uDiff = tol + np.spacing(1)  # 更新步长
    n = 0  # 循环计数器
    sum_uk = 0  # 累加器
    # 跟踪每个迭代的矩阵 // 可以为了内存而丢弃
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)
    # ***用于迭代更新的主循环***
    while (uDiff > tol and n < Niter - 1):  # 还没有收敛并且在迭代限制以下
        # 更新第一个模态的累加器
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        # 通过残差的Wiener滤波器更新第一个模态的频谱
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                1. + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
        # 如果不保持在0，则更新第一个omega
        if not (DC):
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
        # 更新任何其他模式
        for k in np.arange(1, K):
            # 累加器
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # 模式频谱
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            # 中心频率
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T //
                2:T, k]) ** 2)
        # 对偶上升
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)
        # 循环计数器
        n = n + 1
        # 已经收敛了吗？
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))
        uDiff = np.abs(uDiff)
        # 后处理和清理
    # 如果提前收敛，丢弃空间
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]
    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    # 信号重建
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[Niter - 1, T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    # 移除镜像部分
    u = u[:, T // 4:3 * T // 4]
    # 重新计算频谱
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))
    return u, u_hat, omega
