from tqdm import tqdm
import optuna
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import inv
import matplotlib.pyplot as plt
import subprocess
from scipy.optimize import root
from pathlib import Path
import random
import ast
import time
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import matplotlib as mpl

P1  = {'Vb1': 1.779673, 'Vb2': 0.990367, 'R1': 1.227843, 'R2': 16.138529, 'R3': 1e1000}
P2  = {'Vb1': 1.853856, 'Vb2': 0.990641, 'R1': 1.272538, 'R2':  9.183643, 'R3': 1e1000}
P3  = {'Vb1': 1.690397, 'Vb2': 0.990507, 'R1': 1.165004, 'R2': 61.893325, 'R3': 1e1000}
P4  = {'Vb1': 1.798110, 'Vb2': 0.990434, 'R1': 1.239350, 'R2': 13.638891, 'R3': 1e1000}
P5  = {'Vb1': 1.793665, 'Vb2': 0.990394, 'R1': 1.236604, 'R2': 14.173623, 'R3': 1e1000}
P6  = {'Vb1': 1.805422, 'Vb2': 0.990342, 'R1': 1.243829, 'R2': 12.836654, 'R3': 1e1000}
P7  = {'Vb1': 1.790399, 'Vb2': 0.990387, 'R1': 1.234576, 'R2': 14.591774, 'R3': 1e1000}
P8  = {'Vb1': 1.765872, 'Vb2': 0.990329, 'R1': 1.226604, 'R2': 18.073079, 'R3': 1e1000}
P9  = {'Vb1': 1.755415, 'Vb2': 0.990344, 'R1': 1.212144, 'R2': 20.990713, 'R3': 1e1000}
P10 = {'Vb1': 1.823194, 'Vb2': 0.990485, 'R1': 1.254544, 'R2': 11.214192, 'R3': 1e1000}
P11 = {'Vb1': 1.826297, 'Vb2': 0.990397, 'R1': 1.256392, 'R2': 10.970193, 'R3': 1e1000}
P12 = {'Vb1': 1.693070, 'Vb2': 0.990488, 'R1': 1.167150, 'R2': 58.312147, 'R3': 1e1000}
P13 = {'Vb1': 1.696676, 'Vb2': 0.990409, 'R1': 1.170009, 'R2': 53.940908, 'R3': 1e1000}
P14 = {'Vb1': 1.763983, 'Vb2': 0.990349, 'R1': 1.217772, 'R2': 19.011489, 'R3': 1e1000}
P15 = {'Vb1': 1.769534, 'Vb2': 0.990354, 'R1': 1.221368, 'R2': 17.897151, 'R3': 1e1000}
P16 = {'Vb1': 1.656937, 'Vb2': 0.991018, 'R1': 1.136013, 'R2':150.469314, 'R3': 1e1000}
P17 = {'Vb1': 1.684428, 'Vb2': 0.990558, 'R1': 1.160310, 'R2': 71.118951, 'R3': 1e1000}
P18 = {'Vb1': 1.765692, 'Vb2': 0.990352, 'R1': 1.218883, 'R2': 18.655534, 'R3': 1e1000}
P19 = {'Vb1': 1.831511, 'Vb2': 0.990520, 'R1': 1.259483, 'R2': 10.582483, 'R3': 1e1000}
P20 = {'Vb1': 1.733742, 'Vb2': 0.990351, 'R1': 1.197416, 'R2': 27.980265, 'R3': 1e1000}


# 电路初始参数设计（固定）
Is = 1.4e-14  #反向饱和电流
m = 1 # 理想因子，一般取1
VT = 26e-3 # 热电压，一般取26mV


def create_approx_function_new(R1, R2, R3, Vb1, Vb2):
    def Iin_of_x(x):
        I1 = (1 / R2 + 1 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2 * Is * R3)) / R2
        Iin = (I1
               + m * VT / R1 * np.arcsinh(I1 / (2 * Is))
               + I1 * R2 / R1
               - R2 * x / (R1 * R3)
               + Vb1 / R1)
        return Iin

    # 1) 采样并构造 Iin(x) 的“可逆”插值
    x_vals = np.linspace(1e-12, 10, 2000)
    y_vals = Iin_of_x(x_vals)  # y = Iin, x = Vout

    # 排序 + 去重，保证 y 作为自变量单调用于插值
    idx = np.argsort(y_vals)
    y_sorted = y_vals[idx]
    x_sorted = x_vals[idx]
    dy = np.diff(y_sorted)
    keep = np.concatenate([[0], np.where(dy != 0)[0] + 1])
    y_unique = y_sorted[keep]
    x_unique = x_sorted[keep]

    # 线性插值器：给定 Iin -> 返回 Vout(正半轴)
    itp = interp1d(y_unique, x_unique, kind='linear', bounds_error=False,
                   fill_value=(x_unique[0], x_unique[-1]))

    def clamp(v, lo, hi):
        return max(min(v, hi), lo)

    # 2) 求 Iin = 0 时的 V0（注意 0 可能不在采样范围内，故需 clamp）
    y0 = clamp(0.0, y_unique[0], y_unique[-1])
    V0 = float(itp(y0))  # 在你的奇函数扩展中，这就是正半轴对应的零流点电压

    # 3) 构造带“定向去偏”的奇函数
    def custom_function_centered(y):
        if y >= 0:
            y_c = clamp(y, y_unique[0], y_unique[-1])
            V = float(itp(y_c))  # 正半轴：直接查表
            return V - V0  # 定向减去偏置
        else:
            # 负半轴：用奇对称镜像得到 V≈-itp(|y|)，再按规则 +V0
            y_c = clamp(-y, y_unique[0], y_unique[-1])
            V = -float(itp(y_c))
            return V + V0

    return custom_function_centered



def create_approx_function_fast(R1, R2, R3, Vb1, Vb2):
    # —— 先把 Iin(x) 写成“数组友好”的形式 ——
    def Iin_of_x(x):
        x = np.asarray(x, dtype=np.float64)
        I1 = (1.0 / R2 + 1.0 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2.0 * Is * R3)) / R2
        Iin = (I1
               + m * VT / R1 * np.arcsinh(I1 / (2.0 * Is))
               + I1 * R2 / R1
               - R2 * x / (R1 * R3)
               + Vb1 / R1)
        return Iin

    # 1) 采样并构造 Iin(x) 的“可逆”查表
    x_vals = np.linspace(1e-12, 10.0, 2000, dtype=np.float64)
    y_vals = Iin_of_x(x_vals)  # y = Iin, x = Vout

    # 保证 y 单调（排序 + 去重）
    idx = np.argsort(y_vals)
    y_sorted = y_vals[idx]
    x_sorted = x_vals[idx]
    keep = np.concatenate(([True], np.diff(y_sorted) != 0))
    y_unique = y_sorted[keep]
    x_unique = x_sorted[keep]
    y_min, y_max = float(y_unique[0]), float(y_unique[-1])

    # 2) Iin=0 对应的电压（作为奇对称的偏置 V0）
    V0 = float(np.interp(0.0, y_unique, x_unique, left=x_unique[0], right=x_unique[-1]))

    # 3) 返回“数组版”的阈值函数：输入 y（可为 ndarray），输出 V
    def f_arr(y):
        y = np.asarray(y, dtype=np.float64)
        ya = np.abs(y)
        # 超出采样范围的值做裁剪（等价于你的 clamp + fill_value）
        ya = np.clip(ya, y_min, y_max)
        Vabs = np.interp(ya, y_unique, x_unique)        # 查表（C 实现，支持向量）
        # 奇对称 + 去偏置
        return np.where(y >= 0.0, Vabs - V0, -Vabs + V0).astype(np.float64)

    return f_arr


def g(z, gamma=1.0):
    z = np.asarray(z)
    gamma = gamma * np.ones_like(z)
    h = z + gamma
    out = h.copy()
    mask1 = z <= -1 - gamma
    out[mask1] = z[mask1] + gamma[mask1]
    mask2 = (z > -1 - gamma) & (z <= -1)
    out[mask2] = -1
    mask3 = (z > -1) & (z <= 1)
    out[mask3] = z[mask3]
    mask4 = (z > 1) & (z <= 1 + gamma)
    out[mask4] = 1
    mask5 = z > 1 + gamma
    out[mask5] = z[mask5] - gamma[mask5]
    return out.astype(np.float64)




# get_graph2用来对比函数BER功效
def get_graph2(BER_list1, BER_list2, arrSNR=list(np.arange(0, 30.01, 2.5))):
    plt.figure(figsize=(8, 5))
    try:
        plt.plot(arrSNR, BER_list1, 'o-', label='diode-SOAV')
        plt.plot(arrSNR, BER_list2, 'o-', label='SOAV')
    except:
        plt.plot(arrSNR, BER_list1[0], 'o-', label='diode-SOAV')
        plt.plot(arrSNR, BER_list2[0], 'o-', label='SOAV')
    # set the y-axis scale to log
    plt.yscale('log')
    # other settings
    plt.xticks(np.arange(0, 35, 5))
    plt.ylim(1e-5, 1)
    plt.grid(True, which='major')
    plt.xlabel("SNR per receive antenna (dB)")
    plt.ylabel("BER")
    plt.legend()
    plt.tight_layout()  # prevent the elements block each other
    # plt.savefig(f"/Users/hujiehuan/Desktop/new_project2/1.diecrete_summary/data/discrete_parameters_result/{index}/plot.pdf",  bbox_inches='tight')
    plt.savefig(f"../tmp/1.wR3/plot.pdf", bbox_inches='tight')
    plt.show()
    print(f"图像plot保存完成")
    plt.close()



def get_tse(R1, R2, R3, Vb1, Vb2):
    f = create_approx_function_new(R1, R2, R3, Vb1, Vb2)
    f = np.vectorize(f)  # 包装f使其接受向量
    # 定义 x 范围
    x_vals = np.linspace(-2.1, 2.1, 1000)

    # 计算两个函数的值
    soft_vals = g(x_vals)
    circuit_vals = f(x_vals)  # 示例参数

    # 计算平方差
    squared_error = (soft_vals - circuit_vals) ** 2
    total_squared_error = np.sum(squared_error) * (x_vals[1] - x_vals[0])  # 数值积分近似
    return total_squared_error


def make_channel(m, n):
    # generate the Standard complex Gaussian matrix(标准复高斯矩阵)
    H_comp = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2)
    H_real = H_comp.real
    H_imag = H_comp.imag
    H = np.block([[H_real, -H_imag],
                  [H_imag, H_real]])  # Real equivalent channel matrix(生成实数等效信道矩阵)
    return H_comp, H




def calculate_BER_noise(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, arrSNR=[25], func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1,n=50,m=32):
    # n = 50  # dimension of the unknown vector:x
    # m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data
    nIteration = 1
    matNumError = np.zeros((nIteration, np.size(arrSNR)))  # record the error number of bits
    matNumBit = np.zeros((nIteration, np.size(arrSNR)))  # record the total number of bits

    if func == 1:
        func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)  # 直接就是数组函数
    elif func == 2:
        func = g  # construct the normal soft-thresholding-like functions

    for kind in range(kinds):
        H_comp, H = make_channel(m, n)
        HH = H.T @ H
        ProjMat1_SOAV = inv(np.eye(2 * n) + alpha * gamma * HH)
        ProjMat2_SOAV = alpha * gamma * H.T

        for SNR in arrSNR:
            sigma = np.sqrt(n * 2 / 10 ** (SNR / 10))
            sigma_r = sigma / np.sqrt(2)
            for symbolVectorIndex in tqdm(range(nSymbolVector), desc=f"SNR={SNR} simulating,kinds:{kind + 1}",
                                          leave=False):  # nSymbolVector is the experiment times

                data = np.random.randint(0, 2, 2 * n)
                s = -2 * data + 1  # generate the sparse discrete vector
                # sigma = np.sqrt(n * 2 / 10 ** (SNR / 10)) #优化1，把这个移到外层去，使内层噪声固定
                # sigma_r = sigma / np.sqrt(2)
                v = np.random.randn(2 * m) * sigma_r
                y = H @ s + v  # generate the observation vector y

                r = np.zeros(2 * n)  # initial the r

                for k in range(K):  # k times iteration
                    if noise == -1:
                        z = func(r)  # use the soft-thresholding function
                        r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                    elif noise == False:
                        n3 = np.random.normal(0, std, size=n * 2)  # amplifier noise
                        r += n3
                        z = func(r)  # use the soft-thresholding function
                        r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                    else:
                        # Before soft-thresholding:
                        n1 = np.random.poisson(shot, size=n * 2) * on  # shot noise（the big circuit itself)
                        n2 = np.random.normal(0, thermal1, size=n * 2)  # OE thermal noise(OE)
                        n3 = np.random.normal(0, std, size=n * 2)  # circuit noise
                        r += n1 + n2 + n3

                        # Apply soft-thresholding:
                        z = func(r)
                        n4 = np.random.normal(0, thermal2, size=n * 2)  # EO thermal noise(EO)
                        z += n4
                        r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)

                s_hat = np.where(r >= 0, 1, -1)  # output 1 when r close to 1,otherwise is -1
                # matNumError[0, arrSNR.index(SNR)] += np.sum(s - s_hat != 0)
                matNumError[0, arrSNR.index(SNR)] += np.count_nonzero(s != s_hat)  # 优化2，避免运算，直接判断
                matNumBit[0, arrSNR.index(SNR)] += 2 * n
    matBER = matNumError / matNumBit  # calculate the BER
    print(matBER[0])
    return matBER[0]



def calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, SNR=25, func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1,n=50,m=32):
    # n = 50  # dimension of the unknown vector:x
    # m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data

    ber_trace = np.zeros(K)  # 记录每次迭代的平均BER

    if func == 1:  # construct the diode functions
        # diode2 = create_approx_function_new(R1, R2, R3, Vb1, Vb2)
        # func = np.vectorize(diode2)
        func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)  # 直接就是数组函数
    elif func == 2:  # construct the normal soft-thresholding-like functions
        func = g

    for kind in range(kinds):
        H_comp, H = make_channel(m, n)
        HH = H.T @ H
        ProjMat1_SOAV = inv(np.eye(2 * n) + alpha * gamma * HH)
        ProjMat2_SOAV = alpha * gamma * H.T
        sigma = np.sqrt(n * 2 / 10 ** (SNR / 10))
        sigma_r = sigma / np.sqrt(2)

        for symbolVectorIndex in tqdm(range(nSymbolVector), desc=f"SNR={SNR} simulating,kinds:{kind + 1}",
                                      leave=False):  # nSymbolVector is the experiment times

            data = np.random.randint(0, 2, 2 * n)
            s = -2 * data + 1  # generate the sparse discrete vector
            v = np.random.randn(2 * m) * sigma_r
            y = H @ s + v  # generate the observation vector y
            r = np.zeros(2 * n)  # initial the r

            for k in range(K):  # k times iteration
                if noise == -1:

                    z = func(r)  # use the soft-thresholding function
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                elif noise == False:
                    n3 = np.random.normal(0, std, size=n * 2)  # amplifier noise
                    r += n3
                    z = func(r)  # use the soft-thresholding function
                    # z += n3
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                else:
                    # Before soft-thresholding:
                    n1 = np.random.poisson(shot, size=n * 2) * on  # shot noise（the big circuit itself)
                    n2 = np.random.normal(0, thermal1, size=n * 2)  # OE thermal noise(OE)
                    n3 = np.random.normal(0, std, size=n * 2)  # circuit noise
                    r += n1 + n2 + n3

                    # Apply soft-thresholding:
                    z = func(r)
                    n4 = np.random.normal(0, thermal2, size=n * 2)  # EO thermal noise(EO)
                    z += n4
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)

                s_hat = np.where(r >= 0, 1, -1)  # output 1 when r close to 1,otherwise is -1
                errors = np.count_nonzero(s != s_hat)
                ber_trace[k] += errors / (2 * n)  # 每次迭代累积当前BER（逐个sample平均）

    ber_trace /= (nSymbolVector * kinds)  # 所有样本的平均 BER
    print("BER:", ber_trace)
    return ber_trace



def calculate_MSE_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, SNR=25, func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1,n=50,m=32):
    # n = 50  # dimension of the unknown vector:x
    # m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data

    mse_trace = np.zeros(K)  # 新增：记录每次迭代的平均MSE

    if func == 1:  # construct the diode functions
        func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)  # 直接就是数组函数
    elif func == 2:  # construct the normal soft-thresholding-like functions
        func = g

    for kind in range(kinds):
        H_comp, H = make_channel(m, n)
        HH = H.T @ H
        ProjMat1_SOAV = inv(np.eye(2 * n) + alpha * gamma * HH)
        ProjMat2_SOAV = alpha * gamma * H.T
        sigma = np.sqrt(n * 2 / 10 ** (SNR / 10))
        sigma_r = sigma / np.sqrt(2)

        for symbolVectorIndex in tqdm(range(nSymbolVector), desc=f"SNR={SNR} simulating, kinds:{kind + 1}",
                                      leave=False):
            data = np.random.randint(0, 2, 2 * n)
            s = -2 * data + 1  # generate the sparse discrete vector
            v = np.random.randn(2 * m) * sigma_r
            y = H @ s + v  # generate the observation vector y
            r = np.zeros(2 * n)  # initial the r

            for k in range(K):
                if noise == -1:
                    z = func(r)  # use the soft-thresholding function
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                elif noise == False:
                    n3 = np.random.normal(0, std, size=n * 2)  #

                    # r += n3
                    z = func(r)  # use the soft-thresholding function
                    z += n3
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                else:
                    # Before soft-thresholding:
                    n1 = np.random.poisson(shot, size=n * 2) * on  # shot noise（the big circuit itself)
                    n2 = np.random.normal(0, thermal1, size=n * 2)  # OE thermal noise(OE)
                    n3 = np.random.normal(0, std, size=n * 2)  # circuit noise
                    r += n1 + n2 + n3
                    # r += n1 + n2

                    # Apply soft-thresholding:
                    z = func(r)
                    n4 = np.random.normal(0, thermal2, size=n * 2)  # EO thermal noise(EO)
                    # z += n4
                    z += n4
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)

                # MSE
                mse = np.mean((r - s) ** 2)
                mse_trace[k] += mse

    # ber_trace /= (nSymbolVector * kinds)
    mse_trace /= (nSymbolVector * kinds)

    # print("BER:", ber_trace)
    print("MSE:", mse_trace)
    return mse_trace


def calculate_cost_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=2, SNR=25, func=1, kinds=10, noise=False,
                         std=0, thermal1=0, thermal2=0, shot=0, on=1,n=50,m=32):
    # n = 50  # dimension of the unknown vector:x
    # m = 32  # number of linear measurements:y
    K = 300  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data

    cost_trace = np.zeros(K)  # 新增：记录每次迭代的平均BER

    if func == 1:  # construct the diode functions
        func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)  # 直接就是数组函数
    elif func == 2:  # construct the normal soft-thresholding-like functions
        func = g

    for kind in range(kinds):
        H_comp, H = make_channel(m, n)
        HH = H.T @ H
        ProjMat1_SOAV = inv(np.eye(2 * n) + alpha * gamma * HH)
        ProjMat2_SOAV = alpha * gamma * H.T
        sigma = np.sqrt(n * 2 / 10 ** (SNR / 10))
        sigma_r = sigma / np.sqrt(2)

        for symbolVectorIndex in tqdm(range(nSymbolVector), desc=f"SNR={SNR} simulating,kinds:{kind + 1}",
                                      leave=False):  # nSymbolVector is the experiment times

            data = np.random.randint(0, 2, 2 * n)
            s = -2 * data + 1  # generate the sparse discrete vector
            v = np.random.randn(2 * m) * sigma_r
            y = H @ s + v  # generate the observation vector y
            r = np.zeros(2 * n)  # initial the r

            for k in range(K):  # k times iteration
                s_hat = np.where(r >= 0, 1, -1)
                cost = (
                        np.linalg.norm(s_hat - np.ones(len(s_hat)), 1) / 2 +
                        np.linalg.norm(s_hat + np.ones(len(s_hat)), 1) / 2 +
                        alpha / 2 * np.linalg.norm(y - H @ s_hat) ** 2
                )

                if noise == -1:
                    z = func(r)  # use the soft-thresholding function
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                elif noise == False:
                    n3 = np.random.normal(0, std, size=n * 2)  # amplifier noise
                    r += n3
                    z = func(r)  # use the soft-thresholding function
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
                else:
                    # Before soft-thresholding:
                    n1 = np.random.poisson(shot, size=n * 2) * on  # shot noise（the big circuit itself)
                    n2 = np.random.normal(0, thermal1, size=n * 2)  # OE thermal noise(OE)
                    n3 = np.random.normal(0, std, size=n * 2)  # circuit noise
                    r += n1 + n2 + n3

                    # Apply soft-thresholding:
                    z = func(r)
                    n4 = np.random.normal(0, thermal2, size=n * 2)  # EO thermal noise(EO)
                    z += n4
                    r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)

                cost_trace[k] += cost

    cost_trace /= (nSymbolVector * kinds)
    print(cost_trace)
    return cost_trace



def main_base(nSymbolVector = 2,kinds = 10,n=50,m=32,ber=True,mse=True,cost=True):
    crnstd_G512 = 0.0360
    if ber == True:
        calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                            noise=-1,
                            std=crnstd_G512, thermal1=0, thermal2=0,
                            shot=0,
                            on=1,n=n,m=m
                            )



        calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                            noise=False,
                            std=crnstd_G512, thermal1=0, thermal2=0,
                            shot=0,
                            on=1,n=n,m=m
                            )

    if mse == True:
        calculate_MSE_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                            noise=-1,
                            std=crnstd_G512, thermal1=0, thermal2=0,
                            shot=0,
                            on=1,n=n,m=m
                            )

        calculate_MSE_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                            noise=False,
                            std=crnstd_G512, thermal1=0, thermal2=0,
                            shot=0,
                            on=1,n=n,m=m
                            )
    if cost == True:
        calculate_cost_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                             noise=-1,
                             std=crnstd_G512, thermal1=0, thermal2=0,
                             shot=0,
                             on=1, n=n, m=m
                             )
        calculate_cost_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=2, kinds=kinds,
                            noise=False,
                            std=crnstd_G512, thermal1=0, thermal2=0,
                            shot=0,
                            on=1,n=n,m=m
                            )






def main_B(nSymbolVector = 2,kinds = 10,n=50,m=32,ber=True,mse=True,cost=True):


    crnstd_G512 = 0.0360

    kB = 1.380e-23
    T = 300

    # parameters_noise
    Rp = 1 / (1 / R1 + 1 / R2)  # Parallel connection（并联) of R1 and R2 and R3
    q = 1.602e-19  # Electron charge
    Re = 1  # Current detection resistor (assuming 1Ω)
    Popt = 1e-3  # Popt = 1e-3    # Input optical power（输入光功率） (watts)
    C_drs = 1 / (2 * np.pi * Rp * 10e9)  # Equivalent capacitance of the ST circuitST(电路的等效电容：assume 10GHz)
    Be_drs = 1 / (2 * np.pi * C_drs * Rp)  # Effective noise bandwidth(模型的有效噪声带宽)
    var_shot_st = 2 * q * Re * Popt * Be_drs  # 2.Shot Noise Variance(散粒噪声方差)
    var_thermal_OE_drs = 4 * kB * T * Be_drs / Rp  # 3.hot Noise Variance(热噪声方差)
    std_thermal_OE_drs = np.sqrt(var_thermal_OE_drs)
    var_thermal_EO_drs = 4 * kB * T * Be_drs * R2
    std_thermal_EO_drs = np.sqrt(var_thermal_EO_drs)


    if ber == True:
        calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                            noise=-1,
                            std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                            shot=var_shot_st,
                            on=1,n=n,m=m
                            )

        calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                            noise=True,
                            std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                            shot=var_shot_st,
                            on=1,n=n,m=m
                            )

    if mse == True:
        calculate_MSE_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                            noise=-1,
                            std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                            shot=var_shot_st,
                            on=1,n=n,m=m
                            )


        calculate_MSE_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                            noise=True,
                            std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                            shot=var_shot_st,
                            on=1,n=n,m=m
                            )

    if cost == True:
        calculate_cost_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                             noise=-1,
                             std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                             shot=var_shot_st,
                             on=1, n=n, m=m
                             )

        calculate_cost_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=nSymbolVector, SNR=25, func=1, kinds=kinds,
                             noise=True,
                             std=crnstd_G512, thermal1=std_thermal_OE_drs, thermal2=std_thermal_EO_drs,
                             shot=var_shot_st,
                             on=1, n=n, m=m
                             )


# Vb1 = 1.7220887223739179;Vb2 = 1.0098194170747206;R1 = 1.1007384569324639;R2 = 127.66333729643445
# R3 = 1e2000




i = 0
time1 = time.time()
# for P in [P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20]:
# for P in [P12]:
#     Vb1 = P['Vb1']
#     Vb2 = P['Vb2']
#     R1 = P['R1']
#     R2 = P['R2']
#     R3 = P['R3']
#     print(f"第{i}个参数")
#     i += 1
#     main_B(nSymbolVector = 2,kinds = 500)


# Vb1 = 1.749451;
# Vb2 = 1.009939;
# R1 = 1.109403;
# R2 = 63.289371
#
# R3 = 1e2000

# Vb1 = 1.796193;Vb2 = 1.009972;R1 = 1.133484 ;R2 = 27.054355
# R3 = 1e2000

kinds = 20000
kinds = 20000
R1 = 1.16;R2 = 58.31;Vb1=1.69;Vb2=0.99;R3 =1e2000
main_base(nSymbolVector = 2,kinds = kinds,n=48,m=32,mse=False,cost=False)
main_B(nSymbolVector = 2,kinds = kinds,n=48,m=32,mse=False,cost=False)

main_base(nSymbolVector = 2,kinds = kinds,n=24,m=16,mse=False,cost=False)
main_B(nSymbolVector = 2,kinds = kinds,n=24,m=16,mse=False,cost=False)

main_base(nSymbolVector = 2,kinds = kinds,n=12,m=8,mse=False,cost=False)
main_B(nSymbolVector = 2,kinds = kinds,n=12,m=8,mse=False,cost=False)

main_base(nSymbolVector = 2,kinds = kinds,n=6,m=4,mse=False,cost=False)
main_B(nSymbolVector = 2,kinds = kinds,n=6,m=4,mse=False,cost=False)

time2 = time.time()
print(f"花了{(time2-time1)/60}分钟")


