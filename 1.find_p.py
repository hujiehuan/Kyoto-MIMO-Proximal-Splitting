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


def make_channel(m, n):
    # generate the Standard complex Gaussian matrix(标准复高斯矩阵)
    H_comp = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2)
    H_real = H_comp.real
    H_imag = H_comp.imag
    H = np.block([[H_real, -H_imag],
                  [H_imag, H_real]])  # Real equivalent channel matrix(生成实数等效信道矩阵)
    return H_comp, H


def calculate_BER_noise(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, arrSNR=[25], func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1):
    n = 50  # dimension of the unknown vector:x
    m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data
    nIteration = 1
    matNumError = np.zeros((nIteration, np.size(arrSNR)))  # record the error number of bits
    matNumBit = np.zeros((nIteration, np.size(arrSNR)))  # record the total number of bits

    if func == 1:
        diode2 = create_approx_function_new(R1, R2, R3, Vb1, Vb2)  # construct the diode functions
        func = np.vectorize(diode2)
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





def calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, SNR=25, func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1):
    n = 50  # dimension of the unknown vector:x
    m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data

    ber_trace = np.zeros(K)  # 记录每次迭代的平均BER

    if func == 1:  # construct the diode functions
        diode2 = create_approx_function_new(R1, R2, R3, Vb1, Vb2)
        func = np.vectorize(diode2)
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


def solve_equations(Vb1):
    R3 = 1e2000
    def equations(R):
        Vb2,R1, R2 = R
        # 方程1，其中x为纵轴，y为横轴
        x = 0
        y = 0
        I1 = (1 / R2 + 1 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2 * Is * R3)) / (R2)
        Iin = I1 + m * VT / (R1) * np.arcsinh(I1 / (2 * Is)) + I1 * R2 / R1 - R2 * x / (R1 * R3) + Vb1 / R1
        F1 = Iin - y
        # 方程2
        x = 0.99
        y = 1
        I1 = (1 / R2 + 1 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2 * Is * R3)) / (R2)
        Iin = I1 + m * VT / (R1) * np.arcsinh(I1 / (2 * Is)) + I1 * R2 / R1 - R2 * x / (R1 * R3) + Vb1 / R1
        F2 = Iin - y
        # 方程3
        x = 1.01
        y = 2
        I1 = (1 / R2 + 1 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2 * Is * R3)) / (R2)
        Iin = I1 + m * VT / (R1) * np.arcsinh(I1 / (2 * Is)) + I1 * R2 / R1 - R2 * x / (R1 * R3) + Vb1 / R1
        F3 = Iin - y

        return [F1, F2,F3]

    initial_guesses = [
        [0.1, 0.1, 0.1],
        [10, 1, 100],
        [5, 5, 500],
        [10, 10, 1000],
        [10, 2, 500],
        [5, 4, 1000],
        [10, 5, 100],
        [5, 3, 400],
        [10, 5, 2000],
        [5, 50, 100],
        [10, 100, 10],
        [10, 1000000, 1000000],  # 极端点，有时能触发意外解
    ]

    # 解方程组
    for x0 in initial_guesses:
        sol = root(equations, x0)
        # 输出结果
        if sol.success and np.max(np.abs(sol.fun)) < 0.1:
            Vb2_sol,R1_sol, R2_sol = sol.x
            if R1_sol > 0 and R2_sol > 0 and abs(Vb2_sol)<15:
                print(sol.fun)
                print(f"Vb1 = {Vb1},Solved:Vb2 = {Vb2_sol:.4f}, R1 = {R1_sol:.4f},R2 = {R2_sol:.4f}")
                break
        else:
            Vb2_sol=100
            R1_sol = -1
            R2_sol = -1
    return Vb2_sol,R1_sol, R2_sol



def start_solve():
    try:
        df = pd.read_csv("../data/p_raw_woR3.csv")
        print("read the data")
    except:
        print("create the csv file")
        df = pd.DataFrame(columns=["Vb1", "Vb2", "R1", "R2"])

    rng = np.random.default_rng()
    # 保存成功解的结果
    results = 0
    while True:
        Vb1 = rng.uniform(-10, 10)
        # Vb2 = rng.uniform(-10, 10)
        Vb2,R1, R2 = solve_equations(Vb1)
        # print("fail")
        if R1 > 0 and R2 > 0 and abs(Vb2)<15:
            results += 1
            print(f"Vb1 = {Vb1};Vb2 = {Vb2};R1 = {R1};R2 = {R2}")
            # if len(results) % 10 == 0:
            #     # 每次成功都保存到 CSV（可选：每10次保存1次）
            df.loc[len(df)] = [Vb1, Vb2, R1, R2]
            df.to_csv("../data/p_raw_woR3.csv", index=False)
        if results == 200:
            print(f"找到了{results}次")
            break


start_solve()