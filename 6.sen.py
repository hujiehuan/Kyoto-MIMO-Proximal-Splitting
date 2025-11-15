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

def calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=200, SNR=25, func=1, kinds=10, noise=False,
                        std=0, thermal1=0, thermal2=0, shot=0, on=1):
    n = 50  # dimension of the unknown vector:x
    m = 32  # number of linear measurements:y
    K = 200  # diode-SOAV iteration times
    gamma = 1  # fixed to 1,control the function of thresholding-like function
    alpha = 1  # when alpha is large, means more emphasis on fitting with data

    ber_trace = np.zeros(K)  # 记录每次迭代的平均BER

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
    # print("BER:", ber_trace)
    return ber_trace


# Vb1 = 1.7220887223739179;Vb2 = 1.0098194170747206;R1 = 1.1007384569324639;R2 = 127.66333729643445
# R3 = 1e2000
#
#
# time1 = time.time()
# tse = get_tse(R1, R2, R3, Vb1, Vb2)
# print(tse)
# calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector=2, SNR=25, func=1, kinds=50, noise=False,
#                         std=0, thermal1=0, thermal2=0, shot=0, on=1)
# time2 = time.time()
#
# print(f"花了{(time2-time1)/60}分钟")


# ---- 你的 calculate_BER_trace 的公共参数放这里，改一次全局生效 ----
CALC_KW = dict(nSymbolVector=2, SNR=25, func=1, kinds=5000,
               noise=False, std=0, thermal1=0, thermal2=0, shot=0, on=1)
R3_CONST = 1e2000  # 你现在用的固定 R3

def eval_ber_for_row(row, overrides=None):
    """对一行参数（可带覆盖）计算 BER（取轨迹最后一个点）"""
    p = {
        "R1": float(row["R1"]),
        "R2": float(row["R2"]),
        "Vb1": float(row["Vb1"]),
        "Vb2": float(row["Vb2"]),
    }
    if overrides:
        p.update(overrides)

    ber_trace = calculate_BER_trace(
        p["R1"], p["Vb1"], p["Vb2"], p["R2"], R3_CONST, **CALC_KW # **表示拆开字典，当作关键字来使用
    )
    # 兼容 list/ndarray
    ber = float(np.array(ber_trace).ravel()[-1])
    return ber

def sensitivity_for_row(row, deltas=(0.01,), targets=("R1","R2","Vb1","Vb2")):
    """
    对单行做敏感性：对每个 target 参数做 ±delta 扰动（可多个 delta），
    返回一个 Series，包含：
      - BER_base
      - BER_<param>_+1% / _-1%（每个 delta 都会生成列）
      - sens_dBER_d<param>（数值导数）
      - elasticity_<param>（相对敏感度：ΔBER/BER / Δparam/param）
    """
    out = {}
    base = eval_ber_for_row(row)
    out["BER_base"] = base

    for param in targets:
        v = float(row[param])
        for d in deltas:
            # +delta
            ber_p = eval_ber_for_row(row, {param: v * (1 + d)})
            out[f"BER_{param}_+{int(d*100)}%"] = ber_p
            # -delta
            ber_m = eval_ber_for_row(row, {param: v * (1 - d)})
            out[f"BER_{param}_-{int(d*100)}%"] = ber_m

            # # 数值导数与弹性（用最小一个 delta 的中心差分代表该参数）
            # # 只计算一次（用最小 d 结果），避免覆盖
            # tag_der = f"sens_dBER_d{param}"
            # tag_ela = f"elasticity_{param}"
            # if tag_der not in out:
            #     # 中心差分：dBER/dx ≈ (f(x+h)-f(x-h)) / (2*h)；这里 h = d * v
            #     if v != 0:
            #         out[tag_der] = (ber_p - ber_m) / (2 * d * v)
            #     else:
            #         out[tag_der] = np.nan
            #     # 弹性： (ΔBER/BER) / (Δx/x)  ≈ (ber_p-ber_m)/(2*d*base)
            #     out[tag_ela] = np.nan if base == 0 else (ber_p - ber_m) / (2 * d * base)

    return pd.Series(out)

# ====== 用法 1：对单行（如第 i 行）做敏感性 ======

def str_to_list(s):
    # 去掉方括号
    s = s.strip("[]")
    # 按空格切分并转成 float
    return [float(x) for x in s.split() if x.strip()]





# def main():
#     data = pd.read_csv(f"../tmp/parameters/evaluate_woR3.csv")
#     data["BER_list"] = data["BER_list"].apply(str_to_list)
#     data["BER"] = data["BER_list"].apply(lambda x: x[-1])
#     # data = data.sort_values("BER",ascending=True)
#
#     try:
#         df = pd.read_csv(f"../tmp/parameters/sen_woR3.csv")
#         print("read the data")
#     except:
#         print("create the csv file")
#         df = pd.DataFrame(columns=["Vb1", "Vb2", "R1", "R2", "BER", "tse","BER_list","BER_sensitivity"])
#
#     print(f"目前是{len(df)}/{len(data)}")
#
#     for line in range(len(df), len(data)):
#
#         time1 = time.time()
#         column = data.iloc[line]
#         sens_i = sensitivity_for_row(column, deltas=tuple(round(i / 100, 2) for i in range(1, 100,2)),
#                                      targets=("R1","R2","Vb1","Vb2"))
#         column["BER_sensitivity"] = sens_i.to_dict()
#         df.loc[line] = column
#         df.to_csv(f"../tmp/parameters/sen_woR3.csv", index=False)
#         time2 = time.time()
#         print(f"第{line}行保存成功,BER_list为{sens_i}，花了{(time2-time1)/60}分钟")


# if __name__ == "__main__":
#     main()



# 顶部（在 import numpy 之前）避免库内部多线程与进程并行打架
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import json
# 你的 calculate_BER_trace / eval_ber_for_row / sensitivity_for_row 保持不变



def process_one_row(i, row):
    sens_i = sensitivity_for_row(row, deltas=DELTA_TUPLE, targets=TARGETS)
    # 返回修改后的行（把 dict/JSON 都塞进去，CSV更友好）
    row_out = row.copy()
    row_out["BER_sensitivity"] = sens_i.to_dict()
    row_out["BER_sensitivity_json"] = json.dumps(row_out["BER_sensitivity"], ensure_ascii=False)

    # 在每个任务完成时打印一次信息
    print(f"[done] finished row {i}, BER_base={sens_i.get('BER_base', None)}")

    return i, row_out

def main(lines):
    data = pd.read_csv("../tmp/parameters/evaluate_woR3.csv")
    data["BER_list"] = data["BER_list"].apply(str_to_list)
    data["BER"] = data["BER_list"].apply(lambda x: x[-1])

    try:
        df = pd.read_csv("../tmp/parameters/sen_woR3.csv")
        start = len(df)
    except:
        df = pd.DataFrame(columns=["Vb1","Vb2","R1","R2","BER","tse","BER_list","BER_sensitivity","BER_sensitivity_json"])
        start = 0

    print(f"进行到{start}/{len(data)}")

    idxs = range(start, lines)

    # ← 在这里插入
    print(f"[parallel] backend=loky, tasks={len(idxs)}, n_jobs={N_WORKERS}, "
          f"os_cpu={os.cpu_count()}, OMP={os.getenv('OMP_NUM_THREADS')}, "
          f"MKL={os.getenv('MKL_NUM_THREADS')}, OPENBLAS={os.getenv('OPENBLAS_NUM_THREADS')}")

    # 并行跑所有“剩余”行
    results = Parallel(n_jobs=N_WORKERS, backend="loky")(
        delayed(process_one_row)(i, data.iloc[i]) for i in idxs
    )

    # 收集写回
    for i, row_out in results:
        df.loc[i] = row_out

    df.to_csv("../tmp/parameters/sen_woR3.csv", index=False)
    print(f"完成 {len(results)} 行；当前 df 尺寸：{df.shape}")


# ---- 你的 calculate_BER_trace 的公共参数放这里，改一次全局生效 ----
CALC_KW = dict(nSymbolVector=2, SNR=25, func=1, kinds=10000,
               noise=False, std=0, thermal1=0, thermal2=0, shot=0, on=1)
R3_CONST = 1e2000  # 你现在用的固定 R3

DELTA_TUPLE = tuple(round(i/100, 2) for i in range(1, 100))
TARGETS = ("R1","R2","Vb1","Vb2")
N_WORKERS = -1  # 用满机器；或手动给核数

if __name__ == "__main__":
    time1 = time.time()
    main(10)
    time2 = time.time()
    print(f"花了{(time2-time1)/60}分钟")

    time1 = time.time()
    main(20)
    time2 = time.time()
    print(f"花了{(time2 - time1) / 60}分钟")

    # time1 = time.time()
    # main(30)
    # time2 = time.time()
    # print(f"花了{(time2 - time1) / 60}分钟")
    #
    # time1 = time.time()
    # main(40)
    # time2 = time.time()
    # print(f"花了{(time2 - time1) / 60}分钟")





