import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

# 导入我们“重构”的 src 模块
from .algorithms import create_approx_function_fast, g

def make_channel(m, n):
    """
    (这是从 3.BER_list.py [cite: 3.BER_list.py] 里“重构”的)
    生成实数等效信道矩阵 (MIMO Channel)
    """
    H_comp = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2)
    H_real = H_comp.real
    H_imag = H_comp.imag
    H = np.block([[H_real, -H_imag],
                  [H_imag, H_real]])
    return H_comp, H

def _run_drs_simulation_core(args):
    """
    (这是一个“内部”辅助函数，用于“多进程”并行计算)
    """
    # --- 1. 解包所有参数 ---
    (R1, Vb1, Vb2, R2, R3, 
     nSymbolVector, SNR, func_type, K_iterations,
     n, m, kind_idx) = args

    # --- 2. 选择“王牌”函数 (Diode) 还是“对照组” (Standard g) ---
    if func_type == 'diode':
        prox_func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)
    elif func_type == 'standard_g':
        prox_func = g
    else:
        raise ValueError(f"未知的 func_type: {func_type}")

    # --- 3. (这是你 3.BER_list.py [cite: 3.BER_list.py] 里的“核心” DRS 循环) ---
    gamma = 1.0
    alpha = 1.0
    
    # (为这个 kind 创建信道)
    H_comp, H = make_channel(m, n)
    HH = H.T @ H
    ProjMat1_SOAV = inv(np.eye(2 * n) + alpha * gamma * HH)
    ProjMat2_SOAV = alpha * gamma * H.T
    
    sigma = np.sqrt(n * 2 / 10 ** (SNR / 10))
    sigma_r = sigma / np.sqrt(2)

    total_errors = 0
    total_bits = 0
    ber_trace_for_this_kind = np.zeros(K_iterations)

    # (内循环：跑 nSymbolVector 次)
    for _ in range(nSymbolVector):
        data = np.random.randint(0, 2, 2 * n)
        s = -2 * data + 1  # (Ground Truth 信号)
        v = np.random.randn(2 * m) * sigma_r
        y = H @ s + v      # (观测信号)
        
        r = np.zeros(2 * n) # (DRS 算法的“r”向量 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 4, Alg. 2])

        # (DRS 迭代 K_iterations 次)
        for k in range(K_iterations):
            # (这是 DRS 算法的核心 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 4, Alg. 2])
            z = prox_func(r) 
            r = r + gamma * (ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y) - z)
            
            # (在第 k 次迭代时，计算 BER [cite: 5.evaluate2.py])
            s_hat = np.where(r >= 0, 1, -1)
            errors = np.count_nonzero(s != s_hat)
            ber_trace_for_this_kind[k] += (errors / (2 * n)) # (累加 BER)

    # (计算“最终”的 BER)
    s_hat_final = np.where(r >= 0, 1, -1)
    total_errors = np.count_nonzero(s != s_hat_final)
    total_bits = 2 * n * nSymbolVector
    
    # (取 K_iterations 次迭代的“平均” BER trace)
    ber_trace_avg = ber_trace_for_this_kind / nSymbolVector 

    return total_errors, total_bits, ber_trace_avg, SNR


def _run_simulation_parallel(R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m):
    """
    (这是一个“多进程”管理器，用于加速模拟)
    """
    
    # (使用你 `6.sen.py` [cite: 6.sen.py] 里的“多进程”专业技巧)
    num_workers = os.cpu_count()
    print(f"  > 正在启动 {num_workers} 个“并行” worker...")
    
    # (为“kinds”次模拟创建任务列表)
    tasks = []
    for snr in arrSNR:
        for kind in range(kinds):
            tasks.append(
                (R1, Vb1, Vb2, R2, R3, 
                 nSymbolVector, snr, func_type, K_iterations,
                 n, m, kind)
            )

    results = []
    
    # (使用 tqdm 显示“总进度条”)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # (tqdm 100% 抓住了你 [cite: 3.BER_list.py] 的“精髓”)
        futures = [executor.submit(_run_drs_simulation_core, task) for task in tasks]
        for future in tqdm(futures, total=len(tasks), desc=f"模拟 {func_type} (kinds={kinds})"):
            results.append(future.result())

    # --- 汇总结果 ---
    
    # (为 BER vs. SNR 准备)
    # {SNR: [total_errors, total_bits]}
    ber_snr_map = {snr: [0, 0] for snr in arrSNR} 
    
    # (为 BER vs. Iterations 准备)
    # {SNR: [trace_sum_array]}
    ber_trace_map = {snr: np.zeros(K_iterations) for snr in arrSNR}
    
    for res in results:
        total_errors, total_bits, ber_trace_avg, snr = res
        
        ber_snr_map[snr][0] += total_errors
        ber_snr_map[snr][1] += total_bits
        
        ber_trace_map[snr] += ber_trace_avg
        
    # --- 计算最终平均值 ---
    
    final_ber_vs_snr = []
    for snr in arrSNR:
        errors = ber_snr_map[snr][0]
        bits = ber_snr_map[snr][1]
        final_ber_vs_snr.append(errors / bits if bits > 0 else 0)
        
    # (对于 trace，我们只关心一个 SNR 点，但我们计算了所有点)
    # (我们返回“平均”的 trace)
    final_ber_trace = np.sum(list(ber_trace_map.values()), axis=0) / (kinds * len(arrSNR))

    return final_ber_vs_snr, final_ber_trace


def calculate_BER_noise(R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m):
    """
    (这是“重构”后的函数，用于 Fig. 5 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 5])
    """
    ber_vs_snr, _ = _run_simulation_parallel(
        R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m
    )
    print(f"\n  > {func_type} 的最终 BER vs. SNR 结果:")
    print(f"  > SNR (dB): {arrSNR}")
    print(f"  > BER: {[f'{b:.2e}' for b in ber_vs_snr]}")
    return ber_vs_snr

def calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector, SNR, func_type, K_iterations, kinds, n, m):
    """
    (这是“重构”后的函数，用于 Fig. 6 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 6])
    """
    # (我们只关心一个 SNR 点)
    arrSNR = [SNR]
    
    _, ber_trace = _run_simulation_parallel(
        R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m
    )
    print(f"\n  > {func_type} 的最终 BER vs. Iterations 结果 (SNR={SNR}dB):")
    print(f"  > (Iterations: 0, 50, 100, 150, 200)")
    print(f"  > BER: {[f'{ber_trace[i]:.2e}' for i in [0, 49, 99, 149, 199]]}")
    return ber_trace
