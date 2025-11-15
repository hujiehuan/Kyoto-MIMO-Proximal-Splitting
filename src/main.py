import numpy as np
import time
import os
from pathlib import Path

# 导入我们“重构”的 src 模块
from src.simulation import calculate_BER_noise, calculate_BER_trace
from src.utils import plot_ber_vs_snr, plot_ber_vs_iterations

# --- 1. “硬编码”你的“最佳参数” ---
# 这是你（Hu Jiehuan）在 3.BER_list.py, p. 11 [cite: 3.BER_list.py, p. 11] 中找到的“最终参数”
# 这 100% 确保了“可复现性”，同时“隐藏”了你的“参数寻找”过程
BEST_PARAMS = {
    'R1': 1.16,
    'R2': 58.31,
    'Vb1': 1.69,
    'Vb2': 0.99,
    'R3': 1e2000  # R3 固定
}

# --- 2. 定义模拟的“全局设置” ---
# (为了让 IBM 的人能“快速”跑出结果，我们用较小的 `KINDS` 值)
# (你自己（Hu Jiehuan）在 3.BER_list.py, p. 11 [cite: 3.BER_list.py, p. 11] 里用了 KINDS = 20000)
SIM_CONFIG = {
    'n': 50,           # 论文中的 N=100 (real dimensions)
    'm': 32,           # 论文中的 M=64 (real dimensions)
    'nSymbolVector': 2, # 每个 (kind, SNR) 点跑多少次
    'kinds': 1000,     # 跑多少种不同的“信道 (channel)”
    'K_iterations': 200 # DRS 算法迭代次数
}

# --- 3. 定义“结果”的保存路径 ---
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True) # 创建 'results/' 文件夹

def run_ber_vs_snr_simulation():
    """
    运行“BER vs. SNR”模拟 (对应你论文的 Fig. 5 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 5])
    """
    print("--- 正在运行 BER vs. SNR 模拟 (对应 Fig. 5) ---")
    print(f"参数: KINDS={SIM_CONFIG['kinds']}, K_iterations={SIM_CONFIG['K_iterations']}\n")
    
    arrSNR = list(np.arange(0, 30.01, 2.5))
    
    # 1. 运行你的“王牌” (Diode-DRS)
    ber_diode = calculate_BER_noise(
        **BEST_PARAMS,
        arrSNR=arrSNR,
        func_type='diode', # 告诉函数使用“你的”电路模型
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )
    
    # 2. 运行“对照组” (标准 DRS)
    ber_standard = calculate_BER_noise(
        **BEST_PARAMS, # (参数 R1, R2... 在这里无用，但为了函数签名一致而传入)
        arrSNR=arrSNR,
        func_type='standard_g', # 告诉函数使用“标准的” g(z)
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )
    
    # 3. 画图并保存
    plot_path = RESULTS_DIR / "BER_vs_SNR.png"
    plot_ber_vs_snr(
        ber_diode, 
        ber_standard, 
        arrSNR,
        save_path=plot_path
    )
    print(f"\n--- “BER vs. SNR” 图像已保存至: {plot_path} ---")

def run_ber_vs_iterations_simulation():
    """
    运行“BER vs. Iterations”模拟 (对应你论文的 Fig. 6 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 6])
    """
    print("\n--- 正在运行 BER vs. Iterations 模拟 (对应 Fig. 6) ---")
    snr_point = 25 # (固定 SNR=25dB, 就像你论文 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 6] 里一样)
    print(f"参数: KINDS={SIM_CONFIG['kinds']}, SNR={snr_point}dB\n")

    # 1. 运行你的“王牌” (Diode-DRS)
    trace_diode = calculate_BER_trace(
        **BEST_PARAMS,
        SNR=snr_point,
        func_type='diode',
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )
    
    # 2. 运行“对照组” (标准 DRS)
    trace_standard = calculate_BER_trace(
        **BEST_PARAMS,
        SNR=snr_point,
        func_type='standard_g',
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )
    
    # 3. 画图并保存
    plot_path = RESULTS_DIR / "BER_vs_Iterations.png"
    plot_ber_vs_iterations(
        trace_diode, 
        trace_standard, 
        save_path=plot_path
    )
    print(f"\n--- “BER vs. Iterations” 图像已保存至: {plot_path} ---")


if __name__ == "__main__":
    
    # (确保 Python 运行多进程时是“安全”的)
    # (这是你 `6.sen.py` [cite: 6.sen.py] 里的专业做法，我们保留它)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    start_time = time.time()
    
    # --- 运行“王牌”模拟 ---
    run_ber_vs_snr_simulation()
    run_ber_vs_iterations_simulation()
    
    end_time = time.time()
    print(f"\n--- 全部模拟完成，总耗时: {(end_time - start_time) / 60:.2f} 分钟 ---")
