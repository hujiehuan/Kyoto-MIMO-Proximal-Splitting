import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# (设置一个专业的“科研”画图风格)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linestyle'] = '--'

def plot_ber_vs_snr(ber_diode, ber_standard, arrSNR, save_path):
    """
    (这是你 3.BER_list.py [cite: 3.BER_list.py] 里的 get_graph2)
    画出 BER vs. SNR (对应 Fig. 5 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 5])
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(arrSNR, ber_diode, 'o-', label='Diode-DRS (Our Proposed)')
    plt.plot(arrSNR, ber_standard, 's--', label='Standard DRS (g(z))')
    
    plt.yscale('log')
    plt.xticks(np.arange(0, 35, 5))
    plt.ylim(1e-5, 1) # (和你论文 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 5] 保持一致)
    plt.xlabel("SNR per receive antenna (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs. SNR Performance (MIMO 64x100)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  > 图像已保存至: {save_path}")
    plt.close()

def plot_ber_vs_iterations(trace_diode, trace_standard, save_path):
    """
    (这是“升级版”，用于画出 BER vs. Iterations (对应 Fig. 6 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 6]))
    """
    plt.figure(figsize=(10, 6))
    
    iterations = np.arange(len(trace_diode))
    
    plt.plot(iterations, trace_diode, '-', label='Diode-DRS (Our Proposed)')
    plt.plot(iterations, trace_standard, '--', label='Standard DRS (g(z))')
    
    plt.yscale('log')
    plt.ylim(1e-5, 1) # (和你论文 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 5, Fig. 6] 保持一致)
    plt.xlabel("Number of Iterations (K)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Convergence Behavior (SNR = 25 dB)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  > 图像已保存至: {save_path}")
    plt.close()
