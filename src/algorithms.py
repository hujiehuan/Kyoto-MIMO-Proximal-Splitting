import numpy as np
from scipy.interpolate import interp1d

# --- 1. 你的“王牌”核心：模拟电路 (Diode Approximation Function) ---
# (这是从 3.BER_list.py [cite: 3.BER_list.py] 里“重构”的)

# 电路初始参数设计（固定）
Is = 1.4e-14  # 反向饱和电流
m = 1       # 理想因子
VT = 26e-3  # 热电压

def create_approx_function_fast(R1, R2, R3, Vb1, Vb2):
    """
    创建“向量化”的（diode）近似函数。
    这是你（Hu Jiehuan）[cite: 4_0DIECRETE_THESIS (18).pdf, p. 1] 论文 [cite: 4_0DIECRETE_THESIS (18).pdf] 的“核心数学模型”。
    """
    
    # —— 1. Iin(x) 的“数组友好”形式 ——
    def Iin_of_x(x):
        x = np.asarray(x, dtype=np.float64)
        I1 = (1.0 / R2 + 1.0 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2.0 * Is * R3)) / R2
        Iin = (I1
               + m * VT / R1 * np.arcsinh(I1 / (2.0 * Is))
               + I1 * R2 / R1
               - R2 * x / (R1 * R3)
               + Vb1 / R1)
        return Iin

    # —— 2. 采样并构造 Iin(x) 的“可逆”查表 ——
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

    # —— 3. Iin=0 对应的电压 (偏置 V0) ——
    V0 = float(np.interp(0.0, y_unique, x_unique, left=x_unique[0], right=x_unique[-1]))

    # —— 4. 返回“数组版”的阈值函数：输入 y (ndarray), 输出 V ——
    def f_arr(y):
        y = np.asarray(y, dtype=np.float64)
        ya = np.abs(y)
        # 裁剪
        ya = np.clip(ya, y_min, y_max)
        Vabs = np.interp(ya, y_unique, x_unique) # 查表
        # 奇对称 + 去偏置
        return np.where(y >= 0.0, Vabs - V0, -Vabs + V0).astype(np.float64)

    return f_arr


# --- 2. “对照组”核心：标准 g(z) 函数 ---
# (这是从 3.BER_list.py [cite: 3.BER_list.py] 里“重构”的)

def g(z, gamma=1.0):
    """
    标准的（Digital）Proximal Operator of the discrete regularizer (g(z))
    这是你（Hu Jiehuan）[cite: 4_0DIECRETE_THESIS (18).pdf, p. 1] 论文 [cite: 4_0DIECRETE_THESIS (18).pdf, p. 2, Eq. 6] 里的“对照组”。
    """
    z = np.asarray(z)
    gamma = gamma * np.ones_like(z)
    
    out = z.copy() # (mask3: -1 <= z <= 1)
    
    mask1 = z <= -1 - gamma
    out[mask1] = z[mask1] + gamma[mask1]
    
    mask2 = (z > -1 - gamma) & (z <= -1)
    out[mask2] = -1
    
    mask4 = (z > 1) & (z <= 1 + gamma)
    out[mask4] = 1
    
    mask5 = z > 1 + gamma
    out[mask5] = z[mask5] - gamma[mask5]
    
    return out.astype(np.float64)
