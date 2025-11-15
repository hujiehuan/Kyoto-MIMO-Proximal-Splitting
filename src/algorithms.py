import numpy as np
from scipy.interpolate import interp1d

# --- Circuit Parameters (Fixed Constants) ---
Is = 1.4e-14  # Reverse saturation current
m = 1  # Ideality factor
VT = 26e-3  # Thermal voltage


def create_approx_function_fast(R1, R2, R3, Vb1, Vb2):
    """
    Creates a vectorized approximation function for the diode-based circuit.
    This models the nonlinear V-I characteristics of the proposed analog circuit.
    """

    # --- 1. Define I_in as a function of V_out (x) ---
    def Iin_of_x(x):
        x = np.asarray(x, dtype=np.float64)
        # Theoretical model derived from Kirchhoff's laws and Shockley diode equation
        I1 = (1.0 / R2 + 1.0 / R3) * x - Vb2 / R2 + m * VT * np.arcsinh(x / (2.0 * Is * R3)) / R2
        Iin = (I1
               + m * VT / R1 * np.arcsinh(I1 / (2.0 * Is))
               + I1 * R2 / R1
               - R2 * x / (R1 * R3)
               + Vb1 / R1)
        return Iin

    # --- 2. Sampling and Interpolation (Reverse Mapping) ---
    # We sample I_in(V_out) and create an interpolation to find V_out(I_in)
    x_vals = np.linspace(1e-12, 10.0, 2000, dtype=np.float64)
    y_vals = Iin_of_x(x_vals)  # y = Iin, x = Vout

    # Ensure monotonicity for interpolation
    idx = np.argsort(y_vals)
    y_sorted = y_vals[idx]
    x_sorted = x_vals[idx]
    # Remove duplicates
    keep = np.concatenate(([True], np.diff(y_sorted) != 0))
    y_unique = y_sorted[keep]
    x_unique = x_sorted[keep]
    y_min, y_max = float(y_unique[0]), float(y_unique[-1])

    # --- 3. Determine Bias Voltage V0 (at I_in = 0) ---
    V0 = float(np.interp(0.0, y_unique, x_unique, left=x_unique[0], right=x_unique[-1]))

    # --- 4. Vectorized Output Function ---
    def f_arr(y):
        y = np.asarray(y, dtype=np.float64)
        ya = np.abs(y)
        # Clip input to valid interpolation range
        ya = np.clip(ya, y_min, y_max)
        # Interpolate absolute voltage
        Vabs = np.interp(ya, y_unique, x_unique)
        # Apply odd symmetry and remove bias: V_out = sgn(y) * (V_abs - V0)
        return np.where(y >= 0.0, Vabs - V0, -Vabs + V0).astype(np.float64)

    return f_arr


def g(z, gamma=1.0):
    """
    Standard Proximal Operator of the discrete regularizer (Soft-thresholding-like).
    This serves as the baseline for comparison.
    """
    z = np.asarray(z)
    gamma = gamma * np.ones_like(z)

    out = z.copy()  # Case: -1 <= z <= 1

    # Case: z <= -1 - gamma
    mask1 = z <= -1 - gamma
    out[mask1] = z[mask1] + gamma[mask1]

    # Case: -1 - gamma < z <= -1
    mask2 = (z > -1 - gamma) & (z <= -1)
    out[mask2] = -1

    # Case: 1 < z <= 1 + gamma
    mask4 = (z > 1) & (z <= 1 + gamma)
    out[mask4] = 1

    # Case: z > 1 + gamma
    mask5 = z > 1 + gamma
    out[mask5] = z[mask5] - gamma[mask5]

    return out.astype(np.float64)
