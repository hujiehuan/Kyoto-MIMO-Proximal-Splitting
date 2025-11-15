import numpy as np
import time
import os
from pathlib import Path

# Import from our refactored src modules
from src.simulation import calculate_BER_noise, calculate_BER_trace
from src.utils import plot_ber_vs_snr, plot_ber_vs_iterations

# --- 1. Hard-coded Optimal Parameters ---
# These parameters were optimized in our research
BEST_PARAMS = {
    'R1': 1.16,
    'R2': 58.31,
    'Vb1': 1.69,
    'Vb2': 0.99,
    'R3': 1e2000  # R3 is effectively infinite (open circuit)
}

# --- 2. Global Simulation Configuration ---
# Reduced KINDS for faster reproducibility demonstration
SIM_CONFIG = {
    'n': 50,  # Real dimension N (Complex dimension = 25)
    'm': 32,  # Real dimension M (Complex dimension = 16)
    'nSymbolVector': 2,  # Number of symbol vectors per kind
    'kinds': 1000,  # Number of random channel realizations
    'K_iterations': 200  # Number of DRS iterations
}

# --- 3. Results Directory ---
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_ber_vs_snr_simulation():
    """
    Run BER vs. SNR simulation (reproducing Fig. 5 in the paper).
    """
    print("--- Running BER vs. SNR Simulation (Fig. 5) ---")
    print(f"Config: KINDS={SIM_CONFIG['kinds']}, Iterations={SIM_CONFIG['K_iterations']}\n")

    arrSNR = list(np.arange(0, 30.01, 2.5))

    # 1. Run Proposed Diode-DRS
    print(" > Simulating Proposed Diode-DRS...")
    ber_diode = calculate_BER_noise(
        **BEST_PARAMS,
        arrSNR=arrSNR,
        func_type='diode',
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )

    # 2. Run Standard DRS Baseline
    print(" > Simulating Standard DRS Baseline...")
    ber_standard = calculate_BER_noise(
        **BEST_PARAMS,
        arrSNR=arrSNR,
        func_type='standard_g',
        K_iterations=SIM_CONFIG['K_iterations'],
        nSymbolVector=SIM_CONFIG['nSymbolVector'],
        kinds=SIM_CONFIG['kinds'],
        n=SIM_CONFIG['n'],
        m=SIM_CONFIG['m']
    )

    # 3. Plot and Save
    plot_path = RESULTS_DIR / "BER_vs_SNR.png"
    plot_ber_vs_snr(
        ber_diode,
        ber_standard,
        arrSNR,
        save_path=plot_path
    )
    print(f"\n[Success] BER vs. SNR plot saved to: {plot_path}\n")


def run_ber_vs_iterations_simulation():
    """
    Run BER vs. Iterations simulation (reproducing Fig. 6 in the paper).
    """
    print("--- Running BER vs. Iterations Simulation (Fig. 6) ---")
    snr_point = 25  # Fixed SNR at 25dB
    print(f"Config: KINDS={SIM_CONFIG['kinds']}, SNR={snr_point}dB\n")

    # 1. Run Proposed Diode-DRS
    print(" > Simulating Proposed Diode-DRS...")
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

    # 2. Run Standard DRS Baseline
    print(" > Simulating Standard DRS Baseline...")
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

    # 3. Plot and Save
    plot_path = RESULTS_DIR / "BER_vs_Iterations.png"
    plot_ber_vs_iterations(
        trace_diode,
        trace_standard,
        save_path=plot_path
    )
    print(f"\n[Success] BER vs. Iterations plot saved to: {plot_path}\n")


if __name__ == "__main__":
    # Ensure safe multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    start_time = time.time()

    run_ber_vs_snr_simulation()
    run_ber_vs_iterations_simulation()

    end_time = time.time()
    print(f"--- All simulations completed in {(end_time - start_time) / 60:.2f} minutes ---")
