import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

from .algorithms import create_approx_function_fast, g


def make_channel(m, n):
    """
    Generates a real-equivalent MIMO channel matrix from complex Gaussian distribution.
    """
    H_comp = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2)
    H_real = H_comp.real
    H_imag = H_comp.imag
    # Construct real-equivalent matrix (2m x 2n)
    H = np.block([[H_real, -H_imag],
                  [H_imag, H_real]])
    return H_comp, H


def _run_drs_simulation_core(args):
    """
    Worker function for parallel processing. Runs one batch of DRS simulation.
    """
    (R1, Vb1, Vb2, R2, R3,
     nSymbolVector, SNR, func_type, K_iterations,
     n, m, kind_idx) = args

    N_real, M_real = 2 * n, 2 * m

    # Select Proximal Operator Function
    if func_type == 'diode':
        prox_func = create_approx_function_fast(R1, R2, R3, Vb1, Vb2)
    elif func_type == 'standard_g':
        prox_func = g
    else:
        raise ValueError(f"Unknown func_type: {func_type}")

    # Initialize DRS Parameters
    gamma = 1.0
    alpha = 1.0

    # Generate Channel
    _, H = make_channel(m, n)
    HH = H.T @ H
    # Pre-compute projection matrices for efficiency
    ProjMat1_SOAV = inv(np.eye(N_real) + alpha * gamma * HH)
    ProjMat2_SOAV = alpha * gamma * H.T

    # Noise Level
    sigma = np.sqrt(N_real / 10 ** (SNR / 10))
    sigma_r = sigma / np.sqrt(2)

    ber_trace_for_this_kind = np.zeros(K_iterations)

    # Simulation Loop
    for _ in range(nSymbolVector):
        # Generate Source Signal (Binary: -1, +1)
        data = np.random.randint(0, 2, N_real)
        s = -2 * data + 1

        # Generate Observation
        v = np.random.randn(M_real) * sigma_r
        y = H @ s + v

        # Initialize Estimation
        r = np.zeros(N_real)

        # DRS Iterations
        for k in range(K_iterations):
            z = prox_func(r)
            u_t = ProjMat1_SOAV @ (2 * z - r + ProjMat2_SOAV @ y)
            r = r + gamma * (u_t - z)

            # Calculate instantaneous BER
            s_hat = np.where(r >= 0, 1, -1)
            errors = np.count_nonzero(s != s_hat)
            ber_trace_for_this_kind[k] += (errors / N_real)

    # Final BER calculation
    s_hat_final = np.where(r >= 0, 1, -1)
    total_errors_at_end = np.count_nonzero(s != s_hat_final)
    total_bits = N_real * nSymbolVector

    ber_trace_avg = ber_trace_for_this_kind / nSymbolVector

    return total_errors_at_end, total_bits, ber_trace_avg, SNR


def _run_simulation_parallel(R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m):
    """
    Manages parallel execution of the simulation across multiple CPU cores.
    """
    num_workers = os.cpu_count()
    # Prepare tasks
    tasks = []
    for snr in arrSNR:
        for kind_idx in range(kinds):
            tasks.append(
                (R1, Vb1, Vb2, R2, R3,
                 nSymbolVector, snr, func_type, K_iterations,
                 n, m, kind_idx)
            )

    results = []

    # Execute in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_drs_simulation_core, task) for task in tasks]
        for future in tqdm(futures, total=len(tasks), desc=f"Simulating {func_type}"):
            results.append(future.result())

    # Aggregate Results
    ber_snr_map = {snr: [0, 0] for snr in arrSNR}
    ber_trace_map = {snr: np.zeros(K_iterations) for snr in arrSNR}

    for res in results:
        total_errors, total_bits, ber_trace_avg, snr = res

        ber_snr_map[snr][0] += total_errors
        ber_snr_map[snr][1] += total_bits

        ber_trace_map[snr] += ber_trace_avg

    # Compute Averages
    final_ber_vs_snr = []
    for snr in arrSNR:
        errors = ber_snr_map[snr][0]
        bits = ber_snr_map[snr][1]
        final_ber_vs_snr.append(errors / bits if bits > 0 else 0)

    final_ber_trace = np.sum(list(ber_trace_map.values()), axis=0) / (kinds * len(arrSNR))

    return final_ber_vs_snr, final_ber_trace


def calculate_BER_noise(R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m):
    ber_vs_snr, _ = _run_simulation_parallel(
        R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m
    )
    # Print results in a readable format
    print(f"\n  > Final BER vs. SNR for {func_type}:")
    print(f"  > SNR (dB): {arrSNR}")
    print(f"  > BER: {[f'{b:.2e}' for b in ber_vs_snr]}")
    return ber_vs_snr


def calculate_BER_trace(R1, Vb1, Vb2, R2, R3, nSymbolVector, SNR, func_type, K_iterations, kinds, n, m):
    arrSNR = [SNR]
    _, ber_trace = _run_simulation_parallel(
        R1, Vb1, Vb2, R2, R3, nSymbolVector, arrSNR, func_type, K_iterations, kinds, n, m
    )
    print(f"\n  > Final BER Convergence Trace for {func_type} (SNR={SNR}dB):")
    print(f"  > (Sampled Iterations: 0, 50, 100, 150, 200)")
    indices = [0, 49, 99, 149, 199]
    print(f"  > BER: {[f'{ber_trace[i]:.2e}' for i in indices if i < len(ber_trace)]}")
    return ber_trace
