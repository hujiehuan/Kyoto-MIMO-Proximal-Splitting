# Python Simulation Framework: Approximate Proximal Operators for Binary Vector Reconstruction

This repository contains the Python simulation framework developed for my M.Sc. research at **Kyoto University**.

The core objective of this research is to approximate the **proximal operator** of the discrete regularizer ($g(z)$)—a critical component in the **Douglas-Rachford Splitting (DRS)** algorithm—using a novel **diode-based analog circuit**. This project simulates and validates the Bit Error Rate (BER) performance of this approach in an overloaded **MIMO signal detection** scenario.

---

## 👨‍💻 My Core Contribution

My primary contribution to this research project was the **end-to-end development of the Python simulation framework** (`src/`) to validate the theoretical hypothesis.

This codebase implements:
1.  **Algorithm Implementation**: The conventional **DRS algorithm** for Sum-of-Absolute-Values (SOAV) optimization, implemented from scratch in Python.
2.  **Circuit Modeling**: A mathematical model of the proposed **diode-based analog circuit** (`diode-DRS`) to approximate the non-linear $g(z)$ function.
3.  **Simulation Pipeline**: A Monte Carlo simulation engine (`src/simulation.py`) to evaluate **BER vs. SNR** and **BER vs. Iterations** performance.
4.  **Visualization**: Custom plotting utilities (`src/utils.py`) to generate publication-quality figures.

---

## 📄 About the Research Paper

This code was used to generate the experimental results for our conference paper, which is currently **under review**.

* **Title**: *"Approximate Proximal Operators for Binary Vector Reconstruction Using Analog Circuit"*
* **Authors**: **Hu Jiehuan (Myself)**, Taisei Kato, Ryo Hayakawa, and Kazunori Hayashi
* **Status**: Submitted to **APSIPA ASC 2025** (Under Review)

> **⚠️ Note on Copyright**: To respect the intellectual property (IP) of the research group and the copyright policies of the conference, the full PDF manuscript and detailed circuit diagrams (`DRS.pdf`) are **not** included in this public repository. This repository focuses solely on the **simulation source code** authored by me.

---

## 📂 Project Structure

```text
Kyoto-MIMO-Proximal-Splitting/
│
├── main.py               # Main entry point for reproduction
├── requirements.txt      # Python dependencies
│
├── src/                  # Core Source Code
│   ├── algorithms.py     # Implementation of DRS and Diode models
│   ├── simulation.py     # Simulation loops and logic
│   └── utils.py          # Plotting and helper functions
│
└── results/              # Generated Results (Figures)
    ├── BER_vs_SNR.png
    └── BER_vs_Iterations.png
