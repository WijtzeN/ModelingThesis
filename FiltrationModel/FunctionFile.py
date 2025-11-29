import numpy as np

def sigmoid(t, tf, alpha):
    # Avoid overflow by using a piecewise function
    exponent = -alpha * (t - tf)
    safe_exp = np.where(exponent > 0, 1 / (1 + np.exp(-exponent)), np.exp(exponent) / (1 + np.exp(exponent)))
    return safe_exp

def musteam(T, Q_ratio=1):
    # Molar masses (g/mol)
    M_H2O = 18.015
    M_N2 = 28.0134

    # Convert volume ratio to mole fraction
    #N_conc = (Q_ratio / M_N2) / ((1 / M_H2O) + (Q_ratio / M_N2))
    N_conc = Q_ratio
    H2O_conc = 1 - N_conc

    # --- Steam properties ---
    T0_H2O = 350             # K
    mu0_H2O = 1.12e-5        # Pa·s
    S_H2O = 1064             # K

    # --- Nitrogen properties ---
    T0_N2 = 300              # K
    mu0_N2 = 1.76e-5         # Pa·s
    S_N2 = 111               # K

    # Individual viscosities using Sutherland’s law
    mu_H2O = mu0_H2O * (T / T0_H2O)**1.5 * (T0_H2O + S_H2O) / (T + S_H2O)
    mu_N2 = mu0_N2 * (T / T0_N2)**1.5 * (T0_N2 + S_N2) / (T + S_N2)

    # Wilke's mixing rule
    def phi(mu_i, mu_j, M_i, M_j):
        return (1 + (mu_i / mu_j)**0.5 * (M_j / M_i)**0.25)**2 / (8 * (1 + M_i / M_j))**0.5

    phi_12 = phi(mu_H2O, mu_N2, M_H2O, M_N2)
    phi_21 = phi(mu_N2, mu_H2O, M_N2, M_H2O)

    mu_mix = (mu_H2O * H2O_conc) / (H2O_conc + phi_12 * N_conc) + \
             (mu_N2 * N_conc) / (N_conc + phi_21 * H2O_conc)

    return mu_mix
