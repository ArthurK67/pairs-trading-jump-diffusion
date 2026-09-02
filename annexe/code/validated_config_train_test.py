"""
Calibration (kappa, pi*) sur un bloc de VALIDATION distinct du bloc de TEST final,
pour eviter le biais de selection multiple deja identifie par le DSR (section 4.7).
Univers : les 27 paires retenues par le screening FDR (pipeline.py).
  - Formation (estimation des parametres OU+jump)      : 2009-2013 (inchangee)
  - Validation (calibration de kappa, pi*)              : 2014-2018
  - Test (evaluation, JAMAIS vue pendant la calibration) : 2019-2024
Regle de selection pre-enregistree : argmax du Sharpe du portefeuille FILTRE sur la
validation. Le resultat sur le test est rapporte tel quel, sans deuxieme passage.
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expanded_universe_backtest import simulate, sharpe, maxdd

PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "selected_pairs.pkl")
with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)
FORM = saved["FORM"]
pairs = saved["selected_pairs"]

VALID_START, VALID_END = "2014-01-01", "2018-12-31"
TEST_START, TEST_END = "2019-01-01", "2024-12-31"

kappas = [1.5, 2.0, 2.5]
pistars = [0.3, 0.4, 0.5, 0.6, 0.7]

print(f"=== Calibration (kappa, pi*) sur validation {VALID_START}->{VALID_END}, "
      f"test hors-echantillon {TEST_START}->{TEST_END} ({len(pairs)} paires) ===\n")

grid_valid = {}
for kappa in kappas:
    for pistar in pistars:
        rets = []
        for A, B in pairs:
            p = FORM[(A, B)]
            r, _ = simulate(p["spread"], p, kappa, pistar, True, VALID_START, VALID_END)
            rets.append(r)
        port = pd.concat(rets, axis=1).mean(axis=1)
        grid_valid[(kappa, pistar)] = sharpe(port)

print("Grille de validation (Sharpe filtree, 2014-2018) :")
for kappa in kappas:
    line = "  ".join(f"pi*={ps}:{grid_valid[(kappa,ps)]:.3f}" for ps in pistars)
    print(f"  kappa={kappa}: {line}")

best_cfg = max(grid_valid, key=grid_valid.get)
print(f"\nConfig retenue (argmax Sharpe filtree sur VALIDATION uniquement) : "
      f"kappa={best_cfg[0]}, pi*={best_cfg[1]}  (Sharpe validation={grid_valid[best_cfg]:.3f})")

# --- Evaluation hors-echantillon (jamais vue pendant la calibration) ---
kappa_sel, pistar_sel = best_cfg
rets_naive, rets_filt, stats_naive, stats_filt = [], [], [], []
for A, B in pairs:
    p = FORM[(A, B)]
    rn, sn = simulate(p["spread"], p, kappa_sel, 1.0, False, TEST_START, TEST_END)
    rf, sf = simulate(p["spread"], p, kappa_sel, pistar_sel, True, TEST_START, TEST_END)
    rets_naive.append(rn); rets_filt.append(rf)
    stats_naive.append(sn); stats_filt.append(sf)

port_naive = pd.concat(rets_naive, axis=1).mean(axis=1)
port_filt = pd.concat(rets_filt, axis=1).mean(axis=1)
n_naive = sum(s["n_trades"] for s in stats_naive)
n_filt = sum(s["n_trades"] for s in stats_filt)

print(f"\n=== Resultat hors-echantillon sur le TEST ({TEST_START}->{TEST_END}), "
      f"config figee kappa={kappa_sel} pi*={pistar_sel} ===")
print(f"Naive   : Sharpe={sharpe(port_naive):.3f}  MaxDD={maxdd(port_naive):.3f}  n_trades={n_naive}")
print(f"Filtree : Sharpe={sharpe(port_filt):.3f}  MaxDD={maxdd(port_filt):.3f}  n_trades={n_filt}")


def jk_memmel_test(rA, rB):
    df = pd.concat([rA, rB], axis=1).dropna()
    df.columns = ["A", "B"]
    T = len(df)
    muA, muB = df["A"].mean(), df["B"].mean()
    sA, sB = df["A"].std(), df["B"].std()
    sAB = df["A"].cov(df["B"])
    theta_var = (1/T) * (2*sA**2*sB**2 - 2*sA*sB*sAB + 0.5*muA**2*sB**2
                          + 0.5*muB**2*sA**2 - (muA*muB/(sA*sB))*sAB**2)
    z_stat = (sB*muA - sA*muB) / np.sqrt(theta_var)
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val

z_jk, p_jk = jk_memmel_test(port_filt, port_naive)
print(f"\nTest Jobson-Korkie/Memmel (filtree vs naive, sur TEST) : z={z_jk:.3f}  p-value={p_jk:.4f}")

