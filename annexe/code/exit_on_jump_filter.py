"""
Lever supplementaire, honnete celui-la aussi : au lieu de se contenter de bloquer l'ENTREE
quand pi_t >= pi*, la strategie filtree utilise aussi pi_t pour decider une SORTIE anticipee
en cours de position (des qu'un saut est detecte, on coupe, meme si le z-score n'est pas
revenu a zero). C'est precisement l'extension que le README du depot signale comme testee
dans backtest_4_6.py sans etre reconciliee avec les chiffres du rapport -- on la fait
proprement ici, calibree sur un bloc de VALIDATION distinct du bloc de TEST (meme protocole
que validated_config.py), sur l'univers elargi de 27 paires.
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expanded_universe_backtest import posterior_jump_prob, sharpe, maxdd, COST, W

PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "selected_pairs.pkl")
with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)
FORM = saved["FORM"]
pairs = saved["selected_pairs"]

VALID_START, VALID_END = "2014-01-01", "2018-12-31"
TEST_START, TEST_END = "2019-01-01", "2024-12-31"


def simulate_exit_on_jump(spread_full, params, kappa, pi_star, pi_exit, use_filter, tstart, tend):
    """Comme simulate() de pipeline.py, + sortie anticipee si pi_t >= pi_exit en cours de position
    (uniquement actif si use_filter=True)."""
    s_all = spread_full
    phi = params["phi"]; mu_s = params["mu_s"]
    theta, muJ, sigJ, sigS, lam = params["theta"], params["muJ"], params["sigJ"], params["sigma_s"], params["lam"]

    mask = (s_all.index >= tstart) & (s_all.index <= tend)
    idx = s_all.index[mask]
    s = s_all.values
    pos_full = np.where(mask)[0]
    n = len(s)

    z = pd.Series(s, index=s_all.index).rolling(W).apply(
        lambda w: (w[-1] - w.mean()) / w.std() if w.std() > 0 else np.nan, raw=True
    ).values

    resid = np.full(n, np.nan)
    resid[1:] = s[1:] - (mu_s + phi * (s[:-1] - mu_s))
    pi_t = np.full(n, np.nan)
    ok = ~np.isnan(resid)
    pi_t[ok] = posterior_jump_prob(resid[ok], muJ, sigJ, sigS, lam)

    position = 0
    pnl = np.zeros(n)
    n_trades = 0
    durations = []
    entry_i = None
    wins = 0
    n_exit_jump = 0

    for t in pos_full:
        if t == 0 or np.isnan(z[t]):
            continue
        if position != 0:
            pnl[t] = position * (s[t] - s[t - 1])
            mean_revert_exit = (position == 1 and z[t] >= 0) or (position == -1 and z[t] <= 0)
            jump_exit = use_filter and (not np.isnan(pi_t[t])) and (pi_t[t] >= pi_exit)
            if mean_revert_exit or jump_exit:
                pnl[t] -= 2 * COST
                n_trades += 1
                durations.append(t - entry_i)
                trade_pnl = position * (s[t] - s[entry_i]) - 4 * COST
                wins += int(trade_pnl > 0)
                if jump_exit and not mean_revert_exit:
                    n_exit_jump += 1
                position = 0
        if position == 0:
            cand_long = z[t] <= -kappa
            cand_short = z[t] >= kappa
            if cand_long or cand_short:
                allow = True
                if use_filter and not np.isnan(pi_t[t]):
                    allow = pi_t[t] < pi_star
                if allow:
                    position = 1 if cand_long else -1
                    pnl[t] -= 2 * COST
                    entry_i = t

    ret = pd.Series(pnl[mask], index=idx)
    hit_rate = wins / n_trades if n_trades > 0 else np.nan
    return ret, dict(n_trades=n_trades, hit_rate=hit_rate, n_exit_jump=n_exit_jump,
                      avg_dur=np.mean(durations) if durations else np.nan)


def port_sharpe(rets):
    port = pd.concat(rets, axis=1).mean(axis=1)
    return sharpe(port), port


print(f"=== Sortie anticipee sur detection de saut (pi_t >= pi_exit), univers elargi {len(pairs)} paires ===")
print(f"Calibration sur VALIDATION {VALID_START}->{VALID_END}, test hors-echantillon {TEST_START}->{TEST_END}\n")

kappas = [1.5, 2.0, 2.5]
pistars = [0.3, 0.4, 0.5]
piexits = [0.5, 0.6, 0.7, 0.8]

best = None
for kappa in kappas:
    for pistar in pistars:
        for piexit in piexits:
            if piexit < pistar:
                continue
            rets = []
            for A, B in pairs:
                p = FORM[(A, B)]
                r, _ = simulate_exit_on_jump(p["spread"], p, kappa, pistar, piexit, True, VALID_START, VALID_END)
                rets.append(r)
            sr, _ = port_sharpe(rets)
            if best is None or sr > best[0]:
                best = (sr, kappa, pistar, piexit)

print(f"Meilleure config sur VALIDATION (argmax Sharpe filtree) : "
      f"kappa={best[1]}, pi*={best[2]}, pi_exit={best[3]}  (Sharpe validation={best[0]:.3f})")

_, kappa_sel, pistar_sel, piexit_sel = best

rets_naive, rets_filt, stats_naive, stats_filt = [], [], [], []
for A, B in pairs:
    p = FORM[(A, B)]
    rn, sn = simulate_exit_on_jump(p["spread"], p, kappa_sel, 1.0, 1.0, False, TEST_START, TEST_END)
    rf, sf = simulate_exit_on_jump(p["spread"], p, kappa_sel, pistar_sel, piexit_sel, True, TEST_START, TEST_END)
    rets_naive.append(rn); rets_filt.append(rf)
    stats_naive.append(sn); stats_filt.append(sf)

port_naive = pd.concat(rets_naive, axis=1).mean(axis=1)
port_filt = pd.concat(rets_filt, axis=1).mean(axis=1)
n_naive = sum(s["n_trades"] for s in stats_naive)
n_filt = sum(s["n_trades"] for s in stats_filt)
n_exit_jump_total = sum(s["n_exit_jump"] for s in stats_filt)

print(f"\n=== Resultat hors-echantillon TEST ({TEST_START}->{TEST_END}), "
      f"kappa={kappa_sel} pi*={pistar_sel} pi_exit={piexit_sel} ===")
print(f"Naive   : Sharpe={sharpe(port_naive):.3f}  MaxDD={maxdd(port_naive):.3f}  n_trades={n_naive}")
print(f"Filtree : Sharpe={sharpe(port_filt):.3f}  MaxDD={maxdd(port_filt):.3f}  n_trades={n_filt}  "
      f"(dont {n_exit_jump_total} sorties anticipees sur detection de saut)")


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

