"""
Extension du cadre pairs-trading + jump-diffusion (ArthurK67/pairs-trading-jump-diffusion) :
  - univers de paires elargi (9 paires intra-secteur au lieu de 3)
  - fenetre de backtest etendue (formation 2009-2013, trading 2014-2024 au lieu de 2014-2018/2019-2024)
  - re-estimation glissante annuelle (option ROLLING)
Reutilise la logique de estimate_formation()/simulate()/deflated_sharpe_ratio() de robustness_4_7.py,
seule la source de donnees change (Yahoo chart API directe, yfinance etant bloque sur ce reseau).
"""
import sys, math, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.special import factorial
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yahoo_dl import download_close

np.random.seed(0)

SECTOR_TICKERS = {
    "Paiements":    ["MA", "V", "AXP"],
    "Energie":      ["XOM", "CVX", "COP", "OXY"],
    "Retail":       ["HD", "LOW", "TGT", "WMT", "COST"],
    "Banques":      ["JPM", "BAC", "WFC", "C", "USB"],
    "Staples":      ["KO", "PEP", "PG", "CL", "KMB"],
    "Pharma":       ["PFE", "MRK", "JNJ", "BMY", "LLY"],
    "Telecom":      ["VZ", "T"],
    "Airlines":     ["DAL", "UAL", "AAL", "LUV"],
    "Semis":        ["INTC", "TXN", "QCOM", "MU"],
    "Assurance":    ["MET", "PRU", "AIG", "ALL", "TRV"],
    "Homebuilders": ["DHI", "LEN", "PHM", "NVR"],
    "Utilities":    ["DUK", "SO", "NEE", "AEP", "D"],
}

import itertools
SECTOR_PAIRS = []
for sector, tickers in SECTOR_TICKERS.items():
    for A, B in itertools.combinations(tickers, 2):
        SECTOR_PAIRS.append((sector, A, B))
print(f"Univers candidat : {sum(len(v) for v in SECTOR_TICKERS.values())} tickers, "
      f"{len(SECTOR_PAIRS)} paires intra-secteur", file=sys.stderr)

FORM_START, FORM_END = "2009-01-01", "2013-12-31"   # etendu (vs 2014-2018 original)
TRAD_START, TRAD_END = "2014-01-01", "2024-12-31"   # etendu (vs 2019-2024 original) -> 11 ans au lieu de 6
COST = 0.0010
W = 60
N_MIX = 10
KAPPA, PI_STAR = 2.0, 0.5   # config de reference INCHANGEE (pas de re-optimisation sur la fenetre de test)


def mixture_density(x, muJ, sigJ, sigS, lam, dt=1.0):
    k = np.arange(0, N_MIX + 1)
    w = np.exp(-lam * dt) * (lam * dt) ** k / factorial(k)
    mu_k = k * muJ
    var_k = sigS ** 2 * dt + k * sigJ ** 2
    dens = np.zeros_like(x, dtype=float)
    for kk in range(N_MIX + 1):
        dens += w[kk] * norm.pdf(x, loc=mu_k[kk], scale=np.sqrt(var_k[kk]))
    return dens


def neg_loglik(p, resid):
    lam, muJ, sigJ, sigS = p
    if lam <= 0 or sigJ <= 0 or sigS <= 0:
        return 1e10
    d = np.clip(mixture_density(resid, muJ, sigJ, sigS, lam), 1e-12, None)
    return -np.sum(np.log(d))


def fit_jump_mle(resid):
    x0 = [0.05, np.mean(resid), np.std(resid) * 0.8, np.std(resid) * 0.5]
    res = minimize(neg_loglik, x0, args=(resid,), method="Nelder-Mead",
                    options=dict(maxiter=8000, xatol=1e-8, fatol=1e-10))
    return res.x


def gauss_loglik(resid):
    mu, sig = np.mean(resid), np.std(resid)
    return np.sum(norm.logpdf(resid, mu, sig))


def posterior_jump_prob(x, muJ, sigJ, sigS, lam, dt=1.0):
    k = np.arange(0, N_MIX + 1)
    w = np.exp(-lam * dt) * (lam * dt) ** k / factorial(k)
    mu_k = k * muJ
    var_k = sigS ** 2 * dt + k * sigJ ** 2
    num = np.zeros_like(x, dtype=float)
    den = np.zeros_like(x, dtype=float)
    for kk in range(N_MIX + 1):
        c = w[kk] * norm.pdf(x, mu_k[kk], np.sqrt(var_k[kk]))
        den += c
        if kk >= 1:
            num += c
    return num / np.clip(den, 1e-12, None)


def get_pair_series(A, B, start, end):
    yA = download_close(A, start, end)
    yB = download_close(B, start, end)
    df = pd.concat([yA, yB], axis=1, keys=["A", "B"]).dropna()
    return df


def estimate_formation(df, fstart, fend):
    logA, logB = np.log(df["A"]), np.log(df["B"])
    fmask = (df.index >= fstart) & (df.index <= fend)

    Xf = sm.add_constant(logB[fmask])
    ols = sm.OLS(logA[fmask], Xf).fit()
    alpha, beta = ols.params.iloc[0], ols.params.iloc[1]

    spread = logA - alpha - beta * logB
    spread_f = spread[fmask]
    adf_p = adfuller(spread_f)[1]

    s = spread_f.values
    X2 = sm.add_constant(s[:-1])
    ar1 = sm.OLS(s[1:], X2).fit()
    c, phi = ar1.params[0], ar1.params[1]
    resid = np.asarray(ar1.resid)
    theta = -np.log(phi) if 0 < phi < 1 else np.nan
    mu_s = c / (1 - phi)
    sigma_s = resid.std() * np.sqrt(2 * theta / (1 - np.exp(-2 * theta))) if theta and theta > 0 else resid.std()

    lam, muJ, sigJ, sigS_mle = fit_jump_mle(resid)
    ll_mix = -neg_loglik([lam, muJ, sigJ, sigS_mle], resid)
    ll_g = gauss_loglik(resid)
    LR = 2 * (ll_mix - ll_g)
    p_LR = 1 - chi2.cdf(LR, df=2)

    return dict(alpha=alpha, beta=beta, theta=theta, mu_s=mu_s, sigma_s=sigma_s,
                lam=lam, muJ=muJ, sigJ=sigJ, adf_p=adf_p, LR=LR, p_LR=p_LR,
                spread=spread, phi=phi)


def simulate(spread_full, params, kappa, pi_star, use_filter, tstart, tend):
    s_all = spread_full
    phi = params["phi"]
    mu_s = params["mu_s"]
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

    for t in pos_full:
        if t == 0 or np.isnan(z[t]):
            continue
        if position != 0:
            pnl[t] = position * (s[t] - s[t - 1])
            if (position == 1 and z[t] >= 0) or (position == -1 and z[t] <= 0):
                pnl[t] -= 2 * COST
                n_trades += 1
                durations.append(t - entry_i)
                trade_pnl = position * (s[t] - s[entry_i]) - 4 * COST
                wins += int(trade_pnl > 0)
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
    return ret, dict(n_trades=n_trades, hit_rate=hit_rate,
                      avg_dur=np.mean(durations) if durations else np.nan)


def sharpe(ret):
    r = ret.dropna()
    if r.std() == 0 or len(r) < 30:
        return np.nan
    return np.sqrt(252) * r.mean() / r.std()


def maxdd(ret):
    cum = ret.cumsum()
    return (cum - cum.cummax()).min()


def benjamini_hochberg(pvals, q=0.10):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, n + 1) / n) * q
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed))
    cutoff = ranked[kmax]
    return pvals <= cutoff


if __name__ == "__main__":
    print(f"Fenetre de formation : {FORM_START} -> {FORM_END}")
    print(f"Fenetre de trading   : {TRAD_START} -> {TRAD_END}\n")

    print(f"=== Screening ADF sur l'univers elargi ({len(SECTOR_PAIRS)} paires intra-secteur) ===")
    FORM = {}
    rows = []
    skipped = []
    for sector, A, B in SECTOR_PAIRS:
        try:
            df = get_pair_series(A, B, FORM_START, TRAD_END)
            fmask_check = (df.index >= FORM_START) & (df.index <= FORM_END)
            if fmask_check.sum() < 900:  # pas assez d'historique sur la fenetre de formation
                skipped.append((sector, A, B, f"historique insuffisant ({fmask_check.sum()} obs)"))
                continue
            p = estimate_formation(df, FORM_START, FORM_END)
            FORM[(A, B)] = dict(p, df=df, sector=sector)
            rows.append((sector, A, B, p["beta"], p["adf_p"], p["theta"], p["lam"], p["muJ"], p["p_LR"]))
            print(f"{sector:12s} {A}/{B:5s} beta={p['beta']:7.3f}  ADF_p={p['adf_p']:.4f}  "
                  f"theta={p['theta']:.4f}  lambda={p['lam']:.3f}  LR_p={p['p_LR']:.4f}")
        except Exception as e:
            skipped.append((sector, A, B, str(e)))

    if skipped:
        print(f"\n{len(skipped)} paires ecartees (donnees insuffisantes) :")
        for s in skipped:
            print(f"  {s[0]:12s} {s[1]}/{s[2]:5s} -> {s[3]}")

    n_tested = len(rows)
    pvals = [r[4] for r in rows]
    selected_raw = [pvals[i] < 0.05 for i in range(len(pvals))]
    selected_fdr = benjamini_hochberg(pvals, q=0.10)
    print(f"\nADF p<0.05 (brut, sans correction)      : {sum(selected_raw)}/{n_tested} paires")
    print(f"ADF Benjamini-Hochberg (q=0.10)          : {sum(selected_fdr)}/{n_tested} paires")
    for i, r in enumerate(rows):
        flag = "OK(FDR)" if selected_fdr[i] else ("ok(brut seulement)" if selected_raw[i] else "rejete")
        print(f"  {r[0]:10s} {r[1]}/{r[2]:5s} p={r[4]:.4f}  -> {flag}")

    # ================= Backtest portefeuille elargi (naive vs filtree) =================
    selected_pairs = [(rows[i][1], rows[i][2]) for i in range(len(rows)) if selected_fdr[i]]
    print(f"\n=== Backtest portefeuille elargi : {len(selected_pairs)} paires retenues (FDR q=0.10) ===")
    print("Paires :", ", ".join(f"{A}/{B}" for A, B in selected_pairs))
    print(f"kappa={KAPPA}  pi*={PI_STAR}  fenetre de trading {TRAD_START} -> {TRAD_END}  (config INCHANGEE par rapport a 4.6, aucune re-optimisation)\n")

    rets_naive, rets_filt, stats_naive, stats_filt = [], [], [], []
    per_pair_rows = []
    for A, B in selected_pairs:
        p = FORM[(A, B)]
        rn, sn = simulate(p["spread"], p, KAPPA, 1.0, False, TRAD_START, TRAD_END)
        rf, sf = simulate(p["spread"], p, KAPPA, PI_STAR, True, TRAD_START, TRAD_END)
        rets_naive.append(rn); rets_filt.append(rf)
        stats_naive.append(sn); stats_filt.append(sf)
        per_pair_rows.append((A, B, sn["n_trades"], sf["n_trades"], sharpe(rn), sharpe(rf)))
        print(f"  {A}/{B:5s}  trades naive={sn['n_trades']:3d}  trades filtree={sf['n_trades']:3d}  "
              f"Sharpe naive={sharpe(rn):6.3f}  Sharpe filtree={sharpe(rf):6.3f}")

    port_naive = pd.concat(rets_naive, axis=1).mean(axis=1)
    port_filt = pd.concat(rets_filt, axis=1).mean(axis=1)

    def agg_stats(stats_list):
        return dict(n_trades=sum(s["n_trades"] for s in stats_list),
                    hit_rate=np.nanmean([s["hit_rate"] for s in stats_list]),
                    avg_dur=np.nanmean([s["avg_dur"] for s in stats_list]))

    agg_n = agg_stats(stats_naive)
    agg_f = agg_stats(stats_filt)

    print("\n--- Resultats agreges (portefeuille equipondere) ---")
    print(f"Naive   : Sharpe={sharpe(port_naive):.3f}  MaxDD={maxdd(port_naive):.3f}  "
          f"n_trades={agg_n['n_trades']}  hit_rate={agg_n['hit_rate']*100:.1f}%  duree_moy={agg_n['avg_dur']:.1f}j")
    print(f"Filtree : Sharpe={sharpe(port_filt):.3f}  MaxDD={maxdd(port_filt):.3f}  "
          f"n_trades={agg_f['n_trades']}  hit_rate={agg_f['hit_rate']*100:.1f}%  duree_moy={agg_f['avg_dur']:.1f}j")

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
    print(f"\nTest Jobson-Korkie/Memmel (filtree vs naive) : z={z_jk:.3f}  p-value={p_jk:.4f}")

    # sauvegarde pour reutilisation par le script de re-estimation glissante
    import pickle
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "selected_pairs.pkl"), "wb") as f:
        pickle.dump(dict(selected_pairs=selected_pairs,
                          FORM={k: {kk: vv for kk, vv in v.items() if kk != "df"} for k, v in FORM.items() if k in selected_pairs},
                          port_naive=port_naive, port_filt=port_filt,
                          agg_n=agg_n, agg_f=agg_f, per_pair_rows=per_pair_rows,
                          sharpe_naive=sharpe(port_naive), sharpe_filt=sharpe(port_filt),
                          maxdd_naive=maxdd(port_naive), maxdd_filt=maxdd(port_filt),
                          z_jk=z_jk, p_jk=p_jk), f)
    print(f"\nResultats sauvegardes dans {OUT_DIR}/selected_pairs.pkl")
