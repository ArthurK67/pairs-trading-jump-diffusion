import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.special import factorial
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

np.random.seed(0)

PAIRS = [("MA", "V"), ("XOM", "CVX"), ("HD", "LOW")]
FORM_START, FORM_END = "2014-01-01", "2018-12-31"
TRAD_START, TRAD_END = "2019-01-01", "2024-12-31"
COST = 0.0010
W = 60          # fenetre du z-score glissant naif
N_MIX = 10      # troncature du melange de Poisson

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

def download(ticker):
    df = yf.download(ticker, start=FORM_START, end=TRAD_END, progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df.squeeze()

def estimate_formation(A, B):
    yA, yB = download(A), download(B)
    df = pd.concat([yA, yB], axis=1, keys=["A", "B"]).dropna()
    logA, logB = np.log(df["A"]), np.log(df["B"])
    fmask = (df.index >= FORM_START) & (df.index <= FORM_END)

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
    theta = -np.log(phi)
    mu_s = c / (1 - phi)
    sigma_s = resid.std() * np.sqrt(2 * theta / (1 - np.exp(-2 * theta)))

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
            # position=+1 : long le spread (parie sur une remontee) -> gain si le spread monte
            # position=-1 : short le spread (parie sur une baisse) -> gain si le spread baisse
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

print("=== Estimation (fenetre de formation) ===")
FORM = {}
for A, B in PAIRS:
    p = estimate_formation(A, B)
    FORM[(A, B)] = p
    print(f"{A}/{B}: beta={p['beta']:.3f} ADF_p={p['adf_p']:.4f} theta={p['theta']:.4f} "
          f"lambda={p['lam']:.3f} muJ={p['muJ']:.4f} LR_p={p['p_LR']:.4f}")

print("\n=== Grille de sensibilite (Sharpe portefeuille, strategie filtree) ===")
kappas = [1.5, 2.0, 2.5]
pistars = [0.3, 0.4, 0.5, 0.6, 0.7]

grid = pd.DataFrame(index=pistars, columns=kappas, dtype=float)
naive_by_kappa = {}

for kappa in kappas:
    rets_naive = []
    for A, B in PAIRS:
        p = FORM[(A, B)]
        r, _ = simulate(p["spread"], p, kappa, 1.0, False, TRAD_START, TRAD_END)
        rets_naive.append(r)
    port_naive = pd.concat(rets_naive, axis=1).mean(axis=1)
    naive_by_kappa[kappa] = sharpe(port_naive)

    for pistar in pistars:
        rets_f = []
        for A, B in PAIRS:
            p = FORM[(A, B)]
            r, _ = simulate(p["spread"], p, kappa, pistar, True, TRAD_START, TRAD_END)
            rets_f.append(r)
        port_f = pd.concat(rets_f, axis=1).mean(axis=1)
        grid.loc[pistar, kappa] = sharpe(port_f)

print("\nSharpe naive par kappa:", {k: round(v, 3) for k, v in naive_by_kappa.items()})
print("\nSharpe filtree (lignes=pi*, colonnes=kappa):")
print(grid.round(3))

n_trials = len(kappas) * len(pistars) * len(PAIRS)
print(f"\nNombre d'essais dans la grille (kappa x pi* x paires) = {n_trials}")

print("\n=== Stabilite par sous-periode (kappa=2, pi*=0.5) ===")
subperiods = [("2019-01-01", "2020-12-31"),
              ("2021-01-01", "2022-12-31"),
              ("2023-01-01", "2024-12-31")]

sub_results = []
for (t0, t1) in subperiods:
    rets_naive, rets_filt = [], []
    for A, B in PAIRS:
        p = FORM[(A, B)]
        rn, _ = simulate(p["spread"], p, 2.0, 1.0, False, t0, t1)
        rf, _ = simulate(p["spread"], p, 2.0, 0.5, True, t0, t1)
        rets_naive.append(rn)
        rets_filt.append(rf)
    pn = pd.concat(rets_naive, axis=1).mean(axis=1)
    pf = pd.concat(rets_filt, axis=1).mean(axis=1)
    sub_results.append((t0, t1, sharpe(pn), sharpe(pf)))
    print(f"{t0} -> {t1} : Sharpe naive={sharpe(pn):.3f}  Sharpe filtree={sharpe(pf):.3f}")

def deflated_sharpe_ratio(sr_hat, T, N_trials, var_sr_trials, skew=0.0, kurt=3.0):
    """Bailey & Lopez de Prado (2014). sr_hat/var_sr_trials en unites journalieres ;
    N_trials est le nombre d'essais independants utilises pour approximer le Sharpe
    attendu du meilleur essai sous H0 (correction du biais de selection du maximum)."""
    euler_gamma = 0.5772156649
    sr0 = np.sqrt(var_sr_trials) * (
        (1 - euler_gamma) * norm.ppf(1 - 1.0 / N_trials)
        + euler_gamma * norm.ppf(1 - 1.0 / (N_trials * np.e))
    )
    denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    z = (sr_hat - sr0) * np.sqrt(T - 1) / denom
    return sr0, z, norm.cdf(z)

print("\n=== Deflated Sharpe Ratio (config retenue: kappa=2, pi*=0.5) ===")
rets_f_full = []
for A, B in PAIRS:
    p = FORM[(A, B)]
    rf, _ = simulate(p["spread"], p, 2.0, 0.5, True, TRAD_START, TRAD_END)
    rets_f_full.append(rf)
port_f_full = pd.concat(rets_f_full, axis=1).mean(axis=1).dropna()

sr_hat_daily = port_f_full.mean() / port_f_full.std()
T = len(port_f_full)
grid_sr_daily = (grid.values.flatten() / np.sqrt(252))
grid_sr_daily = grid_sr_daily[~np.isnan(grid_sr_daily)]
var_sr_trials = np.var(grid_sr_daily, ddof=1)
skew = port_f_full.skew()
kurt = port_f_full.kurtosis() + 3  # pandas renvoie l'exces de kurtosis

sr0, z_dsr, p_dsr = deflated_sharpe_ratio(sr_hat_daily, T, n_trials, var_sr_trials, skew, kurt)
print(f"SR observe (quotidien) = {sr_hat_daily:.4f}  (annualise = {sr_hat_daily*np.sqrt(252):.3f})")
print(f"SR0 (benchmark sous H0, {n_trials} essais) = {sr0:.4f}  (annualise = {sr0*np.sqrt(252):.3f})")
print(f"DSR z-stat = {z_dsr:.3f}   p-value(one-sided, SR>SR0) = {1-p_dsr:.4f}")

print("\n=== Config de reference kappa=2, pi*=0.5 : metriques completes ===")
rets_naive_full, rets_filt_full, stats_naive, stats_filt = [], [], [], []
for A, B in PAIRS:
    p = FORM[(A, B)]
    rn, sn = simulate(p["spread"], p, 2.0, 1.0, False, TRAD_START, TRAD_END)
    rf, sf = simulate(p["spread"], p, 2.0, 0.5, True, TRAD_START, TRAD_END)
    rets_naive_full.append(rn); rets_filt_full.append(rf)
    stats_naive.append(sn); stats_filt.append(sf)

port_naive = pd.concat(rets_naive_full, axis=1).mean(axis=1)
port_filt = pd.concat(rets_filt_full, axis=1).mean(axis=1)

def maxdd(ret):
    cum = ret.cumsum()
    return (cum - cum.cummax()).min()

def agg_stats(stats_list):
    return dict(n_trades=sum(s["n_trades"] for s in stats_list),
                hit_rate=np.mean([s["hit_rate"] for s in stats_list]),
                avg_dur=np.mean([s["avg_dur"] for s in stats_list]))

print("Naive   :", dict(sharpe=round(sharpe(port_naive), 4), max_dd=round(maxdd(port_naive), 4)), agg_stats(stats_naive))
print("Filtree :", dict(sharpe=round(sharpe(port_filt), 4), max_dd=round(maxdd(port_filt), 4)), agg_stats(stats_filt))

def jk_memmel_test(rA, rB):
    # Test de Jobson-Korkie (1981), correction de variance de Memmel (2003)
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
print(f"\nTest Jobson-Korkie/Memmel : z = {z_jk:.3f}   p-value = {p_jk:.4f}")
print(f"Sharpe naive annualise    : {sharpe(port_naive):.3f}")
print(f"Sharpe filtree annualise  : {sharpe(port_filt):.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(grid.values.astype(float), aspect="auto", cmap="RdYlGn",
                vmin=-np.nanmax(np.abs(grid.values)), vmax=np.nanmax(np.abs(grid.values)))
ax.set_xticks(range(len(kappas))); ax.set_xticklabels(kappas)
ax.set_yticks(range(len(pistars))); ax.set_yticklabels(pistars)
ax.set_xlabel(r"Seuil $\kappa$")
ax.set_ylabel(r"Seuil $\pi^\ast$")
ax.set_title("Sharpe annualisé — stratégie filtrée, grille de sensibilité")
for i in range(len(pistars)):
    for j in range(len(kappas)):
        ax.text(j, i, f"{grid.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Sharpe annualisé")
plt.tight_layout()
plt.savefig("/sessions/peaceful-stoic-ptolemy/mnt/outputs/fig_4_7_heatmap.png", dpi=220, bbox_inches="tight")
print("\nsaved heatmap")

navy, red = "#1f4e79", "#c0392b"
cum_naive = port_naive.cumsum()
cum_filt = port_filt.cumsum()
dd_naive = cum_naive - cum_naive.cummax()
dd_filt = cum_filt - cum_filt.cummax()

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                            gridspec_kw={"height_ratios": [1.3, 1]})

ax2 = axes2[0]
ax2.plot(cum_naive.index, cum_naive.values, color=red, lw=1.3, label=r"$\mathcal{S}_{\mathrm{naïve}}$ (2.7)")
ax2.plot(cum_filt.index, cum_filt.values, color=navy, lw=1.3, label=r"$\mathcal{S}_{\mathrm{filtrée}}$ (4.3)")
ax2.axhline(0, color="gray", lw=0.7)
ax2.set_ylabel("P&L cumulé (unités de spread)")
ax2.set_title("Portefeuille équipondéré (MA/V, XOM/CVX, HD/LOW) — période de trading 2019-2024")
ax2.legend(loc="upper left", frameon=False)

ax3 = axes2[1]
ax3.fill_between(dd_naive.index, dd_naive.values, 0, color=red, alpha=0.35, label="Drawdown naïve")
ax3.fill_between(dd_filt.index, dd_filt.values, 0, color=navy, alpha=0.45, label="Drawdown filtrée")
ax3.set_ylabel("Drawdown")
ax3.legend(loc="lower left", frameon=False)

for a in axes2:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig("/sessions/peaceful-stoic-ptolemy/mnt/outputs/fig_4_6_equity.png", dpi=220, bbox_inches="tight")
print("saved equity curve")
