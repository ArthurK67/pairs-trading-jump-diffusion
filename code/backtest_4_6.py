"""
Backtest empirique complet (parties 4.1-4.5) sur donnees reelles :
compare la strategie naive (2.7-2.9) a la strategie filtree par les
sauts (4.3), sur trois paires sectorielles reelles, avec decoupage
formation/trading (4.5), estimation jump-diffusion par MLE (4.2),
protocole de comparaison (4.4 : couts, metriques, test de Sharpe,
regimes de marche).
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

# ------------------------------------------------------------------
# 0. Parametres globaux
# ------------------------------------------------------------------
PAIRS = [("MA", "V"), ("XOM", "CVX"), ("HD", "LOW")]
START = "2014-01-01"
FORMATION_END = "2018-12-31"
END = "2024-12-31"

KAPPA = 2.0
KAPPA_STOP = 3.5
PI_STAR = 0.5
EXIT_Z = 0.5
WINDOW = 60          # fenetre roulante pour le z-score naif
N_TRUNC = 10
COST_BPS = 0.0010    # 10 bps par jambe, applique a l'ouverture et a la cloture

np.random.seed(0)

# ------------------------------------------------------------------
# 1. Telechargement des donnees
# ------------------------------------------------------------------
tickers = sorted(set([t for p in PAIRS for t in p] + ["^GSPC"]))
raw = yf.download(tickers, start=START, end=END, progress=False)["Close"].dropna()
log_px = np.log(raw)

gspc = log_px["^GSPC"]
gspc_ret = gspc.diff()
realized_vol = gspc_ret.rolling(20).std()

# ------------------------------------------------------------------
# 2. Outils : melange jump-diffusion (identique a 4.2)
# ------------------------------------------------------------------
def mixture_density(x, lam_, mu_j, sig_s, sig_j, dt=1.0, N=N_TRUNC):
    p = np.zeros_like(x, dtype=float)
    for n in range(N + 1):
        w = math.exp(-lam_) * lam_**n / math.factorial(n)
        var = sig_s**2 * dt + n * sig_j**2
        p += w * norm.pdf(x, loc=n * mu_j, scale=math.sqrt(var))
    return p

def fit_jump_mle(u, dt=1.0):
    def negloglik(params):
        log_lam, mu_j, log_sig_s, log_sig_j = params
        lam_, sig_s, sig_j = math.exp(log_lam), math.exp(log_sig_s), math.exp(log_sig_j)
        dens = np.clip(mixture_density(u, lam_, mu_j, sig_s, sig_j, dt=dt), 1e-300, None)
        return -np.sum(np.log(dens))
    x0 = [math.log(0.05), 0.0, math.log(u.std() * 0.8), math.log(u.std() * 1.5)]
    res = minimize(negloglik, x0, method="Nelder-Mead",
                    options={"maxiter": 6000, "xatol": 1e-8, "fatol": 1e-8})
    log_lam, mu_j, log_sig_s, log_sig_j = res.x
    return dict(lam=math.exp(log_lam), muJ=mu_j, sigS=math.exp(log_sig_s), sigJ=math.exp(log_sig_j),
                loglik=-res.fun)

def gauss_loglik(u):
    sig = u.std()
    return np.sum(norm.logpdf(u, loc=0.0, scale=sig))

def posterior_jump_prob(u, lam_, mu_j, sig_s, sig_j, dt=1.0):
    no_jump = (1 - lam_*dt) * norm.pdf(u, loc=0.0, scale=math.sqrt(sig_s**2*dt))
    one_jump = lam_*dt * norm.pdf(u, loc=mu_j, scale=math.sqrt(sig_s**2*dt + sig_j**2))
    return one_jump / (no_jump + one_jump)

# ------------------------------------------------------------------
# 3. Estimation par paire sur la fenetre de FORMATION
# ------------------------------------------------------------------
pair_params = {}
print("=== Fenetre de formation :", START, "->", FORMATION_END, "===\n")
for Y, X in PAIRS:
    form = log_px.loc[START:FORMATION_END]
    y, x = form[Y].values, form[X].values
    beta, alpha = np.polyfit(x, y, 1)
    spread_form = y - beta * x - alpha

    adf_stat, adf_p, *_ = adfuller(spread_form, autolag="AIC")

    s_lag, s_curr = spread_form[:-1], spread_form[1:]
    phi, c = np.polyfit(s_lag, s_curr, 1)
    theta = (1 - phi)
    mu_s = c / (1 - phi)
    u_form = s_curr - phi * s_lag - c
    sigma_s = u_form.std()
    half_life = math.log(2) / theta if theta > 0 else np.nan

    fit = fit_jump_mle(u_form)
    ll_gauss = gauss_loglik(u_form)
    LR = -2 * (ll_gauss - fit["loglik"])
    from scipy.stats import chi2
    p_LR = 1 - chi2.cdf(LR, df=2)

    pair_params[(Y, X)] = dict(beta=beta, alpha=alpha, phi=phi, c=c, theta=theta,
                                mu_s=mu_s, sigma_s=sigma_s, half_life=half_life,
                                lam=fit["lam"], muJ=fit["muJ"], sigS=fit["sigS"], sigJ=fit["sigJ"],
                                adf_p=adf_p, LR=LR, p_LR=p_LR)

    print(f"{Y}/{X}: beta={beta:.3f}  ADF p={adf_p:.4f}  demi-vie={half_life:.1f}j  "
          f"lambda={fit['lam']:.3f}  muJ={fit['muJ']:.3f}  LR={LR:.1f}  p_LR={p_LR:.4f}")

# ------------------------------------------------------------------
# 4. Simulation des deux strategies sur la fenetre de TRADING
# ------------------------------------------------------------------
def simulate(Y, X, params, filtered):
    trad = log_px.loc[FORMATION_END:END]
    y, x = trad[Y].values, trad[X].values
    dates = trad.index
    spread = y - params["beta"] * x - params["alpha"]
    T = len(spread)

    phi, c = params["phi"], params["c"]
    u = np.full(T, np.nan)
    u[1:] = spread[1:] - phi * spread[:-1] - c

    pi_t = np.full(T, np.nan)
    pi_t[1:] = posterior_jump_prob(u[1:], params["lam"], params["muJ"], params["sigS"], params["sigJ"])

    roll_mean = pd.Series(spread).rolling(WINDOW).mean().shift(1).values
    roll_std = pd.Series(spread).rolling(WINDOW).std().shift(1).values
    z = (spread - roll_mean) / roll_std

    position = 0          # +1 long spread, -1 short spread, 0 flat
    entry_day = None
    daily_pnl = np.zeros(T)
    trades = []            # (pnl, duration)

    for t in range(1, T):
        if np.isnan(z[t]):
            continue

        if position != 0:
            # P&L journalier proportionnel a la variation du spread
            daily_pnl[t] = position * (spread[t] - spread[t-1])

            duration = t - entry_day
            exit_now = False
            if abs(z[t]) < EXIT_Z:
                exit_now = True
            elif abs(z[t]) > KAPPA_STOP:
                exit_now = True
            elif duration > 3 * params["half_life"]:
                exit_now = True
            elif filtered and pi_t[t] >= PI_STAR:
                exit_now = True

            if exit_now:
                daily_pnl[t] -= 2 * COST_BPS
                trades.append((np.sign(position) * (spread[t]-spread[entry_day]), duration))
                position = 0
                entry_day = None
        else:
            candidate_short = z[t] > KAPPA
            candidate_long = z[t] < -KAPPA
            allow = (not filtered) or (pi_t[t] < PI_STAR)
            if candidate_short and allow:
                position = -1
                entry_day = t
                daily_pnl[t] -= 2 * COST_BPS
            elif candidate_long and allow:
                position = 1
                entry_day = t
                daily_pnl[t] -= 2 * COST_BPS

    return pd.Series(daily_pnl, index=dates), trades

results_naive = {}
results_filtered = {}
for Y, X in PAIRS:
    pnl_n, tr_n = simulate(Y, X, pair_params[(Y, X)], filtered=False)
    pnl_f, tr_f = simulate(Y, X, pair_params[(Y, X)], filtered=True)
    results_naive[(Y, X)] = (pnl_n, tr_n)
    results_filtered[(Y, X)] = (pnl_f, tr_f)

# ------------------------------------------------------------------
# 5. Portefeuille agrege (equipondere) et metriques
# ------------------------------------------------------------------
def portfolio_returns(results):
    df = pd.concat({p: r[0] for p, r in results.items()}, axis=1)
    return df.mean(axis=1)

port_naive = portfolio_returns(results_naive)
port_filtered = portfolio_returns(results_filtered)

def metrics(port, results):
    sharpe = math.sqrt(252) * port.mean() / port.std()
    cum = port.cumsum()
    dd = (cum - cum.cummax()).min()
    all_trades = [t for r in results.values() for t in r[1]]
    n_trades = len(all_trades)
    hit_rate = np.mean([1 if pnl > 0 else 0 for pnl, dur in all_trades]) if all_trades else np.nan
    avg_dur = np.mean([dur for pnl, dur in all_trades]) if all_trades else np.nan
    return dict(sharpe=sharpe, max_dd=dd, n_trades=n_trades, hit_rate=hit_rate, avg_dur=avg_dur)

m_naive = metrics(port_naive, results_naive)
m_filtered = metrics(port_filtered, results_filtered)

print("\n=== Resultats agreges (portefeuille equipondere, 3 paires) ===")
print("Naive   :", m_naive)
print("Filtree :", m_filtered)

# ------------------------------------------------------------------
# 6. Test de Jobson-Korkie / Memmel
# ------------------------------------------------------------------
common = port_naive.index.intersection(port_filtered.index)
A = port_naive.loc[common].values
B = port_filtered.loc[common].values
Tn = len(A)
muA, muB = A.mean(), B.mean()
sigA, sigB = A.std(), B.std()
sigAB = np.cov(A, B)[0, 1]

g = sigB*muA - sigA*muB
theta = (2*sigA**2*sigB**2 - 2*sigA*sigB*sigAB + 0.5*muA**2*sigB**2 + 0.5*muB**2*sigA**2
         - (muA*muB/(sigA*sigB))*sigAB**2) / Tn
z_JK = g / math.sqrt(theta)
p_JK = 2*(1 - norm.cdf(abs(z_JK)))

print(f"\nTest Jobson-Korkie/Memmel : z = {z_JK:.3f}, p-value = {p_JK:.4f}")
print(f"Sharpe naive annualise    : {m_naive['sharpe']:.3f}")
print(f"Sharpe filtree annualise  : {m_filtered['sharpe']:.3f}")

# ------------------------------------------------------------------
# 7. Segmentation par regime de marche (vol realisee du S&P500)
# ------------------------------------------------------------------
vol_trading = realized_vol.loc[FORMATION_END:END].reindex(common)
median_vol = vol_trading.median()
high_vol_mask = (vol_trading > median_vol).values
low_vol_mask = ~high_vol_mask

def sharpe_of(x):
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std() == 0:
        return np.nan
    return math.sqrt(252) * x.mean() / x.std()

print("\n=== Par regime (vol. realisee S&P500, mediane sur la periode) ===")
print(f"Naive   - vol haute: Sharpe={sharpe_of(A[high_vol_mask]):.3f}   vol basse: Sharpe={sharpe_of(A[low_vol_mask]):.3f}")
print(f"Filtree - vol haute: Sharpe={sharpe_of(B[high_vol_mask]):.3f}   vol basse: Sharpe={sharpe_of(B[low_vol_mask]):.3f}")

print("\nNb de jours en periode de forte volatilite:", high_vol_mask.sum(), "/", len(high_vol_mask))

# ------------------------------------------------------------------
# 8. Figure : courbes d'equite + drawdown
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

navy, red = "#1f4e79", "#c0392b"

cum_naive = port_naive.cumsum()
cum_filtered = port_filtered.cumsum()
dd_naive = cum_naive - cum_naive.cummax()
dd_filtered = cum_filtered - cum_filtered.cummax()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                          gridspec_kw={"height_ratios": [1.3, 1]})

ax = axes[0]
ax.plot(cum_naive.index, cum_naive.values, color=red, lw=1.3, label=r"$\mathcal{S}_{\mathrm{naïve}}$ (2.7)")
ax.plot(cum_filtered.index, cum_filtered.values, color=navy, lw=1.3, label=r"$\mathcal{S}_{\mathrm{filtrée}}$ (4.3)")
ax.axhline(0, color="gray", lw=0.7)
ax.set_ylabel("P&L cumulé (unités de spread)")
ax.set_title("Portefeuille équipondéré (MA/V, XOM/CVX, HD/LOW) — période de trading 2019-2024")
ax.legend(loc="upper left", frameon=False)

ax2 = axes[1]
ax2.fill_between(dd_naive.index, dd_naive.values, 0, color=red, alpha=0.35, step=None, label="Drawdown naïve")
ax2.fill_between(dd_filtered.index, dd_filtered.values, 0, color=navy, alpha=0.45, step=None, label="Drawdown filtrée")
ax2.set_ylabel("Drawdown")
ax2.legend(loc="lower left", frameon=False)

for a in axes:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

plt.tight_layout()
out_path = "/sessions/peaceful-stoic-ptolemy/mnt/outputs/fig_4_6_equity.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
print("\nsaved:", out_path)
