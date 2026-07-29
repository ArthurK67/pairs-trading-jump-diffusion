"""Application du cadre pairs trading + jump-diffusion (2.x / 4.x) a des donnees reelles via yfinance."""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from statsmodels.tsa.stattools import adfuller

TICKER_Y = "MA"           # actif Y (regresse)
TICKER_X = "V"            # actif X (regresseur)
START    = "2019-01-01"
END      = "2024-12-31"
KAPPA    = 2.0            # seuil d'entree sur le z-score (sous-partie 2.7)
DT       = 1.0
N_TRUNC  = 10             # troncature de la somme de Poisson (sous-partie 3.7)

OUT_DIR = "/sessions/peaceful-stoic-ptolemy/mnt/outputs/"

raw = yf.download([TICKER_Y, TICKER_X], start=START, end=END, progress=False)["Close"]
raw = raw.dropna()
Y = np.log(raw[TICKER_Y])
X = np.log(raw[TICKER_X])
dates = raw.index

print(f"Donnees : {len(raw)} observations, du {dates[0].date()} au {dates[-1].date()}")

# hedge ratio OLS (2.4) + spread + test ADF (2.5)
beta_hat, alpha_hat = np.polyfit(X.values, Y.values, 1)
spread = (Y - beta_hat * X - alpha_hat).values

adf_stat, adf_pval, *_ = adfuller(spread, autolag="AIC")
print(f"\nHedge ratio  beta_hat = {beta_hat:.4f}")
print(f"Test ADF sur le spread : stat = {adf_stat:.3f}, p-value = {adf_pval:.4f}"
      f"  -> {'stationnaire (cointegration plausible)' if adf_pval < 0.05 else 'NON stationnaire -- paire a reconsiderer'}")

# regression AR(1) du spread (2.6)
s_lag, s_curr = spread[:-1], spread[1:]
phi_hat, c_hat = np.polyfit(s_lag, s_curr, 1)
theta_hat = (1 - phi_hat) / DT
mu_s_hat  = c_hat / (1 - phi_hat)
u = s_curr - phi_hat * s_lag - c_hat
sigma_u_hat = u.std(ddof=1)
sigma_s_hat = sigma_u_hat / math.sqrt(DT)
half_life = math.log(2) / theta_hat if theta_hat > 0 else np.nan

print(f"\nOU (sans sauts) : theta_hat = {theta_hat:.4f}  mu_s_hat = {mu_s_hat:.4f}  "
      f"sigma_s_hat = {sigma_s_hat:.4f}  demi-vie = {half_life:.1f} jours")

# z-score et signaux (2.7)
window = 60
s_series = pd.Series(spread, index=dates)
roll_mean = s_series.rolling(window).mean().shift(1)  # shift(1) : pas de look-ahead
roll_std  = s_series.rolling(window).std().shift(1)
z_score = (s_series - roll_mean) / roll_std

long_signal  = z_score < -KAPPA
short_signal = z_score > KAPPA

# estimation jump-diffusion par MLE sur les residus reels (4.2)
def mixture_density(x, lam_, mu_j, sig_s, sig_j, dt=DT, N=N_TRUNC):
    p = np.zeros_like(x, dtype=float)
    for n in range(N + 1):
        w = math.exp(-lam_) * lam_**n / math.factorial(n)
        var = sig_s**2 * dt + n * sig_j**2
        p += w * norm.pdf(x, loc=n * mu_j, scale=math.sqrt(var))
    return p

def neg_loglik_jump(params, data):
    log_lam, mu_j, log_sig_s, log_sig_j = params
    lam_, sig_s, sig_j = math.exp(log_lam), math.exp(log_sig_s), math.exp(log_sig_j)
    dens = np.clip(mixture_density(data, lam_, mu_j, sig_s, sig_j), 1e-300, None)
    return -np.sum(np.log(dens))

def neg_loglik_gauss(log_sig, data):
    sig = math.exp(float(log_sig[0]) if hasattr(log_sig, "__len__") else log_sig)
    dens = np.clip(norm.pdf(data, loc=0.0, scale=sig), 1e-300, None)
    return -np.sum(np.log(dens))

x0 = [math.log(0.03), -0.001, math.log(u.std() * 0.8), math.log(u.std() * 1.5)]
res_jump = minimize(neg_loglik_jump, x0, args=(u,), method="Nelder-Mead",
                     options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-9})
log_lam_hat, mu_J_hat, log_sig_s_hat, log_sig_j_hat = res_jump.x
lam_hat   = math.exp(log_lam_hat)
sig_s_hat = math.exp(log_sig_s_hat)
sig_j_hat = math.exp(log_sig_j_hat)
ll_jump   = -res_jump.fun

res_gauss = minimize(neg_loglik_gauss, [math.log(u.std())], args=(u,), method="Nelder-Mead")
ll_gauss  = -res_gauss.fun

# test du rapport de vraisemblance H0: lambda=0 (3.7 / 4.2)
LR = -2 * (ll_gauss - ll_jump)
p_value_LR = 1 - chi2.cdf(LR, df=2)

print(f"\nEstimation jump-diffusion (MLE, melange tronque N={N_TRUNC}) sur les residus reels :")
print(f"  lambda_hat = {lam_hat:.4f}  (~ 1 saut tous les {1/lam_hat:.0f} jours de bourse)")
print(f"  mu_J_hat   = {mu_J_hat:.4f}")
print(f"  sigma_s_hat(diffusion) = {sig_s_hat:.4f}")
print(f"  sigma_J_hat(sauts)     = {sig_j_hat:.4f}")
print(f"\nTest du rapport de vraisemblance (H0 : pas de sauts) :")
print(f"  LR = {LR:.2f}  (chi2, 2 ddl)  p-value = {p_value_LR:.4f}"
      f"  -> {'sauts statistiquement significatifs' if p_value_LR < 0.05 else 'H0 non rejetee : sauts non significatifs sur cette paire/periode'}")

navy, red, grey, green = "#1f4e79", "#c0392b", "#aebfd6", "#2e7d32"

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                          gridspec_kw={"height_ratios": [1.1, 1]})

ax = axes[0]
ax.plot(dates, spread, color=navy, lw=1)
ax.axhline(mu_s_hat, color="gray", ls="--", lw=1, label=r"$\hat\mu_s$ (OU)")
ax.set_title(f"Spread $\\hat s_t$ = ln({TICKER_Y}) $-$ {beta_hat:.3f} $\\times$ ln({TICKER_X})"
             f"  |  ADF p-value = {adf_pval:.3f}")
ax.set_ylabel(r"$\hat s_t$")
ax.legend(loc="upper right", frameon=False, fontsize=9)

ax2 = axes[1]
ax2.plot(dates, z_score, color=navy, lw=0.9)
ax2.axhline(KAPPA, color=red, ls="--", lw=1)
ax2.axhline(-KAPPA, color=green, ls="--", lw=1)
ax2.axhline(0, color="gray", lw=0.8)
ax2.scatter(dates[short_signal], z_score[short_signal], color=red, s=10, zorder=5, label="Entree vendeuse")
ax2.scatter(dates[long_signal], z_score[long_signal], color=green, s=10, zorder=5, label="Entree acheteuse")
ax2.set_title(f"z-score (fenetre glissante {window}j) et signaux ($\\kappa={KAPPA}$)")
ax2.set_ylabel("z-score")
ax2.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR + f"spread_zscore_{TICKER_Y}_{TICKER_X}.png", dpi=200)

fig2, ax3 = plt.subplots(figsize=(8, 5.5))
xs = np.linspace(u.min() - 0.2 * u.std(), u.max() + 0.2 * u.std(), 600)
ax3.hist(u, bins=50, density=True, color=grey, edgecolor="white", alpha=0.9, label=r"Résidus $\hat u_t$ (réels)")
ax3.plot(xs, mixture_density(xs, lam_hat, mu_J_hat, sig_s_hat, sig_j_hat), color=navy, lw=2.2,
         label="Densité de mélange ajustée (MLE)")
ax3.plot(xs, norm.pdf(xs, loc=0.0, scale=math.exp(res_gauss.x[0])), color=red, lw=2, ls="--",
         label="Gaussienne pure (sans sauts)")
ax3.set_title(f"{TICKER_Y}/{TICKER_X} : résidus AR(1) du spread vs. densité jump-diffusion (MLE)")
ax3.set_xlabel(r"$\hat u_t$")
ax3.set_ylabel("Densité")
ax3.legend(loc="upper right", frameon=False, fontsize=9)
plt.tight_layout()
fig2.savefig(OUT_DIR + f"residuals_mixture_{TICKER_Y}_{TICKER_X}.png", dpi=200)

print(f"\nFigures sauvegardees dans {OUT_DIR}")
print(f"  spread_zscore_{TICKER_Y}_{TICKER_X}.png")
print(f"  residuals_mixture_{TICKER_Y}_{TICKER_X}.png")
