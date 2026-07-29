import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(42)

# Simulation d'un spread OU augmente de sauts (Euler-Maruyama, approximation
# bernoullienne : 0 ou 1 saut par pas de temps, cf. sous-parties 3.7 / 4.2)
theta   = 0.05     # vitesse de rappel -> demi-vie ~ ln(2)/theta ~ 13.9 jours
mu_s    = 0.0
sigma_s = 0.12
lam     = 0.025    # P(saut) par jour ~ 1 saut tous les 40 jours
mu_J    = -0.9     # sauts en moyenne negatifs (ruptures a la baisse)
sigma_J = 0.55

dt = 1.0
T  = 750

s = np.zeros(T)
jump_times = []
s[0] = mu_s
for t in range(1, T):
    jump_occurs = np.random.rand() < lam * dt
    xi = np.random.normal(mu_J, sigma_J) if jump_occurs else 0.0
    if jump_occurs:
        jump_times.append(t)
    eps = np.random.normal()
    s[t] = s[t-1] + theta * (mu_s - s[t-1]) * dt + sigma_s * math.sqrt(dt) * eps + xi

jump_times = np.array(jump_times, dtype=int)

# Residus AR(1) "naifs" (regression s_t sur s_{t-1}, sans distinguer les sauts)
s_lag  = s[:-1]
s_curr = s[1:]
phi_hat, c_hat = np.polyfit(s_lag, s_curr, 1)
u = s_curr - phi_hat * s_lag - c_hat

# MLE de la densite de melange tronquee (N=10), eq. (mixture_spread) du rapport
N = 10

def mixture_density(x, lam_, mu_j, sig_s, sig_j, dt=1.0, N=10):
    p = np.zeros_like(x)
    for n in range(N + 1):
        w = math.exp(-lam_) * lam_**n / math.factorial(n)
        var = sig_s**2 * dt + n * sig_j**2
        p += w * norm.pdf(x, loc=n * mu_j, scale=math.sqrt(var))
    return p

def neg_loglik(params):
    log_lam, mu_j, log_sig_s, log_sig_j = params
    lam_   = math.exp(log_lam)
    sig_s  = math.exp(log_sig_s)
    sig_j  = math.exp(log_sig_j)
    dens = mixture_density(u, lam_, mu_j, sig_s, sig_j, dt=dt, N=N)
    dens = np.clip(dens, 1e-300, None)
    return -np.sum(np.log(dens))

x0 = [math.log(0.05), -0.3, math.log(np.std(u) * 0.7), math.log(np.std(u))]
res = minimize(neg_loglik, x0, method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
log_lam_hat, mu_J_hat, log_sig_s_hat, log_sig_j_hat = res.x
lam_hat   = math.exp(log_lam_hat)
sig_s_hat = math.exp(log_sig_s_hat)
sig_j_hat = math.exp(log_sig_j_hat)

print("MLE fit  -> lambda=%.4f  mu_J=%.4f  sigma_s=%.4f  sigma_J=%.4f"
      % (lam_hat, mu_J_hat, sig_s_hat, sig_j_hat))
print("Vrai     -> lambda=%.4f  mu_J=%.4f  sigma_s=%.4f  sigma_J=%.4f"
      % (lam, mu_J, sigma_s, sigma_J))

xs = np.linspace(u.min() - 0.2, u.max() + 0.2, 600)
fitted_mix = mixture_density(xs, lam_hat, mu_J_hat, sig_s_hat, sig_j_hat)
gauss_fit  = norm.pdf(xs, loc=np.mean(u), scale=np.std(u))

navy = "#1f4e79"
red  = "#c0392b"
grey = "#aebfd6"

fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.2), gridspec_kw={"height_ratios": [1, 1.25]})

ax = axes[0]
ax.plot(s, color=navy, lw=1.1)
if len(jump_times) > 0:
    ax.scatter(jump_times, s[jump_times], color=red, zorder=5, s=26, label="Sauts simulés")
ax.axhline(mu_s, color="gray", ls="--", lw=1, label=r"$\mu_s$")
ax.set_title(r"(a) Trajectoire simulée du spread $\hat{s}_t$ — OU augmenté de sauts")
ax.set_xlabel(r"$t$ (jours)")
ax.set_ylabel(r"$\hat{s}_t$")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

ax2 = axes[1]
ax2.hist(u, bins=42, density=True, color=grey, edgecolor="white", alpha=0.9,
          label=r"Résidus $\hat u_t$ (empiriques)")
ax2.plot(xs, fitted_mix, color=navy, lw=2.2,
          label="Densité de mélange ajustée (MLE)")
ax2.plot(xs, gauss_fit, color=red, lw=2, ls="--",
          label="Gaussienne pure (sans sauts)")
ax2.set_title(r"(b) Résidus AR(1) du spread : mélange poissonnien vs. gaussienne pure")
ax2.set_xlabel(r"$\hat u_t$")
ax2.set_ylabel("Densité")
ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

for a in axes:
    for spine in ["top", "right"]:
        a.spines[spine].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.82, 1])
out_path = "/sessions/peaceful-stoic-ptolemy/mnt/outputs/fig_4_2_mixture.png"
plt.savefig(out_path, dpi=230, bbox_inches="tight")
print("saved:", out_path)
