import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(7)

# Memes parametres de simulation que la figure 4.2, pour rester coherent
theta   = 0.05
mu_s    = 0.0
sigma_s = 0.12
lam     = 0.025
mu_J    = -0.9
sigma_J = 0.55
dt = 1.0
T  = 750
kappa   = 2.0     # seuil d'entree sur le z-score (sous-partie 2.7)
pi_star = 0.5     # seuil de probabilite de saut (sous-partie 4.3)

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

# z-score "naif" (2.7) : moyenne/ecart-type empiriques sur fenetre glissante
# du spread brut (donc contamine par les sauts)
window = 60
s_series = s
roll_mean = np.full(T, np.nan)
roll_std = np.full(T, np.nan)
for t in range(window, T):
    w = s_series[t-window:t]
    roll_mean[t] = w.mean()
    roll_std[t] = w.std(ddof=1)
z_naive = (s_series - roll_mean) / roll_std

# On utilise ici les VRAIS parametres de simulation comme proxy des parametres
# estimes par MLE en 4.2 (meme logique pedagogique que la figure 4.2 : verifier
# le mecanisme sur un cas ou la verite est connue)
phi = 1 - theta * dt
c = theta * dt * mu_s
u = np.full(T, np.nan)
u[1:] = s[1:] - phi * s[:-1] - c

def phi_gauss(x, v, V2):
    return norm.pdf(x, loc=v, scale=math.sqrt(V2))

pi_t = np.full(T, np.nan)
for t in range(1, T):
    no_jump = (1 - lam*dt) * phi_gauss(u[t], 0.0, sigma_s**2 * dt)
    one_jump = lam*dt * phi_gauss(u[t], mu_J, sigma_s**2 * dt + sigma_J**2)
    pi_t[t] = one_jump / (no_jump + one_jump)

# Signaux : regle naive (2.7) vs regle filtree (4.3) -- meme z-score, la regle
# filtree ne change que le critere d'ENTREE, en lui ajoutant pi_t < pi*
candidate = np.abs(z_naive) > kappa
filtered_entry = candidate & (pi_t < pi_star)
blocked = candidate & (pi_t >= pi_star)

navy, red, green, grey, orange = "#1f4e79", "#c0392b", "#2e7d32", "#aebfd6", "#e67e22"

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                          gridspec_kw={"height_ratios": [1.1, 0.8, 0.9]})

ax = axes[0]
ax.plot(s, color=navy, lw=1)
ax.scatter(jump_times, s[jump_times], color="black", marker="x", s=35, zorder=6,
           label="Sauts simulés (vérité)")
ax.scatter(np.where(filtered_entry)[0], s[filtered_entry], color=green, s=30, zorder=5,
           label=r"Entrée filtrée ($\pi_t<\pi^\ast$)")
ax.scatter(np.where(blocked)[0], s[blocked], color=red, marker="^", s=45, zorder=7,
           label=r"Signal bloqué par le filtre ($\pi_t\geq\pi^\ast$)")
ax.axhline(mu_s, color="gray", ls="--", lw=1)
ax.set_ylabel(r"$\hat s_t$")
ax.set_title(r"(a) Spread simulé — signaux d'entrée : règle naïve vs règle filtrée")
ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=1)

ax2 = axes[1]
ax2.plot(pi_t, color=navy, lw=1)
ax2.axhline(pi_star, color=red, ls="--", lw=1, label=r"$\pi^\ast=0{,}5$")
ax2.fill_between(np.arange(T), 0, 1, where=(pi_t >= pi_star), color=red, alpha=0.12)
ax2.set_ylabel(r"$\pi_t$")
ax2.set_ylim(-0.03, 1.03)
ax2.set_title(r"(b) Probabilité a posteriori de saut $\pi_t$ (filtrée, éq. Bayes)")
ax2.legend(loc="upper right", frameon=False, fontsize=8)

ax3 = axes[2]
ax3.plot(z_naive, color=navy, lw=1, label=r"$z_t$ (2.7, fenêtre 60j)")
ax3.axhline(kappa, color="black", ls=":", lw=1, label=r"$\pm\kappa=\pm2$")
ax3.axhline(-kappa, color="black", ls=":", lw=1)
ax3.axhline(0, color="gray", lw=0.6)
ax3.scatter(np.where(filtered_entry)[0], z_naive[filtered_entry], color=green, s=22, zorder=5)
ax3.scatter(np.where(blocked)[0], z_naive[blocked], color=red, marker="^", s=32, zorder=6)
ax3.set_ylabel(r"$z_t$")
ax3.set_xlabel(r"$t$ (jours)")
ax3.set_title(r"(c) z-score naïf : franchissements de seuil, conservés vs bloqués par le filtre")
ax3.legend(loc="upper right", frameon=False, fontsize=8)

for a in axes:
    for spine in ["top", "right"]:
        a.spines[spine].set_visible(False)

plt.tight_layout()
out_path = "/sessions/peaceful-stoic-ptolemy/mnt/outputs/fig_4_3_filtered_signal.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
print("saved:", out_path)
print("nb signaux candidats (regle naive 2.7):", int(np.nansum(candidate)))
print("nb entrees conservees par le filtre (4.3):", int(np.nansum(filtered_entry)))
print("nb signaux bloques par le filtre (4.3):", int(np.nansum(blocked)))
