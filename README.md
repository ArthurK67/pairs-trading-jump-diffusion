# Du modèle à la stratégie : un cadre jump-diffusion pour le pairs trading

Summer project explorant le pairs trading (cointégration, spread, z-score, filtre de Kalman) couplé à une dynamique jump-diffusion à la Merton (1976), estimée par maximum de vraisemblance (Ball & Torous, 1985) pour filtrer les signaux de trading en fonction de la probabilité a posteriori qu'un mouvement du spread soit un saut plutôt qu'un excès mean-reverting exploitable.

## Contenu du dépôt

- `report.tex` — le rapport complet (LaTeX). Compile avec `pdflatex` (deux passes) ; `graphes/` doit rester à côté du `.tex` (`\graphicspath{{graphes/}}`).
- `graphes/` — toutes les figures du rapport (spread simulé, densités de mélange, courbes d'équité, heatmap de sensibilité, etc.).
- `code/` — scripts Python utilisés pour la partie empirique et la génération des figures :
  - `pair_trading_real_data.py` — cointégration, spread, z-score sur données réelles (MA/V, PEP/KO). Génère `spread_zscore_MA_V.png`, `spread_zscore_PEP_KO.png`, `residuals_mixture_MA_V.png`, `residuals_mixture_PEP_KO.png`.
  - `make_fig_4_2.py` — simulation OU à sauts + estimation MLE de mélange (figure 4.2). Génère `fig_4_2_mixture.png`.
  - `make_fig_4_3.py` — règle de décision filtrée par la probabilité a posteriori de saut (figure 4.3). Génère `fig_4_3_filtered_signal.png`.
  - `backtest_4_6.py` — backtest comparatif stratégie naïve vs. filtrée (section 4.6). Génère `fig_4_6_equity.png`.
  - `robustness_4_7.py` — grille de sensibilité (κ, π*) et tests de robustesse (section 4.7). Génère `fig_4_7_heatmap.png` et `fig_4_6_equity_corrige.png` (courbe d'équité avec stratégie corrigée, produite dans ce script malgré son nom).

## Compiler le rapport

```bash
pdflatex report.tex
pdflatex report.tex   # deuxième passe pour les renvois/figures
```
