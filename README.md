# Du modele a la strategie : un cadre jump-diffusion pour le pairs trading

Summer project explorant le pairs trading (cointegration, spread, z-score, filtre de Kalman) couple a une dynamique jump-diffusion a la Merton (1976), estimee par maximum de vraisemblance (Ball & Torous, 1985) pour filtrer les signaux de trading en fonction de la probabilite a posteriori qu'un mouvement du spread soit un saut plutot qu'un exces mean-reverting exploitable.

## Contenu du depot

- `report.tex` - le rapport complet (LaTeX). Compile avec pdflatex (deux passes) ; `graphes/` doit rester a cote du `.tex`.
- - `graphes/` - toutes les figures du rapport.
  - - `code/` - scripts Python utilises pour la partie empirique et la generation des figures.
   
    - ## Compiler le rapport
   
    - ```
      pdflatex report.tex
      pdflatex report.tex
      ```
      
