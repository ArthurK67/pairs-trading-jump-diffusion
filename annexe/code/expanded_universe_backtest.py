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

