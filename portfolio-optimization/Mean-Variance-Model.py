import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

file_path = "portfolio-optimization/stock_data.csv"

tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "SPCX", "AVGO", "META", "TSLA", "MU", "AMD", "WMT", "ASML", "INTC", "CSCO", "PLTR", "COST", "LRCX", "AMAT", "NFLX", "PANW", "ARM", "SNDK", "TXN", "KLAC", "MRVL", "LIN", "AMGN", "LLY", "JPM", "V", "XOM", "JNJ", "ABBV", "ORCL", "CVX", "BAC", "KO", "CAT", "MRK"]

#NasdaqTop30 ∪ S&P500Top30 
if not os.path.exists(file_path):
    df = yf.Tickers(" ".join(tickers)).history(period="1y")["Close"]
    df.to_csv(file_path)
else:
    df = pd.read_csv(file_path, header=[0,1], index_col=0)

returns = df.pct_change().dropna()
mean_returns = returns.mean()*252
cov_matrix = returns.cov()*252
n_assets = len(tickers)

def portfolio_variance(weights):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

target_return=0.05
step = 0.01

def minimise(target_return):
    constraints = (
        #total return = R(target return)
        {'type':'eq','fun':lambda w: np.dot(w, mean_returns) - target_return},
        #total capital allocation = 1
        {'type': 'eq','fun':lambda w: np.sum(w)-1.0}
    )

    #no short selling. i.e 0<=wi<=1
    bounds = tuple((0,1) for _ in range(n_assets))

    init_guess = np.repeat(1/ n_assets, n_assets)

    result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result

returns = []
volatility = []
last = 0
opt_return=target_return #set default >0 will change after first loop anyway
while (opt_return>=last and target_return<=0.35):
    result = minimise(target_return)
    opt_weights = result.x
    opt_return = np.dot(opt_weights, mean_returns)
    opt_volatility = np.sqrt(result.fun)
    returns.append(opt_return)
    volatility.append(opt_volatility)
    last=opt_return
    target_return+=step

plt.plot(volatility, returns)

plt.show()
