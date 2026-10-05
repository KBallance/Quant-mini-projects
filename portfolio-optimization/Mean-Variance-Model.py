import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import os
from scipy.optimize import minimize

file_path = "portfolio-optimization/stock_data.csv"
index_holding_path = "portfolio-optimization/SPY_holdings/"
historical_data_path = "portfolio-optimization/stock_history/"


# tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "SPCX", "AVGO", "META", "TSLA", "MU", "AMD", "WMT", "ASML", "INTC", "CSCO", "PLTR", "COST", "LRCX", "AMAT", "NFLX", "PANW", "ARM", "SNDK", "TXN", "KLAC", "MRVL", "LIN", "AMGN", "LLY", "JPM", "V", "XOM", "JNJ", "ABBV", "ORCL", "CVX", "BAC", "KO", "CAT", "MRK"]


# #NasdaqTop30 ∪ S&P500Top30 
# if not os.path.exists(file_path):
#     df = yf.Tickers(" ".join(tickers)).history(period="1y")["Close"]
#     df.to_csv(file_path)
# else:
#     df = pd.read_csv(file_path, header=[0,1], index_col=0)

def calc_help(df):
    returns = df.pct_change().dropna()

    raw_annual_returns = returns.mean()* 252
    universe_mean = raw_annual_returns.mean()

    shrinkage_weight = 0.30
    mean_returns = (shrinkage_weight*raw_annual_returns)+(1-shrinkage_weight)*universe_mean

    lw=LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_*252

    return mean_returns, cov_matrix

def risky_minimise(mu, sigma, w_old, risk_aversion=2.0, cost_pct=0.005):
    N=len(mu)

    def objective(x):
        w= x[:N]
        u= x[N : 2*N]
        v= x[2*N :]

        portfolio_return = np.dot(w, mu)
        portfolio_variance = np.dot(w.T, np.dot(sigma, w))
        total_costs = np.sum(cost_pct * (u+v))

        utility = portfolio_return - (0.5*portfolio_variance*risk_aversion) - total_costs
        return -utility

    init_guess = np.concatenate([w_old, np.zeros(N), np.zeros(N)])

    constraints = [
        #total capital allocation = 1
        {'type': 'eq','fun':lambda x: np.sum(x[:N])-1.0}
    ]

    for i in range(N):
            constraints.append({
                'type':'eq',
                'fun' : lambda x, index=i: x[index] - w_old[index] - (x[N+index] - x[2*N+index])
            })

    #no short selling. i.e 0<=wi<=1
    bounds = [(0,1)]*(3*N)

    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result

cost_pct = 0.005

#calculate optimal portfolio given access to a risk free asset
def rf_minimise(mu, sigma, w_old, rf, risk_aversion=2.0, cost_pct=cost_pct):
    
    N = len(mu)
    def objective(x):
        #0-(n-1)th items are risky assets
        w= x[:N]
        #n-th item is the risk free asset
        w_rf= x[N]
    
        u= x[N+1 : 2*N+1]
        v= x[2*N+1 :]

        #sum of portfolios expected return
        portfolio_return = np.dot(w, mu) + w_rf * rf
        #compute total portfolio risk, by calculating each assets marginal contribution
        portfolio_variance = np.dot(w.T, np.dot(sigma, w))
        #total cost of transaction fees
        total_costs = np.sum(cost_pct * (u+v))

        #function we want to minimise (or maximise really). expected return - risk - transaction fees
        utility = portfolio_return - (0.5*risk_aversion * portfolio_variance) - total_costs
        return -utility

    #initial weights is previous weights plus 0 changes
    init_guess = np.concatenate([w_old, [1.0-np.sum(w_old)], np.zeros(N), np.zeros(N)])

    #we must have all capital invested at a given time
    constraints = [
        {'type':'eq','fun':lambda x: np.sum(x[:N+1])-1.0}
    ]

    for i in range(N):
        constraints.append({
            'type':'eq',
            'fun' : lambda x, index=i: x[index] - w_old[index] - (x[N+1+index] - x[2*N+1+index])
        })

    #0<=wi<=1. i.e no short selling
    bounds = [(0,1)]*(3*N+1)

    #our minimized portfolio
    result = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"Optimisation failed:  {result.message}")

    #weights of risky assets
    optimised_risky_weights = result.x[:N]
    #weight of risk free asset
    optimised_rf_weight = result.x[N]

    return (optimised_risky_weights, optimised_rf_weight)


# #start of efficient frontier plot
# rf_rate = 0.04

# N = len(tickers)

# rf_return_vec=[]
# rf_risk_vec=[]

# risky_return_vec=[]
# risky_risk_vec=[]

# current_guess = np.zeros(len(tickers))

# mean_returns, cov_matrix = calc_help(df)

# lambdas = np.logspace(np.log10(0.01), np.log10(500), num=50)
# for l in lambdas:
#     (r, rf) = rf_minimise(mean_returns, cov_matrix, risk_aversion=l, w_old=current_guess, rf=rf_rate)
#     current_guess=r
#     rf_return_vec.append(np.dot(mean_returns, r)+rf*rf_rate)
#     rf_risk_vec.append(np.sqrt(np.dot(r, np.dot(cov_matrix, r))))

# current_guess = [1/len(tickers)]*len(tickers)

# for l in lambdas:
#     result = risky_minimise(mean_returns, cov_matrix, risk_aversion=l, w_old=current_guess)
#     if result.success:
#         current_guess=result.x[:N]
#     risky_return_vec.append(np.dot(mean_returns, result.x[:N])+result.x[N]*rf_rate)
#     risky_risk_vec.append(np.sqrt(np.dot(result.x[:N], np.dot(cov_matrix, result.x[:N]))))

# # prep data
# risks_rf = np.array(rf_risk_vec)
# returns_rf = np.array(rf_return_vec)

# risks_risky = np.array(risky_risk_vec)
# returns_risky = np.array(risky_return_vec)

# idx_rf = np.argsort(risks_rf)
# risks_rf, returns_rf = risks_rf[idx_rf], returns_rf[idx_rf]

# idx_risky = np.argsort(risks_risky)
# risks_risky, returns_risky = risks_risky[idx_risky], returns_risky[idx_risky]


# #plot graphs
# plt.figure(figsize=(8,5))
# plt.plot(risks_rf, returns_rf, label = "risk-free asset available")
# plt.plot(risks_risky, returns_risky, label = "only risky assets")

# plt.xlabel('portfolio Risk (Volatility)')
# plt.ylabel("Expected Returns")
# plt.legend()
# plt.grid(True)
# plt.show()

#end of efficient frontier plot

#start of simulation

Cc,Cr,Crf = 1.0,1.0,1.0

rr, rrf, cr = [], [], []

risk_free_rate=0.04

def calc_help_sim(df):
    returns = df.pct_change().dropna()
    trading_days = (len(returns)+1)

    raw_annual_returns = returns.mean()*(trading_days)
    universe_mean = raw_annual_returns.mean()

    shrinkage_weight = 0.30
    mean_returns = (shrinkage_weight*raw_annual_returns)+(1-shrinkage_weight)*universe_mean

    lw=LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_*(trading_days)

    return mean_returns, cov_matrix

def simulate_year(risky, risk_free, rfr, data, start_c=1):
    daily_change = data.pct_change()
    total_rf_gain = risk_free*rfr #total gains from capital allocated to risk free assets. not compounding.
    daily_linear_rfr_gain = total_rf_gain/(len(data))
    change_vec = np.dot(daily_change, risky)
    change_vec = change_vec[~np.isnan(change_vec)]
    accum = [sum(risky) * start_c] #percentage of starting capital allocated to risky assets (scales according to starting capital)
    #calculate accumulated alpha daily
    for c in change_vec:
        next = accum[-1]*(1+c)
        accum.append(next)

    #adds risk free gains as linear change that unnafects daily changes through risky investments as in practice only get gains yearly when investment matures.
    rf_accum = [risk_free*start_c] #scale rf weight according to starting capital
    total_accum = accum.copy()
    total_accum[0]+=risk_free
    for i in range(len(change_vec)):
        rf_accum.append(risk_free+daily_linear_rfr_gain*i)
        total_accum[i+1]+=(risk_free+daily_linear_rfr_gain*i)
    
    return change_vec, total_accum, accum, rf_accum

#fetch data for stocks for which data exists for 2 years (from desired start). dont compare agaisnt what next years (SYP) holdings looks like as that garantees at least partial success of the asset     
def fetch_data(year):
    if not os.path.exists(f"{historical_data_path}{year}.csv"):
        #attempt to download historical data using index holdings
        holdings = pd.read_csv(f"{index_holding_path}{year}.csv")
        tickers = holdings["Ticker"].astype(str).str.strip().str.replace('.', '-').tolist()
        df=yf.download(
            tickers=tickers, 
            start=f"{year}-01-01", 
            end=f"{year+2}-01-01", 
            progress=False
        )
        adj_close = df.get("Adj Close", pd.DataFrame(index=df.index))
        close = df.get("Close", pd.DataFrame(index=df.index))
        prices = adj_close.combine_first(close)
        prices = prices.replace(['NaN', 'nan', 'None'], np.nan)
        prices.dropna(axis=1, how='any', inplace=True)
        prices.to_csv(f"{historical_data_path}{year}.csv")
        return prices
    else:
        df = pd.read_csv(f"{historical_data_path}{year}.csv", index_col=0, parse_dates=True)
        return df


rw, rw_last = [], []
rf_last = 0
eps = 1e-7

total = []
risky = []
risk_free = []
years = 2

spy = yf.download(tickers=["SPY"], 
            start=f"2001-01-01", 
            end=f"200{years+1}-01-01",
            progress=False)

for i in range(years):
    progress_string= "*"*(i+1)+"-"*(years-(i+1))
    print(progress_string)
    year = 2000+i
    df=fetch_data(year)

    mu, sigma = calc_help_sim(df[:252]) #take first half of data 
    
    if len(rw)==0:
        rw = [0]*len(df.columns)

    rw_last = rw.copy()
    rfw_last = rfw

    rw, rfw = rf_minimise(mu, sigma, w_old=rw, rf=risk_free_rate, risk_aversion=30.0)

    rw[np.abs(rw)<eps] = 0 #modifies entries in place to zero if their value is less than 0.0000001

    change = (sum(abs(rw_last - rw)) + abs(rfw_last-rfw))*cost_pct
    print(change)

    try:
        start_c=total[-1]
    except:
        start_c=1

    start_c -= change
    print(start_c)

    change, accumulated, risky_only, rf_only = simulate_year(rw, rfw,risk_free_rate, df[252:], start_c) #use second half of dataset for simulation

    total.extend(accumulated)
    risky.extend(risky_only)
    risk_free.extend(rf_only)


spy_pct = spy.get("Close", pd.DataFrame(index=spy.index)).pct_change().to_numpy()
spy_pct = spy_pct[~np.isnan(spy_pct)]
spy_accum = [1]

for c in spy_pct:
    next = spy_accum[-1]*(1+c)
    spy_accum.append(next)

print(f"risk free weight: {rfw}")
print(f"risky weights: {rw}")
    
plt.figure(figsize=(14,10))
plt.plot(accumulated, label="optimised portfolio returns")
plt.plot(risky_only, label="risky assets returns")
plt.plot(rf_only, label="risk free returns")
plt.plot(spy_accum, label="SPY returns")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", borderaxespad=0.0)
plt.grid()
plt.tight_layout()
plt.ylabel("alpha")
plt.xlabel("days")
plt.show()
    
