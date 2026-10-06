import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
import os
from scipy.optimize import minimize
import cvxpy as cp

file_path = "portfolio-optimization/stock_data.csv"
index_holding_path = "portfolio-optimization/SPY_holdings/"
historical_data_path = "portfolio-optimization/stock_history/"
test_weights = "portfolio-optimization/test_weights.csv"


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
def rf_minimise(mu, sigma, w_old, rf, risk_aversion=2.0, cost_pct=cost_pct, max_weight=0.05):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    w_old = np.asarray(w_old, dtype=np.float64)
    
    N = len(mu)

    sigma = (sigma + sigma.T) / 2.0
    
    w = cp.Variable(N, nonneg=True)
    w_rf= cp.Variable(nonneg=True)

    #sum of portfolios expected return
    portfolio_return = w @ mu + w_rf * rf
    #compute total portfolio risk, by calculating each assets marginal contribution
    portfolio_variance = cp.quad_form(w, cp.psd_wrap(sigma))
    #total cost of transaction fees
    total_costs = cost_pct * cp.sum(w - w_old)

    
    utility = portfolio_return - (0.5*risk_aversion * portfolio_variance) - total_costs

    #we must have all capital invested at a given time
    constraints = [
        cp.sum(w) + w_rf <= 1.0,
        w <= max_weight
    ]

    problem = cp.Problem(cp.Maximize(utility), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    return w.value, w_rf.value


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

    raw_annual_returns = returns.mean()*252
    universe_mean = raw_annual_returns.mean()

    shrinkage_weight = 0.30
    mean_returns = (shrinkage_weight*raw_annual_returns) + ((1-shrinkage_weight)*universe_mean)

    lw=LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_*252

    return mean_returns, cov_matrix

def simulate_year(risky, risk_free, rfr, data, start_c=1):
    daily_change = data.pct_change()
    change_vec = np.dot(daily_change, risky)
    change_vec = change_vec[~np.isnan(change_vec)]

    risky_capital = sum(risky)*start_c
    rf_capital = risk_free*start_c

    total_rf_gain = rf_capital*rfr
    n_days = len(change_vec)
    daily_linear_rfr_gain = total_rf_gain/ n_days

    accum = [risky_capital] 
    for c in change_vec:
        accum.append(accum[-1]*(c+1))

    rf_accum = [rf_capital + (daily_linear_rfr_gain*i) for i in range(n_days+1)] 

    total_accum = [r + rf for r, rf in zip(accum, rf_accum)]
    
    return data.index, total_accum, accum, rf_accum, change_vec

#TODO: use previous and next dataframes of stock data to align the previous weight vector with that of the next years dataframe. i.e. add/remove items that now/no longer exist and reorder to match next.
#Note: use fact that weight vector and df are already sorted for last year.
#question: how to deal with assets that we had capital in that dont exist in the next timeframe? enswer: I think just erase the weight from existence
def transform_weights (last_weights, df_last : pd.DataFrame, df_new : pd.DataFrame, year: int= None):
    weight_indexes = np.asarray(last_weights).ravel()
    weight_last = pd.Series(weight_indexes, index=df_last.columns, name="weight")
    active_weights = weight_last[weight_last != 0]
    new_weights = pd.Series(0.0, index=df_new.columns, name="weight")
    common_tickers = active_weights.index.intersection(df_new.columns)
    new_weights.loc[common_tickers] = active_weights.loc[common_tickers]
    not_common_tickers = active_weights.index.difference(df_new.columns).to_list()
    if len(not_common_tickers) >0:
        print(not_common_tickers)
        new = yf.download(
            tickers=not_common_tickers, 
            start=f"{year}-01-01", 
            end=f"{year+2}-01-01", 
            progress=False,
            auto_adjust=True
        )
        new = new.get("Close", pd.DataFrame(index=new.index))
        new = new.replace(['Nan', 'nan', 'None'], np.nan)
        new.dropna(axis=1, how='any', inplace=True)
        df_new_upd = df_new.join(new)
        df_new_upd.to_csv(f"{historical_data_path}{year}.csv")
        return transform_weights(last_weights, df_last, df_new_upd, year)
    return new_weights.to_numpy(), df_new

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
            progress=False,
            auto_adjust=True
        )
        prices = df.get("Close", pd.DataFrame(index=df.index))
        prices = prices.replace(['Nan', 'nan', 'None'], np.nan)
        prices.dropna(axis=1, how='any', inplace=True)
        prices.to_csv(f"{historical_data_path}{year}.csv")
        return prices
    else:
        df = pd.read_csv(f"{historical_data_path}{year}.csv", index_col=0, parse_dates=True)
        return df


rw, rw_last = [], []
rfw=0
rfw_last = 0
eps = 1e-7

total = []
risky = []
risk_free = []
returns = []
dates = []
years = 25

spy = yf.download(tickers=["SPY"], 
            start=f"2001-01-01", 
            end=f"{years+2001}-01-01",
            progress=False,
            auto_adjust=True)

x = np.array([0,0,1,2,0,4])

for i in range(years):
    progress_string= "*"*(i+1)+"-"*(years-(i+1))
    print(progress_string)
    year = 2000+i
    df=fetch_data(year)
    if i>0:
        df_last = fetch_data(year-1)
        #takes weights generated from last rebalance and fits it to a vector compatible with nexy years assets.
        rw, df = transform_weights(rw, df_last, df, year)

    #prep sample data
    mu, sigma = calc_help_sim(df.loc[f'{year}']) #take first half of data 
    
    if len(rw)==0:
        rw = [0]*len(df.columns)

    rw_last = rw.copy()
    rfw_last = rfw

    #optimise
    rw, rfw = rf_minimise(mu, sigma, w_old=rw, rf=risk_free_rate, risk_aversion=6.0, max_weight=0.05)

    #clean
    rw[np.abs(rw)<eps] = 0 #modifies entries in place to zero if their value is less than 0.0000001
    #calculate total loss due to transactions
    change = (sum(abs(rw_last - rw)) + abs(rfw_last-rfw))*cost_pct

    try:
        start_c=total[-1]
    except:
        start_c=1
    #remove costs from total capital
    start_c -= change
    #simulate for proceeding year
    #could compute return as one value, but I want to graph it.
    date, accumulated, risky_only, rf_only, return_vec = simulate_year(rw, rfw,risk_free_rate, df.loc[f'{year+1}'], start_c) #use second half of dataset for simulation
    #extend vectors for graphing
    dates.extend(date)
    total.extend(accumulated)
    risky.extend(risky_only)
    returns.extend(return_vec)
    risk_free.extend(rf_only)

def compute_sharpe(return_vec, trading_days = 252):
    returns = np.asarray(return_vec)
    daily_rf = risk_free_rate/trading_days

    excess_returns = returns - daily_rf

    mean_excess_returns = np.mean(excess_returns)
    daily_std = np.std(excess_returns, ddof=1)

    if daily_std == 0:
        return 0.0

    annualized_sharpe = (mean_excess_returns / daily_std) * np.sqrt(trading_days)

    return annualized_sharpe

def cagr(total_returns):
    start = total_returns[0]
    end = total_returns[-1]

    cagr = ((end/start)**(1/years)-1)*100

    return cagr
    
def max_d(accumulated_returns):
    max_point = accumulated_returns[0]
    max_diff = 1

    for v in accumulated_returns:
        if v > max_point:
            max_point = v

        elif v/max_point < max_diff:
            max_diff = v/max_point

    return (1-max_diff)

spy_pct = spy.get("Close", pd.DataFrame(index=spy.index)).pct_change().to_numpy()
spy_pct = spy_pct[~np.isnan(spy_pct)]
spy_accum = [0.95]

for c in spy_pct:
    spy_accum.append(spy_accum[-1]*(1+c))

s_r = compute_sharpe(returns)
op_cagr = cagr(total)
max_drawdown = max_d(total)
print(f"Sharpe Ratio: {s_r}. \nCompound Annual Growth Rate: {round(op_cagr, 5)} %\nMax Drawdown: {round(max_drawdown*100, 5)} %")
spy_s_r = compute_sharpe(spy_pct)
print(f"SPY Sharpe ratio: {spy_s_r}")

fig = plt.figure(figsize=(14,8))
tot_line, =plt.plot(dates, total, label="optimised portfolio returns")
risk_line, =plt.plot(dates, risky, label="risky assets returns")
rf_line, =plt.plot(dates, risk_free, label="risk free returns")
spy_line, =plt.plot(dates, spy_accum, label="SPY returns")
leg = plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", borderaxespad=0.0, fancybox=True)

lines = [tot_line, risk_line, rf_line, spy_line]
lined = {}
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(True)  # Enable picking on the legend line.
    lined[legline] = origline

def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legline = event.artist
    origline = lined[legline]
    visible = not origline.get_visible()
    origline.set_visible(visible)
    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    legline.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', on_pick)
plt.grid()
plt.tight_layout()
plt.ylabel("alpha")
plt.xlabel("days")
plt.show()
    
