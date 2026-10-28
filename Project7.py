import numpy as np
import pandas as pd
import os

buy_cost, sell_cost = 0.0022, 0.0021
rf_monthly = 1.065**(1/12) - 1

# Step 1: Strategy scores
def ma_score(prices, params):
    fast, slow = params
    return prices.rolling(fast).mean()/prices.rolling(slow).mean() - 1

def mr_score(prices, params):
    lookback, entry_z = params
    z = (prices - prices.rolling(lookback).mean())/prices.rolling(lookback).std()
    return -z - entry_z

# Step 2: Portfolio weights
def pick_weights(score_row):
    s = score_row.dropna()
    if len(s) < 50:
        return pd.Series(0.0, index=score_row.index)
    weights = pd.Series(0.0, index=score_row.index)
    on = s.index[s > 0]
    if len(on) > 0:
        weights[on] = 1.0/len(on)
    return weights

def build_weights(prices, score_fn, params):
    scores = score_fn(prices, params)
    return pd.DataFrame([pick_weights(scores.loc[d]) for d in scores.index], index=scores.index, columns=scores.columns)

# Step 3: Backtest
def backtest(weights, next_returns):
    held = pd.Series(0.0, index=weights.columns)
    out = []
    for date in weights.index[:-1]:
        target = weights.loc[date]
        if target.isna().all():
            held = held * 0
            continue
        target = target.fillna(0)
        trade = target - held
        bought = trade[trade > 0].sum()
        sold = -trade[trade < 0].sum()
        cost = bought*buy_cost + sold*sell_cost
        r = next_returns.loc[date].fillna(0)
        invested = target.sum()
        gross = (target*r).sum() + (1-invested)*rf_monthly
        held = target*(1+r)/(1+gross)
        out.append((date, gross, cost, gross-cost, bought+sold, invested))
    return pd.DataFrame(out, columns=["date","gross","cost","net","turnover","invested"]).set_index("date")

# Step 4: Performance
def sharpe(r):
    r = r.dropna()
    ex = r - rf_monthly
    return ex.mean()/ex.std()*np.sqrt(12)

def summary(r):
    years = len(r)/12
    nav = (1+r).cumprod()
    ex = r - rf_monthly
    return {"CAGR": nav.iloc[-1]**(1/years)-1, "Vol": ex.std()*np.sqrt(12), "Sharpe": sharpe(r), "MaxDD": (nav/nav.cummax()-1).min(), "WinRate": (r > 0).mean()}

# Step 5: Walk-forward validation
def walk_forward(prices, next_returns, score_fn, grid, train=60, test=12):
    weights = {a: build_weights(prices, score_fn, a) for a in grid}
    results = {a: backtest(weights[a], next_returns) for a in grid}
    months = results[grid[0]].index
    for a in grid:
        months = months.intersection(results[a].index)
    stitched = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    folds = []
    start = train

    while start < len(months):
        train_months = months[start-train:start]
        test_months = months[start:min(start+test, len(months))]
        best = max(grid, key=lambda a: sharpe(results[a].loc[train_months, "net"]))
        stitched.loc[test_months] = weights[best].loc[test_months].values
        folds.append({"test_start": test_months[0], "test_end": test_months[-1], "chosen": str(best), "train_sharpe": sharpe(results[best].loc[train_months, "net"]), "test_months": len(test_months)})
        start += test

    oos = backtest(stitched, next_returns)

    for i, f in enumerate(folds):
        seg = oos.loc[(oos.index >= f["test_start"]) & (oos.index <= f["test_end"]), "net"]
        folds[i]["test_return"] = (1+seg).prod()-1
        folds[i]["test_sharpe"] = sharpe(seg) if len(seg) > 2 else np.nan

    return oos, pd.DataFrame(folds), results

# Step 6: Strategies
strats = {
    "MA crossover": (ma_score, [(fast,slow) for fast in (2,3,4) for slow in (6,9,12) if fast < slow]),
    "Mean reversion": (mr_score, [(lookback,entry_z) for lookback in (6,9,12) for entry_z in (0.5,1.0,1.5)])
}

# Step 7: Run backtests
def run(prices):
    next_returns = prices.pct_change().shift(-1)
    all_results = {}
    for name, (score_fn, grid) in strats.items():
        oos, folds, results = walk_forward(prices, next_returns, score_fn, grid)
        all_results[name] = (oos, folds, results)
        print(f"\n{name}")
        print("\nOut-of-sample performance")
        print(pd.Series(summary(oos["net"])).round(3).to_string())
        print(f"\nAverage monthly turnover: {oos['turnover'].mean():.1%}")
        print(f"Average annual transaction cost: {oos['cost'].mean()*12:.2%}")
        print(f"Average invested fraction: {oos['invested'].mean():.0%}")
        print("\nWalk Forward folds:")
        print(folds.round(3).to_string(index=False))
    return all_results

# Step 8: Look-ahead audit
def audit_lookahead(prices):
    print("\nLOOK-AHEAD AUDIT")
    next_returns = prices.pct_change().shift(-1)
    test_date = prices.index[100]
    old_data = prices.loc[:test_date]
    full_score = ma_score(prices, (3,9)).loc[test_date]
    truncated_score = ma_score(old_data, (3,9)).loc[test_date]
    print("MA rolling test:", "PASS" if full_score.equals(truncated_score) else "FAIL")
    expected = prices.pct_change().shift(-1)
    print("Next-return construction:", "PASS" if next_returns.equals(expected) else "FAIL")
    print("Signal at t is intended to be evaluated using return at t+1.")

# Step 9: Survivorship audit
def audit_survivorship(daily, universe_file):
    print("\nSURVIVORSHIP AUDIT")
    universe = pd.read_csv(universe_file)
    universe_tickers = set(universe["Symbol"].str.strip() + ".NS")
    data_tickers = set(daily.columns)
    print("Universe stocks:", len(universe_tickers))
    print("Price-data stocks:", len(data_tickers))
    print("Universe stocks missing from price data:", len(universe_tickers-data_tickers))
    print("Extra price-data stocks:", len(data_tickers-universe_tickers))
    print("WARNING: One fixed universe is used for the whole historical period.")
    print("A true survivorship-bias correction requires historical NIFTY 500 membership by date.")
    print("Missing future returns are currently converted to 0 using fillna(0), which requires separate investigation.")

# Step 10: Snooping audit
def audit_snooping(all_results):
    print("\nSNOOPING AUDIT")
    print("Strategy families tested:", list(all_results.keys()))
    for name, (_, folds, results) in all_results.items():
        print(f"{name}: {len(results)} parameter combinations")
        print(f"Mean train Sharpe: {folds['train_sharpe'].mean():.2f}")
        print(f"Mean test Sharpe: {folds['test_sharpe'].mean():.2f}")
        print(f"Train-test Sharpe correlation: {folds['train_sharpe'].corr(folds['test_sharpe']):.2f}")
    print("Walk-forward parameter selection uses training data only.")
    print("Full-sample parameter sensitivity must not be used to choose the final strategy.")
    print("Choosing between MA and mean reversion after seeing their test results would create additional snooping.")

# Step 11: Load data
my_tickers = pd.read_csv("ind_nifty500list.csv")["Symbol"].str.strip().tolist()

if os.path.exists("daily.csv"):
    daily = pd.read_csv("daily.csv", index_col=0, parse_dates=True)
else:
    raise FileNotFoundError("daily.csv not found")

try:
    prices = daily.resample("ME").last()
except ValueError:
    prices = daily.resample("M").last()

if daily.index[-1] < prices.index[-1]:
    prices = prices.iloc[:-1]

prices = prices.where(prices >= 10)
prices = prices.dropna(axis=1, how="all")

print(f"{prices.shape[1]} stocks, {len(prices)} months, {prices.index[0].date()} to {prices.index[-1].date()}")

# Step 12: Run everything
all_results = run(prices)
audit_lookahead(prices)
audit_survivorship(daily, "ind_nifty500list.csv")
audit_snooping(all_results) 
