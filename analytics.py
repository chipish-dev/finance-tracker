import yfinance as yf
import pandas as pd
import numpy as np

from constants import *

def download_price_data(tickers):
    return yf.download(tickers, auto_adjust=False, period="1y")

def get_adjusted_close(data):
    return data["Adj Close"].dropna()

def get_daily_returns(prices):
    return prices.pct_change().dropna() 

def get_weighted_daily_returns(returns, weights):
    return (returns * weights).sum(axis=1)

def calculate_final_value(investment, weighted_daily_returns):
    daily_growth_factor = 1 + weighted_daily_returns
    cumulative_growth_factor = daily_growth_factor.cumprod()

    portfolio_value = investment * cumulative_growth_factor
    return portfolio_value.iloc[-1]

def calculate_return_on_investment(final_value, investment):
    return (final_value - investment)/investment

def get_portfolio_volatility(returns, weights):
    cov_matrix = returns.cov()
    return np.sqrt(weights.T @ cov_matrix @ weights)

def annualize_volatility(volatility):
    return volatility * np.sqrt(252)

def annualize_return(final_value, investment, periods=252):
    total_return = final_value / investment
    return total_return ** (252 / periods) - 1

def calculate_sharpe_ratio(annualized_return, annual_volatility):
    return (annualized_return - RISK_FREE_RATE)/annual_volatility
