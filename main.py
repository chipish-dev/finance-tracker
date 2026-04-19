import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass

RISK_FREE_RATE = 0.02

@dataclass
class Portfolio:
    name: str
    investment: float
    data: pd.DataFrame
    returns: pd.DataFrame
    weighted_daily_returns: pd.Series
    annualized_return: float
    volatility: float
    annual_volatility: float
    risk_adjusted_return: float
    final_value: float

def get_tickers_and_weights():
    tickers = []
    raw_weights = {}

    while True:
        ticker_symbol = input("\nAdd Ticker (q to quit)\nTicker Symbol: ").strip()
        if ticker_symbol != 'q':
            test_ticker = yf.Ticker(ticker_symbol)
            if len(test_ticker.info) > 1 and 'regularMarketPrice' in test_ticker.info:
                tickers.append(ticker_symbol)
                
                while True:
                    try:
                        weight = float(input(f"\nWeight for {ticker_symbol} (e.g. 0.5 for 50%): ").strip())
                        if weight >= 0 and weight <= 1:
                            break
                    except ValueError:
                        print("Please enter a valid number (0 <= x <= 1).\n")
                
                tickers.append(ticker_symbol)
                raw_weights[ticker_symbol] = weight

            else:
                print("Invalid Ticker Symbol.\n")
        else:
            break
    
    weights = pd.Series(raw_weights)
    return tickers, weights
    
def round2(val):
    return round(val, 2)

def to_percent(val):
    return round(val * 100, 2)

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
    
def create_portfolio_object(name, tickers, weights, investment):
    data = download_price_data(tickers)
    prices = get_adjusted_close(data)
    returns = get_daily_returns(prices)
    weighted_daily_returns = get_weighted_daily_returns(returns, weights)
    final_value = calculate_final_value(investment, weighted_daily_returns)
    periods = len(returns)
    annualized_return = annualize_return(final_value, investment, periods)
    volatility = get_portfolio_volatility(returns, weights)
    annual_volatility = annualize_volatility(volatility)
    risk_adjusted_return = calculate_sharpe_ratio(annualized_return, annual_volatility)
    return Portfolio(name, investment, data, returns, weighted_daily_returns, annualized_return, volatility, annual_volatility, risk_adjusted_return, final_value)

def print_portfolio_summary(portfolio):
    print(portfolio.name, ":", sep="")
    print("Final Value: $", round2(portfolio.final_value), sep="")
    print("Return on Investment (ROI): ", round2((portfolio.final_value - portfolio.investment)/portfolio.investment * 100), "%", sep="")
    print("Volatility: ", to_percent(portfolio.annual_volatility), "%", sep="")
    print("Risk Adjusted Return: ", round2(portfolio.risk_adjusted_return), sep="")
    print("\n", "-"*50, "\n", sep="")

def main():
    investment = int(input("How much to invest?\nAmount: ")) # amount invested in USD
    tickers = []

    # Get user input for ticker symbols and weights
    tickers, weights = get_tickers_and_weights()

    if tickers == []:
        print("No tickers listed.")
        return

    sp500_weight = pd.Series({"^GSPC": 1.0})

    # match weights via pandas to ensure weights are assigned correctly regardless of ticker order.

    # sanity check for weights to make sure they add up to 100% of the investment amount.
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")

    portfolio = create_portfolio_object("Your Portfolio", tickers, weights, investment)
    sp500_portfolio = create_portfolio_object("S&P500", "^GSPC", sp500_weight, investment)

    print("Initial Investment: $", investment, "\n", sep="")

    print_portfolio_summary(portfolio)

    print_portfolio_summary(sp500_portfolio)

    print("\nDifference: $", round2(portfolio.final_value - sp500_portfolio.final_value), sep="")

if __name__ == "__main__":
    main()
