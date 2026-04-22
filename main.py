import argparse
import yfinance as yf
import numpy as np
import pandas as pd
from termcolor import colored, cprint
from dataclasses import dataclass

from constants import *
from models import *
from analytics import *

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

def get_tickers_and_weights_from_file(filepath):
    raw_weights = {}
    tickers = []
    investment = 0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # skip blanks and comments
                continue
            parts = line.split()
            if parts[0] == "investment":
                investment = int(parts[2])
                continue
            if len(parts) != 2:
                raise ValueError(f"Invalid line format: '{line}' — expected: TICKER WEIGHT")
            ticker, weight = parts[0].upper(), float(parts[1])
            tickers.append(ticker)
            raw_weights[ticker] = weight

    weights = pd.Series(raw_weights)
    return tickers, weights, investment

def round2(val):
    return round(val, 2)

def to_percent(val):
    return round(val * 100, 2)

    
def create_portfolio_object(name, tickers, weights, investment):
    data = download_price_data(tickers)
    prices = get_adjusted_close(data)
    returns = get_daily_returns(prices)
    weighted_daily_returns = get_weighted_daily_returns(returns, weights)
    final_value = calculate_final_value(investment, weighted_daily_returns)
    periods = len(returns)
    annualized_return = annualize_return(final_value, investment, periods)
    return_on_investment = round2(calculate_return_on_investment(final_value, investment))
    volatility = get_portfolio_volatility(returns, weights)
    annual_volatility = annualize_volatility(volatility)
    risk_adjusted_return = calculate_sharpe_ratio(annualized_return, annual_volatility)
    return Portfolio(name, tickers, investment, data, returns, weighted_daily_returns, annualized_return, volatility, annual_volatility, risk_adjusted_return, return_on_investment, final_value)

def print_portfolio_summary(portfolio):
    print(portfolio.name, ":\n", sep="")
    print("Tickers:", portfolio.tickers)
    print("Final Value:", f"${round2(portfolio.final_value)}")
    print("Return on Investment (ROI):", f"{to_percent(portfolio.return_on_investment)}%")
    print("Volatility:", f"{to_percent(portfolio.annual_volatility)}%")
    print("Risk Adjusted Return:", f"{round2(portfolio.risk_adjusted_return)}")
    print("\n", "-"*50, "\n", sep="")

def get_comparison_indicator(result, prefer_negative=False):
    if result > 0:
        return colored(result, 'green') if not prefer_negative else colored(result, "red")
    elif result < 0:
        return colored(result, 'red') if not prefer_negative else colored(result, "green")
    else:
        return colored(result, "dark_grey")

def print_portfolio_differences(portfolio1, portfolio2):
    print(f"Portfolio Differences: ({portfolio1.name} vs. {portfolio2.name})\n")

    result = round2(portfolio1.final_value - portfolio2.final_value)
    result_text = get_comparison_indicator(result)
    print(f"Final Price: ${round2(portfolio1.final_value)} - ${round2(portfolio2.final_value)} = ${result_text}")
    
    result = to_percent(portfolio1.return_on_investment - portfolio2.return_on_investment)
    result_text = get_comparison_indicator(result)
    print(f"Return on Investment (ROI): {to_percent(portfolio1.return_on_investment)}% - {to_percent(portfolio2.return_on_investment)}% = {result_text}%")
    
    result = to_percent(portfolio1.annual_volatility - portfolio2.annual_volatility)
    result_text = get_comparison_indicator(result, prefer_negative=True)
    print(f"Volatility: {to_percent(portfolio1.annual_volatility)}% - {to_percent(portfolio2.annual_volatility)}% = {result_text}%")
    
    result = round2(portfolio1.risk_adjusted_return - portfolio2.risk_adjusted_return)
    result_text = get_comparison_indicator(result)
    print(f"Risk Adjusted Return: {round2(portfolio1.risk_adjusted_return)} - {round2(portfolio2.risk_adjusted_return)} = {result_text}")

def main():
    parser = argparse.ArgumentParser(description="Portfolio analyzer")
    parser.add_argument("-f", "--from-file", metavar="FILE", help="Path to file with tickers and weights")
    args = parser.parse_args()
    
    tickers = []
    investment = 0
    
    # Get user input for ticker symbols and weights, or from file if specified
    if args.from_file:
        tickers, weights, investment = get_tickers_and_weights_from_file(args.from_file)
    else:
        tickers, weights = get_tickers_and_weights()
    
    if investment == 0:
        investment = int(input("\nHow much to invest?\nAmount: ")) # amount invested in USD

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

    print_portfolio_differences(portfolio, sp500_portfolio)

if __name__ == "__main__":
    main()
