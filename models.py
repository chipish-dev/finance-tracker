from dataclasses import dataclass

@dataclass
class Portfolio:
    name: str
    tickers: list[str]
    investment: float
    data: pd.DataFrame
    returns: pd.DataFrame
    weighted_daily_returns: pd.Series
    annualized_return: float
    volatility: float
    annual_volatility: float
    risk_adjusted_return: float
    return_on_investment: float
    final_value: float
