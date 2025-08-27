"""
Utility functions for the AI Stock Analysis Bot
Contains helper functions for formatting, validation, and common operations
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

def format_currency(value, currency='USD'):
    """
    Format a number as currency
    
    Args:
        value (float): The value to format
        currency (str): Currency code (default: USD)
        
    Returns:
        str: Formatted currency string
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if abs(value) >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
            
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value, decimal_places=2):
    """
    Format a number as percentage
    
    Args:
        value (float): The value to format (0.1 = 10%)
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        return f"{value * 100:.{decimal_places}f}%"
        
    except (ValueError, TypeError):
        return "N/A"

def format_number(value, decimal_places=2):
    """
    Format a number with appropriate suffix
    
    Args:
        value (float): The value to format
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if abs(value) >= 1e12:
            return f"{value/1e12:.{decimal_places}f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.{decimal_places}f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{decimal_places}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{decimal_places}f}K"
        else:
            return f"{value:.{decimal_places}f}"
            
    except (ValueError, TypeError):
        return "N/A"

def format_market_cap(value):
    """
    Format market capitalization
    
    Args:
        value (float): Market cap value
        
    Returns:
        str: Formatted market cap string
    """
    return format_currency(value)

def validate_ticker(ticker):
    """
    Validate stock ticker symbol
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation - alphanumeric characters and periods/dashes
    ticker = ticker.strip().upper()
    
    # Check length (typically 1-5 characters for US stocks)
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    # Check for valid characters
    if not re.match(r'^[A-Z0-9.\-]+$', ticker):
        return False
    
    # Additional validation - try to fetch basic info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid info
        if not info or 'symbol' not in info:
            return False
        
        return True
        
    except Exception:
        return False

def validate_tickers(tickers):
    """
    Validate multiple ticker symbols
    
    Args:
        tickers (list): List of ticker symbols
        
    Returns:
        tuple: (valid_tickers, invalid_tickers)
    """
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(ticker.strip().upper())
        else:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def clean_ticker(ticker):
    """
    Clean and normalize ticker symbol
    
    Args:
        ticker (str): Raw ticker symbol
        
    Returns:
        str: Cleaned ticker symbol
    """
    if not ticker:
        return ""
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Remove common prefixes/suffixes that might be included
    ticker = re.sub(r'^\$', '', ticker)  # Remove $ prefix
    ticker = re.sub(r'\..*$', '', ticker)  # Remove exchange suffix if present
    
    return ticker

def calculate_returns(data, period='1d'):
    """
    Calculate returns for different periods
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        period (str): Period for returns calculation
        
    Returns:
        pd.Series: Returns series
    """
    try:
        if 'Close' not in data.columns:
            return pd.Series()
        
        if period == '1d':
            return data['Close'].pct_change()
        elif period == '1w':
            return data['Close'].pct_change(periods=5)
        elif period == '1m':
            return data['Close'].pct_change(periods=21)
        elif period == '3m':
            return data['Close'].pct_change(periods=63)
        elif period == '1y':
            return data['Close'].pct_change(periods=252)
        else:
            return data['Close'].pct_change()
            
    except Exception as e:
        logging.error(f"Error calculating returns: {str(e)}")
        return pd.Series()

def calculate_volatility(data, window=20):
    """
    Calculate rolling volatility
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Rolling window size
        
    Returns:
        pd.Series: Volatility series
    """
    try:
        if 'Close' not in data.columns:
            return pd.Series()
        
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        return volatility
        
    except Exception as e:
        logging.error(f"Error calculating volatility: {str(e)}")
        return pd.Series()

def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        risk_free_rate (float): Risk-free rate (annual)
        
    Returns:
        float: Sharpe ratio
    """
    try:
        if 'Close' not in data.columns or len(data) < 2:
            return 0
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        # Annualized return
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # Annualized volatility
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return sharpe_ratio
        
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0

def calculate_max_drawdown(data):
    """
    Calculate maximum drawdown
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        
    Returns:
        float: Maximum drawdown as percentage
    """
    try:
        if 'Close' not in data.columns or len(data) < 2:
            return 0
        
        prices = data['Close']
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return max_drawdown
        
    except Exception as e:
        logging.error(f"Error calculating max drawdown: {str(e)}")
        return 0

def get_trading_days(start_date, end_date):
    """
    Get number of trading days between two dates
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        int: Number of trading days
    """
    try:
        # Approximate trading days (252 per year)
        total_days = (end_date - start_date).days
        trading_days = int(total_days * (252 / 365))
        
        return trading_days
        
    except Exception as e:
        logging.error(f"Error calculating trading days: {str(e)}")
        return 0

def format_date(date_value, format_str='%Y-%m-%d'):
    """
    Format date value
    
    Args:
        date_value: Date value to format
        format_str (str): Format string
        
    Returns:
        str: Formatted date string
    """
    try:
        if pd.isna(date_value):
            return "N/A"
        
        if isinstance(date_value, str):
            # Try to parse string date
            date_value = pd.to_datetime(date_value)
        
        return date_value.strftime(format_str)
        
    except Exception as e:
        logging.error(f"Error formatting date: {str(e)}")
        return "N/A"

def get_business_days_ago(days_ago):
    """
    Get business day that was N days ago
    
    Args:
        days_ago (int): Number of business days ago
        
    Returns:
        datetime: Business day
    """
    try:
        today = datetime.now()
        business_day = today
        
        count = 0
        while count < days_ago:
            business_day -= timedelta(days=1)
            # Skip weekends
            if business_day.weekday() < 5:  # 0-4 are Monday-Friday
                count += 1
        
        return business_day
        
    except Exception as e:
        logging.error(f"Error getting business days ago: {str(e)}")
        return datetime.now()

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division fails
        
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        
        return numerator / denominator
        
    except Exception:
        return default

def calculate_price_change(current_price, previous_price):
    """
    Calculate price change and percentage change
    
    Args:
        current_price (float): Current price
        previous_price (float): Previous price
        
    Returns:
        tuple: (absolute_change, percentage_change)
    """
    try:
        if pd.isna(current_price) or pd.isna(previous_price) or previous_price == 0:
            return 0, 0
        
        absolute_change = current_price - previous_price
        percentage_change = (absolute_change / previous_price) * 100
        
        return absolute_change, percentage_change
        
    except Exception as e:
        logging.error(f"Error calculating price change: {str(e)}")
        return 0, 0

def round_to_significant_digits(value, digits=3):
    """
    Round value to specified number of significant digits
    
    Args:
        value (float): Value to round
        digits (int): Number of significant digits
        
    Returns:
        float: Rounded value
    """
    try:
        if pd.isna(value) or value == 0:
            return value
        
        import math
        return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))
        
    except Exception as e:
        logging.error(f"Error rounding to significant digits: {str(e)}")
        return value

def create_error_message(error_type, ticker=None, details=None):
    """
    Create standardized error message
    
    Args:
        error_type (str): Type of error
        ticker (str): Stock ticker if applicable
        details (str): Additional error details
        
    Returns:
        str: Formatted error message
    """
    base_messages = {
        'invalid_ticker': f"Invalid ticker symbol: {ticker}",
        'no_data': f"No data available for {ticker}",
        'api_error': f"API error for {ticker}",
        'calculation_error': f"Calculation error for {ticker}",
        'general_error': "An error occurred during analysis"
    }
    
    message = base_messages.get(error_type, base_messages['general_error'])
    
    if details:
        message += f" - {details}"
    
    return message

def log_performance(func):
    """
    Decorator to log function performance
    
    Args:
        func: Function to decorate
        
    Returns:
        function: Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logging.info(f"{func.__name__} completed in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logging.error(f"{func.__name__} failed after {duration:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def get_sector_color(sector):
    """
    Get color for sector visualization
    
    Args:
        sector (str): Sector name
        
    Returns:
        str: Color code
    """
    sector_colors = {
        'Technology': '#1f77b4',
        'Healthcare': '#ff7f0e',
        'Financial Services': '#2ca02c',
        'Consumer Discretionary': '#d62728',
        'Communication Services': '#9467bd',
        'Industrials': '#8c564b',
        'Consumer Staples': '#e377c2',
        'Energy': '#7f7f7f',
        'Utilities': '#bcbd22',
        'Real Estate': '#17becf',
        'Materials': '#aec7e8'
    }
    
    return sector_colors.get(sector, '#cccccc')

def get_recommendation_color(recommendation):
    """
    Get color for recommendation visualization
    
    Args:
        recommendation (str): Recommendation action
        
    Returns:
        str: Color code
    """
    colors = {
        'STRONG BUY': '#00ff00',
        'BUY': '#90EE90',
        'HOLD': '#ffff00',
        'SELL': '#FFA500',
        'STRONG SELL': '#ff0000'
    }
    
    return colors.get(recommendation, '#cccccc')

def sanitize_filename(filename):
    """
    Sanitize filename for saving files
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing periods and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename
