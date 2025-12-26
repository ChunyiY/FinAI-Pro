"""
Utility Functions Module

Helper functions for formatting and data processing.
"""
from typing import Union
from datetime import datetime, timedelta


def format_currency(value: float) -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
    
    Returns:
        Formatted currency string
    """
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%"


def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock symbol string
    
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    # Basic validation: alphanumeric, length 1-10
    return symbol.isalnum() and 1 <= len(symbol) <= 10


def prepare_date_range(days: int = 365) -> tuple:
    """
    Prepare date range.
    
    Args:
        days: Number of days
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date
