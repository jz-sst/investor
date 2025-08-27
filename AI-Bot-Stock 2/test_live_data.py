#!/usr/bin/env python3
"""
Test live data access and technical analysis feasibility
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def test_live_data():
    """Test live data access with real market data"""
    print("=== Testing Live Data Access ===\n")
    
    # Test multiple tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Get current stock info
            info = stock.info
            print(f"🔍 {ticker} - {info.get('longName', 'N/A')}")
            print(f"   Current Price: ${info.get('currentPrice', 'N/A')}")
            print(f"   Market Cap: ${info.get('marketCap', 0):,}")
            print(f"   Volume: {info.get('volume', 0):,}")
            print(f"   P/E Ratio: {info.get('trailingPE', 'N/A')}")
            
            # Get recent stock data
            data = stock.history(period='3mo')
            if not data.empty:
                print(f"   Data Points: {len(data)} days")
                print(f"   Latest Close: ${data['Close'].iloc[-1]:.2f}")
                print(f"   30-day Average: ${data['Close'].tail(30).mean():.2f}")
                
                # Test technical indicators
                if len(data) >= 20:
                    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                    print(f"   20-day SMA: ${sma_20:.2f}")
                
                if len(data) >= 14:
                    # Simple RSI calculation
                    delta = data['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    print(f"   RSI: {rsi.iloc[-1]:.2f}")
                
                print(f"   ✅ Technical Analysis Feasible")
            else:
                print(f"   ❌ No data available")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()

def test_technical_analysis():
    """Test comprehensive technical analysis"""
    print("=== Testing Technical Analysis ===\n")
    
    ticker = 'AAPL'
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')
    
    if data.empty:
        print("❌ No data available for technical analysis")
        return
    
    print(f"📊 Analyzing {ticker} with {len(data)} data points")
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Latest values
    latest = data.iloc[-1]
    print(f"Current Price: ${latest['Close']:.2f}")
    print(f"20-day SMA: ${latest['SMA_20']:.2f}")
    print(f"50-day SMA: ${latest['SMA_50']:.2f}")
    print(f"RSI: {latest['RSI']:.2f}")
    print(f"MACD: {latest['MACD']:.4f}")
    print(f"Signal: {latest['Signal']:.4f}")
    
    # Analysis
    print("\n📈 Technical Analysis:")
    
    # Trend Analysis
    if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
        print("✅ Uptrend: Price > SMA20 > SMA50")
    elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
        print("❌ Downtrend: Price < SMA20 < SMA50")
    else:
        print("⚠️  Sideways/Mixed trend")
    
    # RSI Analysis
    if latest['RSI'] > 70:
        print("⚠️  RSI Overbought (>70)")
    elif latest['RSI'] < 30:
        print("🟢 RSI Oversold (<30) - Potential Buy")
    else:
        print("✅ RSI Neutral (30-70)")
    
    # MACD Analysis
    if latest['MACD'] > latest['Signal']:
        print("🟢 MACD Bullish Signal")
    else:
        print("❌ MACD Bearish Signal")
    
    # Bollinger Bands
    if latest['Close'] > latest['BB_Upper']:
        print("⚠️  Price above Upper Bollinger Band")
    elif latest['Close'] < latest['BB_Lower']:
        print("🟢 Price below Lower Bollinger Band - Potential Buy")
    else:
        print("✅ Price within Bollinger Bands")
    
    print("\n✅ Technical Analysis System Fully Functional!")

def test_fundamental_data():
    """Test fundamental data access"""
    print("=== Testing Fundamental Data ===\n")
    
    ticker = 'AAPL'
    stock = yf.Ticker(ticker)
    info = stock.info
    
    print(f"📊 Fundamental Data for {ticker}:")
    print(f"Market Cap: ${info.get('marketCap', 0):,}")
    print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"EPS: ${info.get('trailingEps', 'N/A')}")
    print(f"Revenue: ${info.get('totalRevenue', 0):,}")
    print(f"Profit Margins: {info.get('profitMargins', 'N/A')}")
    print(f"ROE: {info.get('returnOnEquity', 'N/A')}")
    print(f"Debt to Equity: {info.get('debtToEquity', 'N/A')}")
    print(f"Current Ratio: {info.get('currentRatio', 'N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    
    print("\n✅ Fundamental Analysis Data Available!")

if __name__ == "__main__":
    test_live_data()
    test_technical_analysis()
    test_fundamental_data()