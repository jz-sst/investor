#!/usr/bin/env python3
"""
Test yfinance API status and signal generation
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_yfinance_api():
    """Test yfinance API functionality"""
    print("=== Testing yfinance API Status ===")
    
    # Test basic functionality
    try:
        spy = yf.Ticker('SPY')
        spy_data = spy.history(period='1d')
        print(f"✅ SPY data retrieved: {len(spy_data)} records")
        print(f"✅ Current SPY price: ${spy_data['Close'].iloc[-1]:.2f}")
        
        # Test multiple stocks
        test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        results = []
        
        for ticker in test_stocks:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='5d')
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    results.append(f"{ticker}: ${price:.2f}")
                else:
                    results.append(f"{ticker}: No data")
            except Exception as e:
                results.append(f"{ticker}: Error - {str(e)[:30]}")
        
        print(f"✅ Stock prices: {' | '.join(results)}")
        
        # Test discovery stocks
        discovery_categories = {
            'Growth': ['NVDA', 'TSLA', 'SHOP'],
            'Value': ['BRK.B', 'JPM', 'JNJ'],
            'Tech': ['AAPL', 'MSFT', 'GOOGL'],
            'Dividend': ['T', 'VZ', 'XOM']
        }
        
        signals = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        total_analyzed = 0
        detailed_results = []
        
        for category, tickers in discovery_categories.items():
            for ticker in tickers[:1]:  # Test 1 per category
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period='1mo')
                    if not data.empty:
                        # Simple signal generation
                        current_price = data['Close'].iloc[-1]
                        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                        
                        price_vs_sma = (current_price - sma_20) / sma_20 * 100
                        
                        if current_price > sma_20 * 1.02:
                            signals['BUY'] += 1
                            signal = 'BUY'
                        elif current_price < sma_20 * 0.98:
                            signals['SELL'] += 1
                            signal = 'SELL'
                        else:
                            signals['HOLD'] += 1
                            signal = 'HOLD'
                        
                        detailed_results.append({
                            'ticker': ticker,
                            'category': category,
                            'current_price': current_price,
                            'sma_20': sma_20,
                            'price_vs_sma': price_vs_sma,
                            'signal': signal
                        })
                        
                        total_analyzed += 1
                except Exception as e:
                    print(f"Error analyzing {ticker}: {str(e)}")
        
        print(f"✅ Discovery analysis: {total_analyzed} stocks analyzed")
        print(f"✅ Signal distribution: BUY={signals['BUY']}, HOLD={signals['HOLD']}, SELL={signals['SELL']}")
        
        print("\n--- Detailed Results ---")
        for result in detailed_results:
            print(f"{result['ticker']} ({result['category']}): {result['signal']}")
            print(f"  Price: ${result['current_price']:.2f}, SMA20: ${result['sma_20']:.2f}")
            print(f"  Price vs SMA20: {result['price_vs_sma']:+.1f}%")
        
        if signals['BUY'] == 0 and signals['SELL'] == 0:
            print("\n⚠️ Zero strong signals detected - reasons:")
            print("   • Market may be in neutral/sideways trend")
            print("   • Prices close to moving averages (within 2% threshold)")
            print("   • Need longer analysis period for clear signals")
            print("   • Consider lowering signal thresholds or adding more indicators")
            
            # Suggest more sensitive thresholds
            print("\n💡 With more sensitive thresholds (1% vs 2%):")
            sensitive_signals = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
            for result in detailed_results:
                if result['price_vs_sma'] > 1.0:
                    sensitive_signals['BUY'] += 1
                elif result['price_vs_sma'] < -1.0:
                    sensitive_signals['SELL'] += 1
                else:
                    sensitive_signals['HOLD'] += 1
            
            print(f"   BUY={sensitive_signals['BUY']}, HOLD={sensitive_signals['HOLD']}, SELL={sensitive_signals['SELL']}")
        
        print("\n✅ yfinance API is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing yfinance: {str(e)}")
        return False

if __name__ == "__main__":
    test_yfinance_api()