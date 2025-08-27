"""
Technical analysis module for the AI Stock Analysis Bot
Implements various technical indicators and pattern detection
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import signal
from scipy.stats import linregress
import logging

class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, data):
        """
        Calculate technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            dict: Dictionary of calculated indicators
        """
        indicators = {}
        
        try:
            # Simple Moving Averages
            indicators['SMA_5'] = ta.sma(data['Close'], length=5)
            indicators['SMA_10'] = ta.sma(data['Close'], length=10)
            indicators['SMA_20'] = ta.sma(data['Close'], length=20)
            indicators['SMA_50'] = ta.sma(data['Close'], length=50)
            indicators['SMA_200'] = ta.sma(data['Close'], length=200)
            
            # Exponential Moving Averages
            indicators['EMA_12'] = ta.ema(data['Close'], length=12)
            indicators['EMA_26'] = ta.ema(data['Close'], length=26)
            indicators['EMA_50'] = ta.ema(data['Close'], length=50)
            
            # RSI (Relative Strength Index)
            indicators['RSI'] = ta.rsi(data['Close'], length=14)
            
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(data['Close'])
            indicators['MACD'] = macd['MACD_12_26_9']
            indicators['MACD_Signal'] = macd['MACDs_12_26_9']
            indicators['MACD_Histogram'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(data['Close'], length=20, std=2)
            indicators['BB_Upper'] = bb['BBU_20_2.0']
            indicators['BB_Middle'] = bb['BBM_20_2.0']
            indicators['BB_Lower'] = bb['BBL_20_2.0']
            
            # Stochastic Oscillator
            stoch = ta.stoch(data['High'], data['Low'], data['Close'])
            indicators['Stoch_K'] = stoch['STOCHk_14_3_3']
            indicators['Stoch_D'] = stoch['STOCHd_14_3_3']
            
            # Average True Range
            indicators['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
            
            # Volume indicators
            indicators['Volume_SMA'] = ta.sma(data['Volume'], length=20)
            indicators['OBV'] = ta.obv(data['Close'], data['Volume'])
            
            # Price momentum
            indicators['Momentum'] = ta.mom(data['Close'], length=10)
            indicators['ROC'] = ta.roc(data['Close'], length=10)
            
            # Support and Resistance levels
            indicators['Support'] = self.calculate_support_resistance(data, 'support')
            indicators['Resistance'] = self.calculate_support_resistance(data, 'resistance')
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def calculate_support_resistance(self, data, level_type='support'):
        """
        Calculate support and resistance levels
        
        Args:
            data (pd.DataFrame): Stock data
            level_type (str): 'support' or 'resistance'
            
        Returns:
            pd.Series: Support or resistance levels
        """
        try:
            window = 20
            if level_type == 'support':
                # Find local minima
                lows = data['Low'].rolling(window=window, center=True).min()
                support = data['Low'].where(data['Low'] == lows)
            else:
                # Find local maxima
                highs = data['High'].rolling(window=window, center=True).max()
                support = data['High'].where(data['High'] == highs)
            
            return support.ffill().bfill()
            
        except Exception as e:
            self.logger.error(f"Error calculating {level_type}: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def detect_patterns(self, data, indicators):
        """
        Detect chart patterns
        
        Args:
            data (pd.DataFrame): Stock data
            indicators (dict): Technical indicators
            
        Returns:
            list: List of detected patterns
        """
        patterns = []
        
        try:
            # Double bottom pattern
            if self.detect_double_bottom(data):
                patterns.append("Double Bottom - Bullish reversal pattern detected")
            
            # Double top pattern
            if self.detect_double_top(data):
                patterns.append("Double Top - Bearish reversal pattern detected")
            
            # Head and shoulders
            if self.detect_head_shoulders(data):
                patterns.append("Head and Shoulders - Bearish reversal pattern detected")
            
            # Inverse head and shoulders
            if self.detect_inverse_head_shoulders(data):
                patterns.append("Inverse Head and Shoulders - Bullish reversal pattern detected")
            
            # Triangle patterns
            triangle_pattern = self.detect_triangle_pattern(data)
            if triangle_pattern:
                patterns.append(f"{triangle_pattern} - Continuation pattern detected")
            
            # Flag and pennant patterns
            flag_pattern = self.detect_flag_pattern(data)
            if flag_pattern:
                patterns.append(f"{flag_pattern} - Continuation pattern detected")
            
            # Cup and handle
            if self.detect_cup_handle(data):
                patterns.append("Cup and Handle - Bullish continuation pattern detected")
            
            # Candlestick patterns
            candlestick_patterns = self.detect_candlestick_patterns(data)
            patterns.extend(candlestick_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return []
    
    def detect_double_bottom(self, data, window=20):
        """Detect double bottom pattern"""
        try:
            lows = data['Low'].rolling(window=window).min()
            local_minima = data['Low'] == lows
            
            if local_minima.sum() < 2:
                return False
            
            # Get the two most recent significant lows
            recent_lows = data[local_minima]['Low'].tail(2)
            
            if len(recent_lows) < 2:
                return False
            
            # Check if the lows are similar (within 2% difference)
            low1, low2 = recent_lows.iloc[0], recent_lows.iloc[1]
            if abs(low1 - low2) / low1 <= 0.02:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting double bottom: {str(e)}")
            return False
    
    def detect_double_top(self, data, window=20):
        """Detect double top pattern"""
        try:
            highs = data['High'].rolling(window=window).max()
            local_maxima = data['High'] == highs
            
            if local_maxima.sum() < 2:
                return False
            
            # Get the two most recent significant highs
            recent_highs = data[local_maxima]['High'].tail(2)
            
            if len(recent_highs) < 2:
                return False
            
            # Check if the highs are similar (within 2% difference)
            high1, high2 = recent_highs.iloc[0], recent_highs.iloc[1]
            if abs(high1 - high2) / high1 <= 0.02:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting double top: {str(e)}")
            return False
    
    def detect_head_shoulders(self, data):
        """Detect head and shoulders pattern"""
        try:
            # Simplified head and shoulders detection
            # Look for three peaks with the middle one being the highest
            highs = data['High'].rolling(window=10).max()
            peaks = data['High'] == highs
            
            if peaks.sum() < 3:
                return False
            
            recent_peaks = data[peaks]['High'].tail(3)
            
            if len(recent_peaks) < 3:
                return False
            
            left_shoulder, head, right_shoulder = recent_peaks.iloc[0], recent_peaks.iloc[1], recent_peaks.iloc[2]
            
            # Head should be higher than both shoulders
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly the same height
                if abs(left_shoulder - right_shoulder) / left_shoulder <= 0.05:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {str(e)}")
            return False
    
    def detect_inverse_head_shoulders(self, data):
        """Detect inverse head and shoulders pattern"""
        try:
            # Look for three troughs with the middle one being the lowest
            lows = data['Low'].rolling(window=10).min()
            troughs = data['Low'] == lows
            
            if troughs.sum() < 3:
                return False
            
            recent_troughs = data[troughs]['Low'].tail(3)
            
            if len(recent_troughs) < 3:
                return False
            
            left_shoulder, head, right_shoulder = recent_troughs.iloc[0], recent_troughs.iloc[1], recent_troughs.iloc[2]
            
            # Head should be lower than both shoulders
            if head < left_shoulder and head < right_shoulder:
                # Shoulders should be roughly the same height
                if abs(left_shoulder - right_shoulder) / left_shoulder <= 0.05:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting inverse head and shoulders: {str(e)}")
            return False
    
    def detect_triangle_pattern(self, data):
        """Detect triangle patterns"""
        try:
            # Calculate trend lines for highs and lows
            high_slope = self.calculate_trend_slope(data['High'])
            low_slope = self.calculate_trend_slope(data['Low'])
            
            # Ascending triangle: flat resistance, rising support
            if abs(high_slope) < 0.1 and low_slope > 0.1:
                return "Ascending Triangle"
            
            # Descending triangle: declining resistance, flat support
            if high_slope < -0.1 and abs(low_slope) < 0.1:
                return "Descending Triangle"
            
            # Symmetrical triangle: declining resistance, rising support
            if high_slope < -0.1 and low_slope > 0.1:
                return "Symmetrical Triangle"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting triangle pattern: {str(e)}")
            return None
    
    def detect_flag_pattern(self, data):
        """Detect flag and pennant patterns"""
        try:
            # Look for consolidation after strong move
            recent_data = data.tail(20)
            
            # Check for strong prior move
            price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
            
            if abs(price_change) > 0.1:  # Strong move (>10%)
                # Check for consolidation (low volatility)
                volatility = recent_data['Close'].std() / recent_data['Close'].mean()
                
                if volatility < 0.02:  # Low volatility consolidation
                    if price_change > 0:
                        return "Bull Flag"
                    else:
                        return "Bear Flag"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting flag pattern: {str(e)}")
            return None
    
    def detect_cup_handle(self, data):
        """Detect cup and handle pattern"""
        try:
            # Simplified cup and handle detection
            # Look for U-shaped recovery followed by small pullback
            
            if len(data) < 50:
                return False
            
            # Get data for cup formation
            cup_data = data.tail(50)
            
            # Find the low point (bottom of cup)
            low_idx = cup_data['Low'].idxmin()
            
            # Check if recovery from low forms a U-shape
            left_side = cup_data.loc[:low_idx]
            right_side = cup_data.loc[low_idx:]
            
            if len(left_side) < 10 or len(right_side) < 10:
                return False
            
            # Check if current price is near the highs
            current_price = cup_data['Close'].iloc[-1]
            cup_high = cup_data['High'].max()
            
            if current_price > cup_high * 0.9:  # Within 10% of high
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting cup and handle: {str(e)}")
            return False
    
    def detect_candlestick_patterns(self, data):
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            # Get recent candlesticks
            recent = data.tail(5)
            
            if len(recent) < 2:
                return patterns
            
            # Current and previous candles
            current = recent.iloc[-1]
            previous = recent.iloc[-2]
            
            # Hammer pattern
            if self.is_hammer(current):
                patterns.append("Hammer - Bullish reversal candlestick")
            
            # Shooting star
            if self.is_shooting_star(current):
                patterns.append("Shooting Star - Bearish reversal candlestick")
            
            # Doji
            if self.is_doji(current):
                patterns.append("Doji - Indecision candlestick")
            
            # Engulfing patterns
            if self.is_bullish_engulfing(previous, current):
                patterns.append("Bullish Engulfing - Strong bullish reversal")
            
            if self.is_bearish_engulfing(previous, current):
                patterns.append("Bearish Engulfing - Strong bearish reversal")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return []
    
    def is_hammer(self, candle):
        """Check if candle is a hammer"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            lower_shadow = candle['Open'] - candle['Low'] if candle['Close'] > candle['Open'] else candle['Close'] - candle['Low']
            upper_shadow = candle['High'] - candle['Close'] if candle['Close'] > candle['Open'] else candle['High'] - candle['Open']
            
            return lower_shadow > 2 * body and upper_shadow < 0.5 * body
            
        except Exception:
            return False
    
    def is_shooting_star(self, candle):
        """Check if candle is a shooting star"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            lower_shadow = candle['Open'] - candle['Low'] if candle['Close'] > candle['Open'] else candle['Close'] - candle['Low']
            upper_shadow = candle['High'] - candle['Close'] if candle['Close'] > candle['Open'] else candle['High'] - candle['Open']
            
            return upper_shadow > 2 * body and lower_shadow < 0.5 * body
            
        except Exception:
            return False
    
    def is_doji(self, candle):
        """Check if candle is a doji"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            total_range = candle['High'] - candle['Low']
            
            return body < 0.1 * total_range
            
        except Exception:
            return False
    
    def is_bullish_engulfing(self, prev_candle, curr_candle):
        """Check for bullish engulfing pattern"""
        try:
            # Previous candle should be bearish
            prev_bearish = prev_candle['Close'] < prev_candle['Open']
            
            # Current candle should be bullish
            curr_bullish = curr_candle['Close'] > curr_candle['Open']
            
            # Current candle should engulf previous candle
            engulfs = (curr_candle['Open'] < prev_candle['Close'] and 
                      curr_candle['Close'] > prev_candle['Open'])
            
            return prev_bearish and curr_bullish and engulfs
            
        except Exception:
            return False
    
    def is_bearish_engulfing(self, prev_candle, curr_candle):
        """Check for bearish engulfing pattern"""
        try:
            # Previous candle should be bullish
            prev_bullish = prev_candle['Close'] > prev_candle['Open']
            
            # Current candle should be bearish
            curr_bearish = curr_candle['Close'] < curr_candle['Open']
            
            # Current candle should engulf previous candle
            engulfs = (curr_candle['Open'] > prev_candle['Close'] and 
                      curr_candle['Close'] < prev_candle['Open'])
            
            return prev_bullish and curr_bearish and engulfs
            
        except Exception:
            return False
    
    def calculate_trend_slope(self, series):
        """Calculate trend slope using linear regression"""
        try:
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            valid_idx = ~np.isnan(y)
            x = x[valid_idx]
            y = y[valid_idx]
            
            if len(x) < 2:
                return 0
            
            slope, _, _, _, _ = linregress(x, y)
            return slope
            
        except Exception as e:
            self.logger.error(f"Error calculating trend slope: {str(e)}")
            return 0
    
    def calculate_ta_score(self, data, indicators):
        """
        Calculate technical analysis score
        
        Args:
            data (pd.DataFrame): Stock data
            indicators (dict): Technical indicators
            
        Returns:
            float: TA score (0-100)
        """
        try:
            score = 0
            total_weight = 0
            
            # Current values
            current_price = data['Close'].iloc[-1]
            current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else 50
            current_macd = indicators['MACD'].iloc[-1] if not indicators['MACD'].empty else 0
            current_macd_signal = indicators['MACD_Signal'].iloc[-1] if not indicators['MACD_Signal'].empty else 0
            
            # RSI scoring (weight: 20)
            if not np.isnan(current_rsi):
                if current_rsi < 30:
                    score += 80 * 20  # Oversold - bullish
                elif current_rsi < 50:
                    score += 60 * 20
                elif current_rsi < 70:
                    score += 40 * 20
                else:
                    score += 20 * 20  # Overbought - bearish
                total_weight += 20
            
            # MACD scoring (weight: 15)
            if not np.isnan(current_macd) and not np.isnan(current_macd_signal):
                if current_macd > current_macd_signal:
                    score += 70 * 15  # Bullish signal
                else:
                    score += 30 * 15  # Bearish signal
                total_weight += 15
            
            # Moving average scoring (weight: 20)
            sma_20 = indicators['SMA_20'].iloc[-1] if not indicators['SMA_20'].empty else current_price
            sma_50 = indicators['SMA_50'].iloc[-1] if not indicators['SMA_50'].empty else current_price
            
            if not np.isnan(sma_20) and not np.isnan(sma_50):
                if current_price > sma_20 > sma_50:
                    score += 80 * 20  # Strong uptrend
                elif current_price > sma_20:
                    score += 60 * 20  # Uptrend
                elif current_price < sma_20 < sma_50:
                    score += 20 * 20  # Strong downtrend
                else:
                    score += 40 * 20  # Downtrend
                total_weight += 20
            
            # Volume analysis (weight: 10)
            current_volume = data['Volume'].iloc[-1]
            avg_volume = indicators['Volume_SMA'].iloc[-1] if not indicators['Volume_SMA'].empty else current_volume
            
            if not np.isnan(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > 1.5:
                    score += 70 * 10  # High volume
                elif volume_ratio > 1.0:
                    score += 60 * 10  # Above average volume
                else:
                    score += 40 * 10  # Below average volume
                total_weight += 10
            
            # Bollinger Bands scoring (weight: 10)
            bb_upper = indicators['BB_Upper'].iloc[-1] if not indicators['BB_Upper'].empty else current_price
            bb_lower = indicators['BB_Lower'].iloc[-1] if not indicators['BB_Lower'].empty else current_price
            
            if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                if current_price < bb_lower:
                    score += 80 * 10  # Oversold
                elif current_price > bb_upper:
                    score += 20 * 10  # Overbought
                else:
                    score += 50 * 10  # Normal range
                total_weight += 10
            
            # Stochastic scoring (weight: 10)
            stoch_k = indicators['Stoch_K'].iloc[-1] if not indicators['Stoch_K'].empty else 50
            
            if not np.isnan(stoch_k):
                if stoch_k < 20:
                    score += 80 * 10  # Oversold
                elif stoch_k > 80:
                    score += 20 * 10  # Overbought
                else:
                    score += 50 * 10  # Normal range
                total_weight += 10
            
            # Momentum scoring (weight: 15)
            momentum = indicators['Momentum'].iloc[-1] if not indicators['Momentum'].empty else 0
            
            if not np.isnan(momentum):
                if momentum > 0:
                    score += 70 * 15  # Positive momentum
                else:
                    score += 30 * 15  # Negative momentum
                total_weight += 15
            
            # Calculate final score
            if total_weight > 0:
                final_score = score / total_weight
            else:
                final_score = 50  # Neutral score if no indicators available
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating TA score: {str(e)}")
            return 50
    
    def analyze_stock(self, data, ticker):
        """
        Perform complete technical analysis on a stock
        
        Args:
            data (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Complete technical analysis results
        """
        try:
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            
            # Detect patterns
            patterns = self.detect_patterns(data, indicators)
            
            # Calculate TA score
            score = self.calculate_ta_score(data, indicators)
            
            # Generate signals
            signals = self.generate_signals(data, indicators)
            
            results = {
                'ticker': ticker,
                'indicators': indicators,
                'patterns': patterns,
                'score': score,
                'signals': signals,
                'analysis_date': pd.Timestamp.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing stock {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'indicators': {},
                'patterns': [],
                'score': 50,
                'signals': [],
                'analysis_date': pd.Timestamp.now().isoformat()
            }
    
    def generate_signals(self, data, indicators):
        """Generate trading signals based on technical analysis"""
        signals = []
        
        try:
            current_price = data['Close'].iloc[-1]
            
            # RSI signals
            current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else 50
            if not np.isnan(current_rsi):
                if current_rsi < 30:
                    signals.append("RSI Oversold - Potential buy signal")
                elif current_rsi > 70:
                    signals.append("RSI Overbought - Potential sell signal")
            
            # MACD signals
            current_macd = indicators['MACD'].iloc[-1] if not indicators['MACD'].empty else 0
            current_macd_signal = indicators['MACD_Signal'].iloc[-1] if not indicators['MACD_Signal'].empty else 0
            
            if not np.isnan(current_macd) and not np.isnan(current_macd_signal):
                if current_macd > current_macd_signal:
                    signals.append("MACD Bullish - Price above signal line")
                else:
                    signals.append("MACD Bearish - Price below signal line")
            
            # Moving average signals
            sma_20 = indicators['SMA_20'].iloc[-1] if not indicators['SMA_20'].empty else current_price
            sma_50 = indicators['SMA_50'].iloc[-1] if not indicators['SMA_50'].empty else current_price
            
            if not np.isnan(sma_20) and not np.isnan(sma_50):
                if current_price > sma_20 > sma_50:
                    signals.append("Strong Uptrend - Price above both MAs")
                elif current_price < sma_20 < sma_50:
                    signals.append("Strong Downtrend - Price below both MAs")
            
            # Bollinger Bands signals
            bb_upper = indicators['BB_Upper'].iloc[-1] if not indicators['BB_Upper'].empty else current_price
            bb_lower = indicators['BB_Lower'].iloc[-1] if not indicators['BB_Lower'].empty else current_price
            
            if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                if current_price < bb_lower:
                    signals.append("Bollinger Bands - Price below lower band (oversold)")
                elif current_price > bb_upper:
                    signals.append("Bollinger Bands - Price above upper band (overbought)")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
