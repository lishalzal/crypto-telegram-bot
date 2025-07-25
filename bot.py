import os
import asyncio
import logging
import requests
import math
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import Dict, List, Optional, Tuple
from flask import Flask, request
import threading
import json

# إعداد الـ logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTechnicalAnalyzer:
    """محلل فني متقدم مع مؤشرات احترافية"""
    
    def __init__(self):
        self.timeframes = {
            '15m': {'limit': 200, 'name': '15 دقيقة'},
            '1h': {'limit': 168, 'name': 'ساعة واحدة'},
            '4h': {'limit': 168, 'name': '4 ساعات'},
            '1d': {'limit': 100, 'name': 'يوم واحد'},
            '1w': {'limit': 52, 'name': 'أسبوع واحد'}
        }
        
    def get_price_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[List]:
        """جلب بيانات الأسعار من Binance API مع معالجة محسنة للأخطاء"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': timeframe,
                'limit': limit
            }
            
            # تحسين timeout وheaders
            headers = {
                'User-Agent': 'TelegramBot/1.0'
            }
            
            response = requests.get(
                base_url, 
                params=params, 
                timeout=20,
                headers=headers
            )
            
            if response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limit for {symbol}, retrying...")
                import time
                time.sleep(2)
                return self.get_price_data(symbol, timeframe, limit)
            
            elif response.status_code == 400:
                logger.error(f"Invalid symbol: {symbol}")
                return None
                
            elif response.status_code != 200:
                logger.error(f"API Error {response.status_code} for {symbol}")
                return None
                
            data = response.json()
            if not data or len(data) < 10:
                logger.warning(f"Insufficient data for {symbol}")
                return None
                
            # تحويل البيانات مع validation محسن
            processed_data = []
            for i, row in enumerate(data):
                try:
                    if len(row) < 9:
                        continue
                        
                    candle = {
                        'timestamp': int(row[0]),
                        'open': float(row[1]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'close': float(row[4]),
                        'volume': float(row[5]),
                        'close_time': int(row[6]),
                        'quote_volume': float(row[7]),
                        'trades': int(row[8])
                    }
                    
                    # التحقق من صحة البيانات
                    if (candle['high'] >= candle['low'] and 
                        candle['high'] >= candle['close'] and 
                        candle['high'] >= candle['open'] and
                        candle['low'] <= candle['close'] and
                        candle['low'] <= candle['open'] and
                        candle['volume'] >= 0):
                        processed_data.append(candle)
                        
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f"Invalid candle data at index {i}: {e}")
                    continue
            
            if len(processed_data) >= 10:
                logger.info(f"Successfully loaded {len(processed_data)} candles for {symbol} {timeframe}")
                return processed_data
            else:
                logger.warning(f"Not enough valid data for {symbol} {timeframe}")
                return None
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, data: List[Dict]) -> Dict:
        """حساب المؤشرات الفنية المتقدمة بدون مكتبات خارجية"""
        try:
            if len(data) < 20:  # تقليل الحد الأدنى
                logger.warning(f"Insufficient data for indicators: {len(data)} candles")
                return {}
            
            # استخراج البيانات مع التحقق
            try:
                closes = [float(item['close']) for item in data if 'close' in item]
                highs = [float(item['high']) for item in data if 'high' in item]
                lows = [float(item['low']) for item in data if 'low' in item]
                volumes = [float(item['volume']) for item in data if 'volume' in item]
                
                if len(closes) != len(highs) or len(closes) != len(lows):
                    logger.error("Data length mismatch")
                    return {}
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Data extraction error: {e}")
                return {}
            
            indicators = {}
            current_price = closes[-1] if closes else 0
            
            # المتوسطات المتحركة البسيطة
            try:
                indicators['sma_20'] = self.calculate_sma(closes, min(20, len(closes)))
                indicators['sma_50'] = self.calculate_sma(closes, min(50, len(closes)))
                indicators['sma_200'] = self.calculate_sma(closes, min(200, len(closes)))
            except Exception as e:
                logger.warning(f"SMA calculation error: {e}")
                indicators['sma_20'] = current_price
                indicators['sma_50'] = current_price
                indicators['sma_200'] = current_price
            
            # المتوسطات المتحركة الأسية
            try:
                indicators['ema_9'] = self.calculate_ema(closes, min(9, len(closes)))
                indicators['ema_21'] = self.calculate_ema(closes, min(21, len(closes)))
                indicators['ema_50'] = self.calculate_ema(closes, min(50, len(closes)))
                indicators['ema_200'] = self.calculate_ema(closes, min(200, len(closes)))
            except Exception as e:
                logger.warning(f"EMA calculation error: {e}")
                indicators['ema_9'] = current_price
                indicators['ema_21'] = current_price
                indicators['ema_50'] = current_price
                indicators['ema_200'] = current_price
            
            # RSI مع فترات متعددة
            try:
                indicators['rsi_14'] = self.calculate_rsi(closes, min(14, len(closes) - 1))
                indicators['rsi_21'] = self.calculate_rsi(closes, min(21, len(closes) - 1))
            except Exception as e:
                logger.warning(f"RSI calculation error: {e}")
                indicators['rsi_14'] = 50
                indicators['rsi_21'] = 50
            
            # MACD
            try:
                macd_data = self.calculate_macd(closes)
                indicators.update(macd_data)
            except Exception as e:
                logger.warning(f"MACD calculation error: {e}")
                indicators.update({'macd': 0, 'macd_signal': 0, 'macd_histogram': 0})
            
            # نطاقات بولينجر
            try:
                bb_data = self.calculate_bollinger_bands(closes, min(20, len(closes)))
                indicators.update(bb_data)
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation error: {e}")
                indicators.update({
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98,
                    'bb_width': current_price * 0.04
                })
            
            # Stochastic
            try:
                stoch_data = self.calculate_stochastic(highs, lows, closes)
                indicators.update(stoch_data)
            except Exception as e:
                logger.warning(f"Stochastic calculation error: {e}")
                indicators.update({'stoch_k': 50, 'stoch_d': 50})
            
            # ADX
            try:
                adx_data = self.calculate_adx(highs, lows, closes)
                indicators.update(adx_data)
            except Exception as e:
                logger.warning(f"ADX calculation error: {e}")
                indicators.update({'adx': 25, 'plus_di': 25, 'minus_di': 25})
            
            # Volume indicators
            try:
                if volumes:
                    indicators['volume_sma'] = self.calculate_sma(volumes, min(20, len(volumes)))
                    indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
                else:
                    indicators['volume_sma'] = 0
                    indicators['volume_ratio'] = 1
            except Exception as e:
                logger.warning(f"Volume calculation error: {e}")
                indicators['volume_sma'] = 0
                indicators['volume_ratio'] = 1
            
            # Support and Resistance
            try:
                support_resistance = self.calculate_support_resistance(data[-min(50, len(data)):])
                indicators.update(support_resistance)
            except Exception as e:
                logger.warning(f"Support/Resistance calculation error: {e}")
                indicators.update({
                    'resistance': current_price * 1.05,
                    'support': current_price * 0.95,
                    'pivot_high': current_price * 1.1,
                    'pivot_low': current_price * 0.9
                })
            
            # Fibonacci levels
            try:
                fib_levels = self.calculate_fibonacci_levels(data[-min(100, len(data)):])
                indicators.update(fib_levels)
            except Exception as e:
                logger.warning(f"Fibonacci calculation error: {e}")
            
            # Current price data
            try:
                current = data[-1]
                indicators['current_price'] = float(current['close'])
                indicators['current_volume'] = float(current.get('volume', 0))
                
                if len(data) >= 24:
                    price_change = ((current['close'] - data[-24]['close']) / data[-24]['close'] * 100)
                    indicators['price_change_24h'] = price_change
                else:
                    indicators['price_change_24h'] = 0
                    
            except Exception as e:
                logger.warning(f"Current price calculation error: {e}")
                indicators['current_price'] = current_price
                indicators['current_volume'] = 0
                indicators['price_change_24h'] = 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def calculate_sma(self, prices: List[float], period: int) -> float:
        """حساب المتوسط المتحرك البسيط"""
        try:
            if len(prices) < period:
                return 0
            return sum(prices[-period:]) / period
        except:
            return 0

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """حساب المتوسط المتحرك الأسي"""
        try:
            if len(prices) < period:
                return self.calculate_sma(prices, len(prices))
            
            multiplier = 2 / (period + 1)
            ema = self.calculate_sma(prices[:period], period)
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
        except:
            return 0

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """حساب RSI محسن"""
        try:
            if len(prices) < period + 1:
                return 50
                
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return 50
            
            # حساب المتوسطات الأولية
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            # تطبيق صيغة Wilder's smoothing
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """حساب MACD"""
        try:
            if len(prices) < slow:
                return {'macd': 0, 'macd_signal': 0, 'macd_histogram': 0}
            
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            
            macd_line = ema_fast - ema_slow
            
            # حساب إشارة MACD (EMA للـ MACD line)
            macd_values = []
            for i in range(slow, len(prices)):
                temp_fast = self.calculate_ema(prices[:i+1], fast)
                temp_slow = self.calculate_ema(prices[:i+1], slow)
                macd_values.append(temp_fast - temp_slow)
            
            signal_line = self.calculate_ema(macd_values, signal) if len(macd_values) >= signal else macd_line
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }
        except:
            return {'macd': 0, 'macd_signal': 0, 'macd_histogram': 0}

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """حساب نطاقات بولينجر"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0
                return {
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98,
                    'bb_width': current_price * 0.04
                }
            
            sma = self.calculate_sma(prices, period)
            
            # حساب الانحراف المعياري
            recent_prices = prices[-period:]
            variance = sum((price - sma) ** 2 for price in recent_prices) / period
            std = math.sqrt(variance)
            
            return {
                'bb_upper': sma + (std * std_dev),
                'bb_middle': sma,
                'bb_lower': sma - (std * std_dev),
                'bb_width': std * std_dev * 2
            }
        except:
            current_price = prices[-1] if prices else 0
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98,
                'bb_width': current_price * 0.04
            }

    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Dict:
        """حساب Stochastic Oscillator"""
        try:
            if len(closes) < k_period:
                return {'stoch_k': 50, 'stoch_d': 50}
            
            k_values = []
            
            for i in range(k_period - 1, len(closes)):
                period_high = max(highs[i - k_period + 1:i + 1])
                period_low = min(lows[i - k_period + 1:i + 1])
                
                if period_high == period_low:
                    k_percent = 50
                else:
                    k_percent = ((closes[i] - period_low) / (period_high - period_low)) * 100
                
                k_values.append(k_percent)
            
            k_current = k_values[-1] if k_values else 50
            d_current = self.calculate_sma(k_values, min(d_period, len(k_values))) if k_values else 50
            
            return {
                'stoch_k': k_current,
                'stoch_d': d_current
            }
        except:
            return {'stoch_k': 50, 'stoch_d': 50}

    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
        """حساب ADX مبسط"""
        try:
            if len(closes) < period + 1:
                return {'adx': 25, 'plus_di': 25, 'minus_di': 25}
            
            # حساب True Range و Directional Movement
            true_ranges = []
            plus_dms = []
            minus_dms = []
            
            for i in range(1, len(closes)):
                # True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
                
                # Directional Movement
                plus_dm = max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
                minus_dm = max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
                
                plus_dms.append(plus_dm)
                minus_dms.append(minus_dm)
            
            if len(true_ranges) < period:
                return {'adx': 25, 'plus_di': 25, 'minus_di': 25}
            
            # حساب المتوسطات
            avg_tr = self.calculate_sma(true_ranges[-period:], period)
            avg_plus_dm = self.calculate_sma(plus_dms[-period:], period)
            avg_minus_dm = self.calculate_sma(minus_dms[-period:], period)
            
            # حساب DI
            plus_di = (avg_plus_dm / avg_tr * 100) if avg_tr > 0 else 0
            minus_di = (avg_minus_dm / avg_tr * 100) if avg_tr > 0 else 0
            
            # حساب ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
            
            # تقدير ADX كمتوسط للـ DX (مبسط)
            adx = dx
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except:
            return {'adx': 25, 'plus_di': 25, 'minus_di': 25}

    def calculate_support_resistance(self, data):
        """حساب مستويات الدعم والمقاومة"""
        try:
            highs = [candle['high'] for candle in data]
            lows = [candle['low'] for candle in data]
            closes = [candle['close'] for candle in data]
            
            # Pivot points
            recent_high = max(highs[-10:])
            recent_low = min(lows[-10:])
            
            # Dynamic support/resistance
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(data) - 2):
                # Resistance
                if (data[i]['high'] > data[i-1]['high'] and 
                    data[i]['high'] > data[i-2]['high'] and
                    data[i]['high'] > data[i+1]['high'] and 
                    data[i]['high'] > data[i+2]['high']):
                    resistance_levels.append(data[i]['high'])
                
                # Support
                if (data[i]['low'] < data[i-1]['low'] and 
                    data[i]['low'] < data[i-2]['low'] and
                    data[i]['low'] < data[i+1]['low'] and 
                    data[i]['low'] < data[i+2]['low']):
                    support_levels.append(data[i]['low'])
            
            current_price = data[-1]['close']
            
            # أقرب مستويات الدعم والمقاومة
            resistance = min([r for r in resistance_levels if r > current_price], default=recent_high)
            support = max([s for s in support_levels if s < current_price], default=recent_low)
            
            return {
                'resistance': resistance,
                'support': support,
                'pivot_high': recent_high,
                'pivot_low': recent_low
            }
        except:
            current_price = data[-1]['close']
            return {
                'resistance': current_price * 1.05,
                'support': current_price * 0.95,
                'pivot_high': current_price * 1.1,
                'pivot_low': current_price * 0.9
            }

    def calculate_fibonacci_levels(self, data):
        """حساب مستويات فيبوناتشي"""
        try:
            highs = [candle['high'] for candle in data]
            lows = [candle['low'] for candle in data]
            
            swing_high = max(highs)
            swing_low = min(lows)
            
            diff = swing_high - swing_low
            
            # مستويات فيبوناتشي الرئيسية
            levels = {
                'fib_0': swing_high,
                'fib_236': swing_high - (diff * 0.236),
                'fib_382': swing_high - (diff * 0.382),
                'fib_50': swing_high - (diff * 0.5),
                'fib_618': swing_high - (diff * 0.618),
                'fib_786': swing_high - (diff * 0.786),
                'fib_100': swing_low
            }
            
            return levels
        except:
            return {}

    def analyze_market_structure(self, indicators: Dict) -> Dict:
        """تحليل هيكل السوق والاتجاه"""
        try:
            analysis = {
                'trend_strength': 0,
                'trend_direction': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volatility': 'MEDIUM',
                'volume_analysis': 'NORMAL',
                'signals': []
            }
            
            current_price = indicators.get('current_price', 0)
            
            # تحليل الاتجاه باستخدام المتوسطات المتحركة
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            ema_200 = indicators.get('ema_200', current_price)
            
            trend_signals = 0
            
            # ترتيب المتوسطات
            if current_price > ema_9 > ema_21 > ema_50:
                trend_signals += 2
                analysis['signals'].append("📈 ترتيب صاعد للمتوسطات")
            elif current_price < ema_9 < ema_21 < ema_50:
                trend_signals -= 2
                analysis['signals'].append("📉 ترتيب هابط للمتوسطات")
            
            # RSI Analysis
            rsi_14 = indicators.get('rsi_14', 50)
            if rsi_14 < 30:
                analysis['signals'].append("🔵 RSI في منطقة الشراء المفرط")
                trend_signals += 1
            elif rsi_14 > 70:
                analysis['signals'].append("🔴 RSI في منطقة البيع المفرط")
                trend_signals -= 1
            
            # MACD Analysis
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > 0:
                analysis['signals'].append("✅ MACD إشارة صاعدة")
                trend_signals += 1
            elif macd < macd_signal and macd_histogram < 0:
                analysis['signals'].append("❌ MACD إشارة هابطة")
                trend_signals -= 1
            
            # Bollinger Bands
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            bb_middle = indicators.get('bb_middle', current_price)
            
            if current_price <= bb_lower:
                analysis['signals'].append("🎯 السعر عند النطاق السفلي")
                trend_signals += 0.5
            elif current_price >= bb_upper:
                analysis['signals'].append("⚠️ السعر عند النطاق العلوي")
                trend_signals -= 0.5
            
            # ADX Trend Strength
            adx = indicators.get('adx', 25)
            plus_di = indicators.get('plus_di', 25)
            minus_di = indicators.get('minus_di', 25)
            
            if adx > 25:
                if plus_di > minus_di:
                    analysis['signals'].append(f"💪 اتجاه صاعد قوي (ADX: {adx:.1f})")
                    trend_signals += 1
                else:
                    analysis['signals'].append(f"💪 اتجاه هابط قوي (ADX: {adx:.1f})")
                    trend_signals -= 1
            
            # Volume Analysis
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                analysis['volume_analysis'] = 'HIGH'
                analysis['signals'].append("📊 حجم تداول مرتفع")
            elif volume_ratio < 0.7:
                analysis['volume_analysis'] = 'LOW'
                analysis['signals'].append("📊 حجم تداول منخفض")
            
            # تحديد الاتجاه النهائي
            analysis['trend_strength'] = abs(trend_signals)
            
            if trend_signals > 2:
                analysis['trend_direction'] = 'STRONG_BULLISH'
                analysis['momentum'] = 'BULLISH'
            elif trend_signals > 0.5:
                analysis['trend_direction'] = 'BULLISH'
                analysis['momentum'] = 'BULLISH'
            elif trend_signals < -2:
                analysis['trend_direction'] = 'STRONG_BEARISH'
                analysis['momentum'] = 'BEARISH'
            elif trend_signals < -0.5:
                analysis['trend_direction'] = 'BEARISH'
                analysis['momentum'] = 'BEARISH'
            else:
                analysis['trend_direction'] = 'NEUTRAL'
                analysis['momentum'] = 'NEUTRAL'
            
            # تحليل التقلبات
            bb_width = indicators.get('bb_width', 0)
            if bb_width > current_price * 0.1:
                analysis['volatility'] = 'HIGH'
            elif bb_width < current_price * 0.05:
                analysis['volatility'] = 'LOW'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {
                'trend_strength': 0,
                'trend_direction': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volatility': 'MEDIUM',
                'volume_analysis': 'NORMAL',
                'signals': ['خطأ في التحليل']
            }

    def generate_trading_signals(self, indicators: Dict, market_analysis: Dict) -> Dict:
        """توليد إشارات التداول المتقدمة"""
        try:
            current_price = indicators.get('current_price', 0)
            
            signals = {
                'action': 'HOLD',
                'confidence': 0,
                'entry_points': [],
                'stop_loss': 0,
                'take_profits': [],
                'risk_reward': 0,
                'timeframe_recommendation': '4h',
                'position_size': 'MEDIUM'
            }
            
            trend_direction = market_analysis.get('trend_direction', 'NEUTRAL')
            trend_strength = market_analysis.get('trend_strength', 0)
            
            # تحديد الإجراء الأساسي
            if trend_direction in ['STRONG_BULLISH', 'BULLISH']:
                signals['action'] = 'BUY'
                signals['confidence'] = min(trend_strength * 20, 90)
                
                # نقاط الدخول للشراء
                support = indicators.get('support', current_price * 0.98)
                fib_618 = indicators.get('fib_618', current_price * 0.99)
                bb_lower = indicators.get('bb_lower', current_price * 0.97)
                
                signals['entry_points'] = [
                    max(support, current_price * 0.995),  # دخول فوري
                    support,  # دخول عند الدعم
                    min(fib_618, bb_lower)  # دخول عند مستويات فيبوناتشي
                ]
                
                # أهداف الربح
                resistance = indicators.get('resistance', current_price * 1.05)
                fib_236 = indicators.get('fib_236', current_price * 1.03)
                bb_upper = indicators.get('bb_upper', current_price * 1.04)
                
                signals['take_profits'] = [
                    current_price * 1.02,  # هدف سريع
                    min(resistance, fib_236),  # هدف متوسط
                    max(resistance, bb_upper),  # هدف رئيسي
                    current_price * 1.10  # هدف طموح
                ]
                
                # وقف الخسارة
                pivot_low = indicators.get('pivot_low', current_price * 0.92)
                signals['stop_loss'] = max(pivot_low, current_price * 0.95)
                
            elif trend_direction in ['STRONG_BEARISH', 'BEARISH']:
                signals['action'] = 'SELL'
                signals['confidence'] = min(trend_strength * 20, 90)
                
                # نقاط الدخول للبيع
                resistance = indicators.get('resistance', current_price * 1.02)
                fib_382 = indicators.get('fib_382', current_price * 1.01)
                bb_upper = indicators.get('bb_upper', current_price * 1.03)
                
                signals['entry_points'] = [
                    min(resistance, current_price * 1.005),  # دخول فوري
                    resistance,  # دخول عند المقاومة
                    max(fib_382, bb_upper)  # دخول عند مستويات فيبوناتشي
                ]
                
                # أهداف الربح (للبيع)
                support = indicators.get('support', current_price * 0.95)
                fib_618 = indicators.get('fib_618', current_price * 0.97)
                bb_lower = indicators.get('bb_lower', current_price * 0.96)
                
                signals['take_profits'] = [
                    current_price * 0.98,  # هدف سريع
                    max(support, fib_618),  # هدف متوسط
                    min(support, bb_lower),  # هدف رئيسي
                    current_price * 0.90  # هدف طموح
                ]
                
                # وقف الخسارة
                pivot_high = indicators.get('pivot_high', current_price * 1.08)
                signals['stop_loss'] = min(pivot_high, current_price * 1.05)
            
            else:
                signals['action'] = 'HOLD'
                signals['confidence'] = 30
                signals['entry_points'] = [current_price]
                signals['stop_loss'] = current_price * 0.95
                signals['take_profits'] = [current_price * 1.03]
            
            # حساب نسبة المخاطرة للعائد
            if signals['stop_loss'] and signals['take_profits']:
                risk = abs(current_price - signals['stop_loss'])
                reward = abs(signals['take_profits'][0] - current_price)
                signals['risk_reward'] = reward / risk if risk > 0 else 0
            
            # تحديد حجم المركز بناءً على التقلبات
            volatility = market_analysis.get('volatility', 'MEDIUM')
            if volatility == 'HIGH':
                signals['position_size'] = 'SMALL'
            elif volatility == 'LOW':
                signals['position_size'] = 'LARGE'
            else:
                signals['position_size'] = 'MEDIUM'
            
            # تحديد الإطار الزمني المناسب
            if trend_strength > 3:
                signals['timeframe_recommendation'] = '1d'
            elif trend_strength > 1:
                signals['timeframe_recommendation'] = '4h'
            else:
                signals['timeframe_recommendation'] = '1h'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'entry_points': [indicators.get('current_price', 0)],
                'stop_loss': indicators.get('current_price', 0) * 0.95,
                'take_profits': [indicators.get('current_price', 0) * 1.03],
                'risk_reward': 1,
                'timeframe_recommendation': '4h',
                'position_size': 'MEDIUM'
            }

    def comprehensive_analysis(self, symbol: str) -> Dict:
        """تحليل شامل متقدم مع معالجة محسنة للأخطاء"""
        analysis_results = {}
        
        for timeframe, config in self.timeframes.items():
            try:
                logger.info(f"Analyzing {symbol} on {timeframe}")
                
                # محاولة جلب البيانات مع retry
                data = None
                for attempt in range(2):  # محاولتان
                    data = self.get_price_data(symbol, timeframe, config['limit'])
                    if data and len(data) >= 20:  # تقليل الحد الأدنى
                        break
                    elif attempt == 0:
                        logger.warning(f"Retrying {symbol} {timeframe}")
                        import time
                        time.sleep(1)
                
                if not data or len(data) < 20:
                    logger.warning(f"Skipping {symbol} {timeframe} - insufficient data")
                    continue
                
                # حساب المؤشرات المتقدمة
                try:
                    indicators = self.calculate_advanced_indicators(data)
                    if not indicators:
                        logger.warning(f"No indicators calculated for {symbol} {timeframe}")
                        continue
                except Exception as ind_error:
                    logger.error(f"Indicators error for {symbol} {timeframe}: {ind_error}")
                    continue
                
                # تحليل هيكل السوق
                try:
                    market_analysis = self.analyze_market_structure(indicators)
                except Exception as market_error:
                    logger.error(f"Market analysis error for {symbol} {timeframe}: {market_error}")
                    market_analysis = {
                        'trend_direction': 'NEUTRAL',
                        'momentum': 'NEUTRAL',
                        'volatility': 'MEDIUM',
                        'volume_analysis': 'NORMAL',
                        'signals': []
                    }
                
                # توليد إشارات التداول
                try:
                    trading_signals = self.generate_trading_signals(indicators, market_analysis)
                except Exception as signals_error:
                    logger.error(f"Trading signals error for {symbol} {timeframe}: {signals_error}")
                    trading_signals = {
                        'action': 'HOLD',
                        'confidence': 0,
                        'entry_points': [indicators.get('current_price', 0)],
                        'stop_loss': indicators.get('current_price', 0) * 0.95,
                        'take_profits': [indicators.get('current_price', 0) * 1.03],
                        'risk_reward': 1,
                        'timeframe_recommendation': timeframe,
                        'position_size': 'MEDIUM'
                    }
                
                analysis_results[timeframe] = {
                    'timeframe_name': config['name'],
                    'indicators': indicators,
                    'market_analysis': market_analysis,
                    'trading_signals': trading_signals,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Completed analysis for {symbol} {timeframe}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                continue
        
        if not analysis_results:
            logger.error(f"No analysis results for {symbol}")
        
        return analysis_results

class ProfessionalCryptoBot:
    """بوت التداول الاحترافي المتقدم"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.webhook_url = os.getenv('WEBHOOK_URL')
        self.analyzer = AdvancedTechnicalAnalyzer()
        self.user_settings = {}
        self.application = None
        
        if not self.token:
            raise ValueError("BOT_TOKEN environment variable is required")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية المحسن"""
        welcome_message = """
🚀 *مرحباً بك في بوت التحليل الفني الاحترافي*

🎯 *ميزات متقدمة:*
• تحليل فني شامل على 5 فريمات
• مؤشرات احترافية (RSI, MACD, ADX, Stochastic)
• نقاط دخول وخروج دقيقة
• حساب المخاطرة والعائد
• تحليل الحجم والزخم
• مستويات فيبوناتشي والدعم/المقاومة

📊 *الإطارات الزمنية:*
15m | 1h | 4h | 1d | 1w

🎛️ *الأوامر الرئيسية:*
/pro BTC - تحليل احترافي شامل
/quick ETH - تحليل سريع
/compare BTC ETH - مقارنة عملتين
/alerts BTC - تنبيهات ذكية
/settings - إعدادات شخصية

💡 *نصيحة:* ابدأ بـ /pro BTC للحصول على تحليل متكامل

⚡ *تحديثات لحظية 24/7*
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل احترافي", callback_data="pro_analysis")],
            [InlineKeyboardButton("⚡ تحليل سريع", callback_data="quick_analysis")],
            [InlineKeyboardButton("📈 مقارنة عملات", callback_data="compare_coins")],
            [InlineKeyboardButton("🔔 إعداد التنبيهات", callback_data="setup_alerts")],
            [InlineKeyboardButton("⚙️ الإعدادات", callback_data="user_settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown', reply_markup=reply_markup)

    async def pro_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """التحليل الاحترافي الشامل"""
        try:
            if not context.args:
                await update.message.reply_text("❌ يرجى تحديد رمز العملة\nمثال: /pro BTC")
                return
                
            symbol = context.args[0].upper()
            
            # رسالة انتظار متقدمة
            waiting_msg = await update.message.reply_text(
                f"🔍 *تحليل احترافي لـ {symbol}*\n\n"
                f"📊 جاري تحليل 5 إطارات زمنية...\n"
                f"🧮 حساب 15+ مؤشر فني...\n"
                f"🎯 توليد إشارات التداول...\n\n"
                f"⏳ يرجى الانتظار 10-15 ثانية...",
                parse_mode='Markdown'
            )
            
            # إجراء التحليل الشامل
            try:
                analysis = self.analyzer.comprehensive_analysis(symbol)
            except Exception as analysis_error:
                logger.error(f"Analysis error: {analysis_error}")
                analysis = None
            
            if not analysis:
                await waiting_msg.edit_text(f"❌ لم أتمكن من الحصول على بيانات {symbol}")
                return
                
            # تنسيق التقرير الاحترافي
            report = self.format_professional_report(symbol, analysis)
            
            # أزرار التفاعل المتقدمة
            keyboard = [
                [InlineKeyboardButton("📊 تفاصيل المؤشرات", callback_data=f"details_{symbol}")],
                [InlineKeyboardButton("🎯 إشارات التداول", callback_data=f"signals_{symbol}")],
                [InlineKeyboardButton("📈 مقارنة الفريمات", callback_data=f"timeframes_{symbol}")],
                [InlineKeyboardButton("🔔 إضافة تنبيه", callback_data=f"alert_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_pro_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in pro analysis: {e}")
            try:
                await waiting_msg.edit_text(f"❌ حدث خطأ في التحليل الاحترافي لـ {symbol}")
            except:
                await update.message.reply_text(f"❌ حدث خطأ في التحليل الاحترافي لـ {symbol}")

    def format_professional_report(self, symbol: str, analysis: Dict) -> str:
        """تنسيق التقرير الاحترافي"""
        try:
            if not analysis:
                return "❌ لا توجد بيانات متاحة للتحليل"
            
            # الحصول على أهم الإطارات الزمنية
            primary_timeframes = ['1d', '4h', '1h']
            available_timeframes = [tf for tf in primary_timeframes if tf in analysis]
            
            if not available_timeframes:
                available_timeframes = list(analysis.keys())[:3]
            
            # البيانات الأساسية من الإطار اليومي أو المتاح
            main_tf = available_timeframes[0] if available_timeframes else list(analysis.keys())[0]
            main_data = analysis[main_tf]
            
            current_price = main_data['indicators'].get('current_price', 0)
            price_change = main_data['indicators'].get('price_change_24h', 0)
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.8f}"
            elif current_price < 1:
                price_str = f"${current_price:.6f}"
            elif current_price < 100:
                price_str = f"${current_price:.4f}"
            else:
                price_str = f"${current_price:,.2f}"
            
            # رمز التغيير
            change_symbol = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            change_color = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪"
            
            # بداية التقرير
            report = f"""
🎯 *التحليل الاحترافي لـ {symbol}/USDT*
━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* {price_str}
{change_symbol} *التغيير 24س:* {change_color} {price_change:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            
            # تحليل الاتجاه الإجمالي
            trend_summary = self.get_overall_trend(analysis)
            report += f"""
📊 *التحليل الإجمالي:*

🎯 *الاتجاه العام:* {trend_summary['direction']}
💪 *قوة الاتجاه:* {trend_summary['strength']}
🎲 *التوصية:* {trend_summary['recommendation']}
🔮 *مستوى الثقة:* {trend_summary['confidence']}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            
            # تحليل الإطارات الزمنية
            report += "⏰ *تحليل الإطارات الزمنية:*\n\n"
            
            for tf in available_timeframes[:3]:
                if tf not in analysis:
                    continue
                    
                tf_data = analysis[tf]
                tf_name = tf_data.get('timeframe_name', tf)
                market_analysis = tf_data.get('market_analysis', {})
                trading_signals = tf_data.get('trading_signals', {})
                
                trend_dir = market_analysis.get('trend_direction', 'NEUTRAL')
                action = trading_signals.get('action', 'HOLD')
                confidence = trading_signals.get('confidence', 0)
                
                # أيقونات الاتجاه
                trend_icon = self.get_trend_icon(trend_dir)
                action_icon = self.get_action_icon(action)
                
                report += f"🔸 *{tf_name}:* {trend_icon} {action_icon} ({confidence:.0f}%)\n"
            
            # أهم المؤشرات الفنية
            indicators = main_data.get('indicators', {})
            report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 *المؤشرات الفنية الرئيسية:*

🔹 *RSI(14):* {indicators.get('rsi_14', 0):.1f} {self.get_rsi_status(indicators.get('rsi_14', 50))}
🔹 *MACD:* {self.get_macd_status(indicators)}
🔹 *ADX:* {indicators.get('adx', 0):.1f} {self.get_adx_status(indicators.get('adx', 25))}
🔹 *Stochastic:* {indicators.get('stoch_k', 0):.1f} {self.get_stoch_status(indicators.get('stoch_k', 50))}
            """
            
            # مستويات الدعم والمقاومة
            support = indicators.get('support', current_price * 0.95)
            resistance = indicators.get('resistance', current_price * 1.05)
            
            report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 *مستويات حرجة:*

🟢 *الدعم الرئيسي:* ${support:.6f}
🔴 *المقاومة الرئيسية:* ${resistance:.6f}
📊 *المسافة للدعم:* {((current_price - support) / current_price * 100):.1f}%
📊 *المسافة للمقاومة:* {((resistance - current_price) / current_price * 100):.1f}%
            """
            
            # إشارات التداول المفصلة
            trading_signals = main_data.get('trading_signals', {})
            if trading_signals.get('action') != 'HOLD':
                report += self.format_trading_signals(trading_signals, current_price)
            
            # خلاصة وتوصيات
            market_analysis = main_data.get('market_analysis', {})
            signals_list = market_analysis.get('signals', [])
            
            if signals_list:
                report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 *الإشارات المهمة:*

"""
                for signal in signals_list[:4]:  # أهم 4 إشارات
                    report += f"• {signal}\n"
            
            # تذييل التقرير
            report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ *تنبيه:* هذا تحليل فني تعليمي
🕒 *آخر تحديث:* {datetime.now().strftime('%H:%M:%S')}
⚡ *تحديث آلي كل 15 دقيقة*
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting professional report: {e}")
            return f"❌ حدث خطأ في تنسيق تقرير {symbol}"

    def get_overall_trend(self, analysis: Dict) -> Dict:
        """تحديد الاتجاه الإجمالي"""
        try:
            timeframes = ['1d', '4h', '1h']
            available_tfs = [tf for tf in timeframes if tf in analysis]
            
            if not available_tfs:
                available_tfs = list(analysis.keys())[:3]
            
            trend_scores = []
            total_confidence = 0
            
            for tf in available_tfs:
                if tf not in analysis:
                    continue
                    
                market_analysis = analysis[tf].get('market_analysis', {})
                trading_signals = analysis[tf].get('trading_signals', {})
                
                trend_dir = market_analysis.get('trend_direction', 'NEUTRAL')
                confidence = trading_signals.get('confidence', 0)
                
                # تحويل الاتجاه إلى نقاط
                if trend_dir == 'STRONG_BULLISH':
                    score = 2
                elif trend_dir == 'BULLISH':
                    score = 1
                elif trend_dir == 'STRONG_BEARISH':
                    score = -2
                elif trend_dir == 'BEARISH':
                    score = -1
                else:
                    score = 0
                
                # ترجيح النتيجة بناءً على الإطار الزمني
                weight = 3 if tf == '1d' else 2 if tf == '4h' else 1
                trend_scores.append(score * weight)
                total_confidence += confidence * weight
            
            if not trend_scores:
                return {
                    'direction': '⚪ غير محدد',
                    'strength': 'ضعيف',
                    'recommendation': '⏸️ انتظار',
                    'confidence': 0
                }
            
            avg_score = sum(trend_scores) / len(trend_scores)
            avg_confidence = total_confidence / sum([3, 2, 1][:len(available_tfs)])
            
            # تحديد الاتجاه
            if avg_score > 1.5:
                direction = "🚀 صاعد قوي"
                recommendation = "🟢 شراء قوي"
                strength = "قوي جداً"
            elif avg_score > 0.5:
                direction = "📈 صاعد"
                recommendation = "🔵 شراء"
                strength = "قوي"
            elif avg_score < -1.5:
                direction = "📉 هابط قوي"
                recommendation = "🔴 بيع قوي"
                strength = "قوي جداً"
            elif avg_score < -0.5:
                direction = "📉 هابط"
                recommendation = "🟠 بيع"
                strength = "متوسط"
            else:
                direction = "➡️ محايد"
                recommendation = "⚪ انتظار"
                strength = "ضعيف"
            
            return {
                'direction': direction,
                'strength': strength,
                'recommendation': recommendation,
                'confidence': min(int(avg_confidence), 95)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall trend: {e}")
            return {
                'direction': '⚪ غير محدد',
                'strength': 'ضعيف',
                'recommendation': '⏸️ انتظار',
                'confidence': 0
            }

    def get_trend_icon(self, trend_direction: str) -> str:
        """الحصول على أيقونة الاتجاه"""
        icons = {
            'STRONG_BULLISH': '🚀',
            'BULLISH': '📈',
            'NEUTRAL': '➡️',
            'BEARISH': '📉',
            'STRONG_BEARISH': '💥'
        }
        return icons.get(trend_direction, '❓')

    def get_action_icon(self, action: str) -> str:
        """الحصول على أيقونة الإجراء"""
        icons = {
            'BUY': '🟢 شراء',
            'SELL': '🔴 بيع',
            'HOLD': '⚪ انتظار'
        }
        return icons.get(action, '❓')

    def get_rsi_status(self, rsi: float) -> str:
        """حالة RSI"""
        if rsi < 30:
            return "🔵 شراء مفرط"
        elif rsi > 70:
            return "🔴 بيع مفرط"
        elif 45 <= rsi <= 55:
            return "⚪ محايد"
        elif rsi < 45:
            return "🔵 ميول شرائية"
        else:
            return "🔴 ميول بيعية"

    def get_macd_status(self, indicators: Dict) -> str:
        """حالة MACD"""
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        histogram = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and histogram > 0:
            return "🟢 إيجابي"
        elif macd < macd_signal and histogram < 0:
            return "🔴 سلبي"
        else:
            return "⚪ محايد"

    def get_adx_status(self, adx: float) -> str:
        """حالة ADX"""
        if adx > 50:
            return "💪 قوي جداً"
        elif adx > 25:
            return "💪 قوي"
        elif adx > 20:
            return "📊 متوسط"
        else:
            return "😴 ضعيف"

    def get_stoch_status(self, stoch: float) -> str:
        """حالة Stochastic"""
        if stoch < 20:
            return "🔵 شراء مفرط"
        elif stoch > 80:
            return "🔴 بيع مفرط"
        else:
            return "⚪ طبيعي"

    def format_trading_signals(self, trading_signals: Dict, current_price: float) -> str:
        """تنسيق إشارات التداول"""
        try:
            action = trading_signals.get('action', 'HOLD')
            confidence = trading_signals.get('confidence', 0)
            entry_points = trading_signals.get('entry_points', [])
            take_profits = trading_signals.get('take_profits', [])
            stop_loss = trading_signals.get('stop_loss', 0)
            risk_reward = trading_signals.get('risk_reward', 0)
            
            if action == 'HOLD':
                return ""
            
            action_text = "🟢 الشراء" if action == 'BUY' else "🔴 البيع"
            
            signals_text = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 *إشارات التداول - {action_text}:*

💪 *مستوى الثقة:* {confidence:.0f}%
⚖️ *نسبة المخاطرة/العائد:* 1:{risk_reward:.1f}

📍 *نقاط الدخول:*
"""
            
            for i, entry in enumerate(entry_points[:3], 1):
                if entry > 0:
                    distance = ((entry - current_price) / current_price * 100)
                    signals_text += f"🔸 {i}. ${entry:.6f} ({distance:+.1f}%)\n"
            
            if take_profits:
                signals_text += "\n🎯 *أهداف الربح:*\n"
                for i, target in enumerate(take_profits[:4], 1):
                    if target > 0:
                        profit = ((target - current_price) / current_price * 100)
                        if action == 'SELL':
                            profit = -profit
                        signals_text += f"🥇 {i}. ${target:.6f} ({profit:+.1f}%)\n"
            
            if stop_loss > 0:
                loss = ((stop_loss - current_price) / current_price * 100)
                if action == 'SELL':
                    loss = -loss
                signals_text += f"\n🛑 *وقف الخسارة:* ${stop_loss:.6f} ({loss:+.1f}%)"
            
            return signals_text
            
        except Exception as e:
            logger.error(f"Error formatting trading signals: {e}")
            return ""

    async def quick_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """التحليل السريع"""
        try:
            if not context.args:
                await update.message.reply_text("❌ يرجى تحديد رمز العملة\nمثال: /quick BTC")
                return
                
            symbol = context.args[0].upper()
            
            waiting_msg = await update.message.reply_text(f"⚡ تحليل سريع لـ {symbol}...")
            
            # إجراء التحليل السريع
            try:
                data = self.analyzer.get_price_data(symbol, '1h', 100)
                if not data:
                    await waiting_msg.edit_text(f"❌ لا توجد بيانات لـ {symbol}")
                    return
                
                indicators = self.analyzer.calculate_advanced_indicators(data)
                market_analysis = self.analyzer.analyze_market_structure(indicators)
                trading_signals = self.analyzer.generate_trading_signals(indicators, market_analysis)
            except Exception as analysis_error:
                logger.error(f"Quick analysis error: {analysis_error}")
                await waiting_msg.edit_text(f"❌ حدث خطأ في التحليل السريع لـ {symbol}")
                return
            
            # تنسيق التقرير السريع
            report = self.format_quick_report(symbol, indicators, market_analysis, trading_signals)
            
            keyboard = [
                [InlineKeyboardButton("📊 تحليل شامل", callback_data=f"pro_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_quick_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            await update.message.reply_text(f"❌ حدث خطأ في التحليل السريع لـ {symbol}")

    def format_quick_report(self, symbol: str, indicators: Dict, market_analysis: Dict, trading_signals: Dict) -> str:
        """تنسيق التقرير السريع"""
        try:
            current_price = indicators.get('current_price', 0)
            price_change = indicators.get('price_change_24h', 0)
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.8f}"
            elif current_price < 1:
                price_str = f"${current_price:.6f}"
            else:
                price_str = f"${current_price:.2f}"
            
            # الاتجاه والإجراء
            trend_direction = market_analysis.get('trend_direction', 'NEUTRAL')
            action = trading_signals.get('action', 'HOLD')
            confidence = trading_signals.get('confidence', 0)
            
            trend_icon = self.get_trend_icon(trend_direction)
            action_icon = self.get_action_icon(action)
            change_icon = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            
            report = f"""
⚡ *تحليل سريع - {symbol}/USDT*

💰 *السعر:* {price_str} {change_icon} {price_change:+.1f}%

🎯 *التوصية:* {action_icon}
💪 *الثقة:* {confidence:.0f}%
📊 *الاتجاه:* {trend_icon}

📈 *المؤشرات:*
• RSI: {indicators.get('rsi_14', 0):.0f} {self.get_rsi_status(indicators.get('rsi_14', 50))}
• MACD: {self.get_macd_status(indicators)}

🎯 *المستويات:*
• دعم: ${indicators.get('support', 0):.6f}
• مقاومة: ${indicators.get('resistance', 0):.6f}

🕒 *{datetime.now().strftime('%H:%M:%S')}*
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting quick report: {e}")
            return f"❌ خطأ في تنسيق التقرير لـ {symbol}"

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأزرار المحسن مع جميع الوظائف"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            user_id = update.effective_user.id
            
            # التحليل الاحترافي
            if data.startswith("pro_"):
                symbol = data.split("_", 1)[1]
                context.args = [symbol]
                # تحديث الرسالة لإظهار التحليل قيد التقدم
                await query.edit_message_text(
                    f"🔍 جاري التحليل الاحترافي لـ {symbol}...\nيرجى الانتظار...",
                    parse_mode='Markdown'
                )
                # تشغيل التحليل
                update.message = query.message  # محاكاة message للـ command handler
                await self.pro_analysis_command(update, context)
                
            # تحديث التحليل الاحترافي
            elif data.startswith("refresh_pro_"):
                symbol = data.split("_", 2)[2]
                context.args = [symbol]
                await query.edit_message_text(
                    f"🔄 تحديث التحليل الاحترافي لـ {symbol}...",
                    parse_mode='Markdown'
                )
                update.message = query.message
                await self.pro_analysis_command(update, context)
                
            # التحليل السريع
            elif data.startswith("refresh_quick_"):
                symbol = data.split("_", 2)[2]
                context.args = [symbol]
                await query.edit_message_text(f"⚡ تحديث التحليل السريع لـ {symbol}...")
                update.message = query.message
                await self.quick_analysis_command(update, context)
                
            # تفاصيل المؤشرات
            elif data.startswith("details_"):
                symbol = data.split("_", 1)[1]
                await self.show_indicators_details(query, symbol)
                
            # إشارات التداول
            elif data.startswith("signals_"):
                symbol = data.split("_", 1)[1]
                await self.show_trading_signals_details(query, symbol)
                
            # مقارنة الفريمات
            elif data.startswith("timeframes_"):
                symbol = data.split("_", 1)[1]
                await self.show_timeframes_comparison(query, symbol)
                
            # إضافة تنبيه
            elif data.startswith("alert_"):
                symbol = data.split("_", 1)[1]
                await self.setup_price_alert(query, symbol, user_id)
                
            # القائمة الرئيسية
            elif data == "pro_analysis":
                await query.edit_message_text(
                    "📊 *التحليل الاحترافي*\n\n"
                    "أرسل رمز العملة للحصول على تحليل شامل:\n\n"
                    "🔹 مثال: `BTC` أو `ETH` أو `ADA`\n"
                    "🔹 أو استخدم: `/pro BTC`\n\n"
                    "📈 *يشمل:*\n"
                    "• 5 إطارات زمنية\n"
                    "• 15+ مؤشر فني\n"
                    "• إشارات دخول وخروج\n"
                    "• حساب المخاطرة والعائد",
                    parse_mode='Markdown'
                )
                
            elif data == "quick_analysis":
                await query.edit_message_text(
                    "⚡ *التحليل السريع*\n\n"
                    "أرسل رمز العملة للحصول على تحليل سريع:\n\n"
                    "🔹 مثال: `BTC` أو `ETH` أو `ADA`\n"
                    "🔹 أو استخدم: `/quick BTC`\n\n"
                    "📊 *يشمل:*\n"
                    "• تحليل الإطار الساعي\n"
                    "• المؤشرات الرئيسية\n"
                    "• التوصية السريعة",
                    parse_mode='Markdown'
                )
                
            elif data == "compare_coins":
                await query.edit_message_text(
                    "📈 *مقارنة العملات*\n\n"
                    "قريباً: مقارنة شاملة بين عملتين\n\n"
                    "📊 *ستشمل:*\n"
                    "• مقارنة الأداء\n"
                    "• مقارنة المؤشرات\n"
                    "• التوصيات النسبية\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "setup_alerts":
                await query.edit_message_text(
                    "🔔 *إعداد التنبيهات*\n\n"
                    "قريباً: تنبيهات ذكية للأسعار والإشارات\n\n"
                    "📱 *ستشمل:*\n"
                    "• تنبيهات الأسعار\n"
                    "• تنبيهات المؤشرات\n"
                    "• تنبيهات الإشارات\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "user_settings":
                await query.edit_message_text(
                    "⚙️ *الإعدادات الشخصية*\n\n"
                    "قريباً: إعدادات مخصصة للمستخدم\n\n"
                    "🎛️ *ستشمل:*\n"
                    "• تفضيلات العرض\n"
                    "• إعدادات التنبيهات\n"
                    "• المؤشرات المفضلة\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "help":
                await self.help_command(update, context)
                
            # الرجوع للقائمة الرئيسية
            elif data == "back_to_main":
                update.message = query.message
                await self.start_command(update, context)
                
            # مقارنة العملات
            elif data.startswith("compare_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    symbol1, symbol2 = parts[1], parts[2]
                    context.args = [symbol1, symbol2]
                    update.message = query.message
                    await self.compare_command(update, context)
                    
            # إعدادات مختلفة
            elif data == "display_settings":
                await query.edit_message_text(
                    "🎨 *تفضيلات العرض*\n\n"
                    "قريباً: تخصيص طريقة عرض التحاليل\n\n"
                    "📱 *سيشمل:*\n"
                    "• اختيار الألوان\n"
                    "• حجم الخط\n"
                    "• نوع الرسوم البيانية\n"
                    "• ترتيب المعلومات\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "alert_settings":
                await query.edit_message_text(
                    "🔔 *إعدادات التنبيهات*\n\n"
                    "قريباً: تخصيص التنبيهات الذكية\n\n"
                    "📱 *سيشمل:*\n"
                    "• أوقات التنبيهات\n"
                    "• أنواع التنبيهات\n"
                    "• مستويات الأهمية\n"
                    "• طرق الإشعار\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "indicator_settings":
                await query.edit_message_text(
                    "📊 *المؤشرات المفضلة*\n\n"
                    "قريباً: تخصيص المؤشرات المعروضة\n\n"
                    "📈 *سيشمل:*\n"
                    "• اختيار المؤشرات\n"
                    "• فترات المؤشرات\n"
                    "• حساسية الإشارات\n"
                    "• ترتيب الأولوية\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
            elif data == "locale_settings":
                await query.edit_message_text(
                    "🌐 *اللغة والوقت*\n\n"
                    "قريباً: تخصيص اللغة والمنطقة الزمنية\n\n"
                    "🌍 *سيشمل:*\n"
                    "• اختيار اللغة\n"
                    "• المنطقة الزمنية\n"
                    "• تنسيق التاريخ\n"
                    "• عملة العرض\n\n"
                    "⏳ قيد التطوير...",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in button callback: {e}")
            try:
                await query.edit_message_text(
                    "❌ حدث خطأ في معالجة الطلب\n"
                    "يرجى المحاولة مرة أخرى أو استخدام الأوامر النصية\n\n"
                    "مثال: /pro BTC أو /quick ETH"
                )
            except:
                pass

    async def show_indicators_details(self, query, symbol: str):
        """عرض تفاصيل المؤشرات"""
        try:
            await query.edit_message_text(
                f"📊 *تفاصيل المؤشرات لـ {symbol}*\n\n"
                f"🔍 جاري تحليل المؤشرات بالتفصيل...",
                parse_mode='Markdown'
            )
            
            # جلب بيانات سريعة للإطار اليومي
            data = self.analyzer.get_price_data(symbol, '1d', 100)
            if not data:
                await query.edit_message_text(f"❌ لا توجد بيانات متاحة لـ {symbol}")
                return
                
            indicators = self.analyzer.calculate_advanced_indicators(data)
            if not indicators:
                await query.edit_message_text(f"❌ لا يمكن حساب المؤشرات لـ {symbol}")
                return
            
            details_text = f"""
📊 *تفاصيل المؤشرات - {symbol}*
━━━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* ${indicators.get('current_price', 0):.6f}

📈 *المتوسطات المتحركة:*
🔸 SMA(20): ${indicators.get('sma_20', 0):.6f}
🔸 SMA(50): ${indicators.get('sma_50', 0):.6f}
🔸 EMA(9): ${indicators.get('ema_9', 0):.6f}
🔸 EMA(21): ${indicators.get('ema_21', 0):.6f}

🎯 *مؤشرات الزخم:*
🔸 RSI(14): {indicators.get('rsi_14', 0):.1f}
🔸 RSI(21): {indicators.get('rsi_21', 0):.1f}
🔸 Stochastic K: {indicators.get('stoch_k', 0):.1f}
🔸 Stochastic D: {indicators.get('stoch_d', 0):.1f}

⚡ *MACD:*
🔸 MACD Line: {indicators.get('macd', 0):.6f}
🔸 Signal Line: {indicators.get('macd_signal', 0):.6f}
🔸 Histogram: {indicators.get('macd_histogram', 0):.6f}

💪 *قوة الاتجاه:*
🔸 ADX: {indicators.get('adx', 0):.1f}
🔸 +DI: {indicators.get('plus_di', 0):.1f}
🔸 -DI: {indicators.get('minus_di', 0):.1f}

🎪 *نطاقات بولينجر:*
🔸 العلوي: ${indicators.get('bb_upper', 0):.6f}
🔸 الوسط: ${indicators.get('bb_middle', 0):.6f}
🔸 السفلي: ${indicators.get('bb_lower', 0):.6f}

📊 *الدعم والمقاومة:*
🔸 المقاومة: ${indicators.get('resistance', 0):.6f}
🔸 الدعم: ${indicators.get('support', 0):.6f}

🕒 *{datetime.now().strftime('%H:%M:%S')}*
            """
            
            keyboard = [
                [InlineKeyboardButton("🎯 إشارات التداول", callback_data=f"signals_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"details_{symbol}")],
                [InlineKeyboardButton("🔙 رجوع", callback_data=f"pro_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(details_text, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error showing indicators details: {e}")
            await query.edit_message_text(f"❌ حدث خطأ في عرض تفاصيل المؤشرات لـ {symbol}")

    async def show_trading_signals_details(self, query, symbol: str):
        """عرض تفاصيل إشارات التداول"""
        try:
            await query.edit_message_text(
                f"🎯 *إشارات التداول لـ {symbol}*\n\n"
                f"🔍 جاري تحليل الإشارات...",
                parse_mode='Markdown'
            )
            
            # تحليل سريع للحصول على الإشارات
            data = self.analyzer.get_price_data(symbol, '4h', 100)
            if not data:
                await query.edit_message_text(f"❌ لا توجد بيانات متاحة لـ {symbol}")
                return
                
            indicators = self.analyzer.calculate_advanced_indicators(data)
            market_analysis = self.analyzer.analyze_market_structure(indicators)
            trading_signals = self.analyzer.generate_trading_signals(indicators, market_analysis)
            
            current_price = indicators.get('current_price', 0)
            action = trading_signals.get('action', 'HOLD')
            confidence = trading_signals.get('confidence', 0)
            entry_points = trading_signals.get('entry_points', [])
            take_profits = trading_signals.get('take_profits', [])
            stop_loss = trading_signals.get('stop_loss', 0)
            risk_reward = trading_signals.get('risk_reward', 0)
            
            action_icon = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "⚪"
            action_text = "شراء" if action == 'BUY' else "بيع" if action == 'SELL' else "انتظار"
            
            signals_text = f"""
🎯 *إشارات التداول المفصلة - {symbol}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* ${current_price:.6f}

{action_icon} *التوصية:* {action_text}
💪 *مستوى الثقة:* {confidence:.0f}%
⚖️ *نسبة المخاطرة/العائد:* 1:{risk_reward:.1f}

📍 *نقاط الدخول المقترحة:*
            """
            
            for i, entry in enumerate(entry_points[:3], 1):
                if entry > 0:
                    distance = ((entry - current_price) / current_price * 100)
                    signals_text += f"🔸 {i}. ${entry:.6f} ({distance:+.1f}%)\n"
            
            if take_profits:
                signals_text += "\n🎯 *أهداف الربح:*\n"
                for i, target in enumerate(take_profits[:3], 1):
                    if target > 0:
                        profit = ((target - current_price) / current_price * 100)
                        if action == 'SELL':
                            profit = -profit
                        signals_text += f"🥇 {i}. ${target:.6f} ({profit:+.1f}%)\n"
            
            if stop_loss > 0:
                loss = ((stop_loss - current_price) / current_price * 100)
                if action == 'SELL':
                    loss = -loss
                signals_text += f"\n🛑 *وقف الخسارة:* ${stop_loss:.6f} ({loss:+.1f}%)"
            
            # الإشارات الداعمة
            signals_list = market_analysis.get('signals', [])
            if signals_list:
                signals_text += "\n\n🔍 *الإشارات الداعمة:*\n"
                for signal in signals_list[:3]:
                    signals_text += f"• {signal}\n"
            
            signals_text += f"\n🕒 *{datetime.now().strftime('%H:%M:%S')}*"
            
            keyboard = [
                [InlineKeyboardButton("📊 تفاصيل المؤشرات", callback_data=f"details_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"signals_{symbol}")],
                [InlineKeyboardButton("🔙 رجوع", callback_data=f"pro_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(signals_text, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error showing trading signals: {e}")
            await query.edit_message_text(f"❌ حدث خطأ في عرض إشارات التداول لـ {symbol}")

    async def show_timeframes_comparison(self, query, symbol: str):
        """عرض مقارنة الإطارات الزمنية"""
        try:
            await query.edit_message_text(
                f"📈 *مقارنة الفريمات لـ {symbol}*\n\n"
                f"🔍 جاري تحليل الإطارات الزمنية...",
                parse_mode='Markdown'
            )
            
            # تحليل مبسط لثلاث إطارات رئيسية
            timeframes = ['1h', '4h', '1d']
            results = {}
            
            for tf in timeframes:
                try:
                    data = self.analyzer.get_price_data(symbol, tf, 50)
                    if data and len(data) >= 20:
                        indicators = self.analyzer.calculate_advanced_indicators(data)
                        market_analysis = self.analyzer.analyze_market_structure(indicators)
                        trading_signals = self.analyzer.generate_trading_signals(indicators, market_analysis)
                        
                        results[tf] = {
                            'trend': market_analysis.get('trend_direction', 'NEUTRAL'),
                            'action': trading_signals.get('action', 'HOLD'),
                            'confidence': trading_signals.get('confidence', 0),
                            'rsi': indicators.get('rsi_14', 50)
                        }
                except:
                    continue
            
            if not results:
                await query.edit_message_text(f"❌ لا يمكن الحصول على بيانات كافية لـ {symbol}")
                return
            
            comparison_text = f"""
📈 *مقارنة الإطارات الزمنية - {symbol}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            for tf, data in results.items():
                trend_icon = self.get_trend_icon(data['trend'])
                action_icon = self.get_action_icon(data['action'])
                tf_name = {'1h': 'ساعة', '4h': '4 ساعات', '1d': 'يوم'}[tf]
                
                comparison_text += f"""
⏰ *{tf_name} ({tf}):*
🎯 الاتجاه: {trend_icon}
💡 التوصية: {action_icon}
💪 الثقة: {data['confidence']:.0f}%
📊 RSI: {data['rsi']:.0f}

"""
            
            # تحليل التناسق
            actions = [data['action'] for data in results.values()]
            if all(action == 'BUY' for action in actions):
                consensus = "🟢 إجماع على الشراء"
            elif all(action == 'SELL' for action in actions):
                consensus = "🔴 إجماع على البيع"
            elif actions.count('BUY') > actions.count('SELL'):
                consensus = "🔵 ميول شرائية"
            elif actions.count('SELL') > actions.count('BUY'):
                consensus = "🟠 ميول بيعية"
            else:
                consensus = "⚪ إشارات متضاربة"
            
            comparison_text += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎲 *الإجماع العام:* {consensus}

🕒 *{datetime.now().strftime('%H:%M:%S')}*
            """
            
            keyboard = [
                [InlineKeyboardButton("🎯 إشارات التداول", callback_data=f"signals_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"timeframes_{symbol}")],
                [InlineKeyboardButton("🔙 رجوع", callback_data=f"pro_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(comparison_text, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error showing timeframes comparison: {e}")
            await query.edit_message_text(f"❌ حدث خطأ في مقارنة الإطارات الزمنية لـ {symbol}")

    async def setup_price_alert(self, query, symbol: str, user_id: int):
        """إعداد تنبيه السعر (مبدئي)"""
        try:
            # جلب السعر الحالي
            data = self.analyzer.get_price_data(symbol, '1h', 10)
            if not data:
                await query.edit_message_text(f"❌ لا يمكن الحصول على سعر {symbol}")
                return
                
            current_price = data[-1]['close']
            
            alert_text = f"""
🔔 *إعداد تنبيه السعر - {symbol}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* ${current_price:.6f}

📈 *مستويات مقترحة للتنبيه:*

🟢 *للشراء:*
• ${current_price * 0.95:.6f} (-5%)
• ${current_price * 0.90:.6f} (-10%)

🔴 *للبيع:*
• ${current_price * 1.05:.6f} (+5%)
• ${current_price * 1.10:.6f} (+10%)

⚠️ *ملاحظة:* 
ميزة التنبيهات قيد التطوير
ستكون متاحة قريباً مع:
• تنبيهات مخصصة
• تنبيهات المؤشرات
• تنبيهات الإشارات

🕒 *{datetime.now().strftime('%H:%M:%S')}*
            """
            
            keyboard = [
                [InlineKeyboardButton("🔙 رجوع", callback_data=f"pro_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(alert_text, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error setting up price alert: {e}")
            await query.edit_message_text(f"❌ حدث خطأ في إعداد التنبيه لـ {symbol}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر المساعدة المتقدم والشامل"""
        help_text = """
📚 *دليل البوت الاحترافي الشامل:*

🎯 *الأوامر الرئيسية:*
• `/start` - القائمة الرئيسية التفاعلية
• `/pro BTC` - تحليل احترافي شامل
• `/quick ETH` - تحليل سريع وفوري
• `/compare BTC ETH` - مقارنة عملتين
• `/help` - هذا الدليل
• `BTC` - إرسال رمز العملة مباشرة

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *التحليل الاحترافي يشمل:*

⏰ *5 إطارات زمنية:*
• 15 دقيقة - للتداول السريع
• ساعة واحدة - للتحليل قصير المدى  
• 4 ساعات - للتحليل متوسط المدى
• يوم واحد - للتحليل طويل المدى
• أسبوع - للاتجاه العام

📈 *15+ مؤشر فني متقدم:*
• RSI (14, 21) - القوة النسبية
• MACD + Signal + Histogram
• ADX + DI+ + DI- - قوة الاتجاه
• Stochastic %K %D
• Bollinger Bands
• EMA/SMA متعددة
• مستويات فيبوناتشي
• الدعم والمقاومة

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 *إشارات التداول الذكية:*

📍 *نقاط الدخول متعددة*
🎯 *أهداف ربح متدرجة*
🛑 *وقف خسارة محسوب*
⚖️ *نسبة المخاطرة/العائد*

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎲 *رموز التوصيات:*

🚀 صاعد قوي | 📈 صاعد | ➡️ محايد
📉 هابط | 💥 هابط قوي

🟢 شراء قوي | 🔵 شراء | ⚪ انتظار
🟠 بيع | 🔴 بيع قوي

━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ *تنبيه مهم:*
جميع التحاليل للأغراض التعليمية فقط
يرجى إجراء البحث الخاص قبل التداول

🤖 *البوت يعمل 24/7 مع تحديثات لحظية*
        """
        
        keyboard = [
            [InlineKeyboardButton("🚀 جرب الآن", callback_data="pro_analysis")],
            [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=reply_markup)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص المحسن"""
        try:
            text = update.message.text.upper().strip()
            
            # التحقق من رمز العملة
            if len(text) <= 10 and text.isalpha():
                # إظهار خيارات التحليل
                keyboard = [
                    [InlineKeyboardButton("📊 تحليل احترافي", callback_data=f"pro_{text}")],
                    [InlineKeyboardButton("⚡ تحليل سريع", callback_data=f"refresh_quick_{text}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    f"🎯 اختر نوع التحليل لـ *{text}*:",
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    "💡 أرسل رمز العملة للتحليل (مثل: BTC)\n"
                    "أو استخدم:\n"
                    "• /pro BTC - للتحليل الاحترافي\n"
                    "• /quick BTC - للتحليل السريع"
                )
                
        except Exception as e:
            logger.error(f"Error in text handler: {e}")

    async def compare_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مقارنة عملتين"""
        try:
            if len(context.args) < 2:
                await update.message.reply_text(
                    "❌ يرجى تحديد عملتين للمقارنة\n"
                    "مثال: /compare BTC ETH"
                )
                return
                
            symbol1 = context.args[0].upper()
            symbol2 = context.args[1].upper()
            
            waiting_msg = await update.message.reply_text(
                f"📈 مقارنة {symbol1} مع {symbol2}...\n"
                f"🔍 جاري التحليل..."
            )
            
            # تحليل العملتين
            results = {}
            for symbol in [symbol1, symbol2]:
                try:
                    data = self.analyzer.get_price_data(symbol, '1d', 50)
                    if data and len(data) >= 20:
                        indicators = self.analyzer.calculate_advanced_indicators(data)
                        market_analysis = self.analyzer.analyze_market_structure(indicators)
                        trading_signals = self.analyzer.generate_trading_signals(indicators, market_analysis)
                        
                        results[symbol] = {
                            'price': indicators.get('current_price', 0),
                            'change_24h': indicators.get('price_change_24h', 0),
                            'trend': market_analysis.get('trend_direction', 'NEUTRAL'),
                            'action': trading_signals.get('action', 'HOLD'),
                            'confidence': trading_signals.get('confidence', 0),
                            'rsi': indicators.get('rsi_14', 50),
                            'volume_ratio': indicators.get('volume_ratio', 1)
                        }
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            if len(results) != 2:
                await waiting_msg.edit_text("❌ لا يمكن الحصول على بيانات كافية للمقارنة")
                return
            
            # تنسيق تقرير المقارنة
            comparison_report = self.format_comparison_report(symbol1, symbol2, results)
            
            keyboard = [
                [InlineKeyboardButton(f"📊 تحليل {symbol1}", callback_data=f"pro_{symbol1}")],
                [InlineKeyboardButton(f"📊 تحليل {symbol2}", callback_data=f"pro_{symbol2}")],
                [InlineKeyboardButton("🔄 تحديث المقارنة", callback_data=f"compare_{symbol1}_{symbol2}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(comparison_report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in compare command: {e}")
            await update.message.reply_text("❌ حدث خطأ في المقارنة")

    def format_comparison_report(self, symbol1: str, symbol2: str, results: Dict) -> str:
        """تنسيق تقرير المقارنة"""
        try:
            data1 = results[symbol1]
            data2 = results[symbol2]
            
            # تحديد الفائز في كل معيار
            price_winner = symbol1 if data1['change_24h'] > data2['change_24h'] else symbol2
            momentum_winner = symbol1 if data1['confidence'] > data2['confidence'] else symbol2
            volume_winner = symbol1 if data1['volume_ratio'] > data2['volume_ratio'] else symbol2
            
            report = f"""
📈 *مقارنة شاملة: {symbol1} vs {symbol2}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 *الأسعار والأداء:*

🔸 *{symbol1}:* ${data1['price']:.6f}
   📊 التغيير 24س: {data1['change_24h']:+.2f}% {'🏆' if price_winner == symbol1 else ''}

🔸 *{symbol2}:* ${data2['price']:.6f}
   📊 التغيير 24س: {data2['change_24h']:+.2f}% {'🏆' if price_winner == symbol2 else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 *التوصيات والاتجاه:*

🔸 *{symbol1}:*
   {self.get_trend_icon(data1['trend'])} {self.get_action_icon(data1['action'])}
   💪 الثقة: {data1['confidence']:.0f}% {'🏆' if momentum_winner == symbol1 else ''}

🔸 *{symbol2}:*
   {self.get_trend_icon(data2['trend'])} {self.get_action_icon(data2['action'])}
   💪 الثقة: {data2['confidence']:.0f}% {'🏆' if momentum_winner == symbol2 else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *المؤشرات الفنية:*

🔸 *RSI(14):*
   • {symbol1}: {data1['rsi']:.0f} {self.get_rsi_status(data1['rsi'])}
   • {symbol2}: {data2['rsi']:.0f} {self.get_rsi_status(data2['rsi'])}

🔸 *نشاط التداول:*
   • {symbol1}: {data1['volume_ratio']:.1f}x {'🏆' if volume_winner == symbol1 else ''}
   • {symbol2}: {data2['volume_ratio']:.1f}x {'🏆' if volume_winner == symbol2 else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 *النتيجة الإجمالية:*

            """
            
            # حساب النقاط
            score1 = 0
            score2 = 0
            
            if price_winner == symbol1:
                score1 += 1
            else:
                score2 += 1
                
            if momentum_winner == symbol1:
                score1 += 1
            else:
                score2 += 1
                
            if volume_winner == symbol1:
                score1 += 1
            else:
                score2 += 1
            
            if score1 > score2:
                report += f"🥇 *الفائز:* {symbol1} ({score1}-{score2})\n"
                report += f"📈 {symbol1} يظهر أداءً أفضل حالياً"
            elif score2 > score1:
                report += f"🥇 *الفائز:* {symbol2} ({score2}-{score1})\n"
                report += f"📈 {symbol2} يظهر أداءً أفضل حالياً"
            else:
                report += f"🤝 *تعادل* ({score1}-{score2})\n"
                report += f"📊 أداء متقارب بين العملتين"
            
            report += f"\n🕒 *{datetime.now().strftime('%H:%M:%S')}*"
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting comparison report: {e}")
            return f"❌ خطأ في تنسيق تقرير المقارنة"

    async def alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إدارة التنبيهات"""
        await update.message.reply_text(
            "🔔 *نظام التنبيهات*\n\n"
            "قريباً: نظام تنبيهات متطور للعملات المشفرة\n\n"
            "📱 *الميزات القادمة:*\n"
            "• تنبيهات الأسعار المخصصة\n"
            "• تنبيهات عند كسر المقاومة/الدعم\n"
            "• تنبيهات المؤشرات الفنية\n"
            "• تنبيهات الإشارات القوية\n"
            "• جدولة التنبيهات\n\n"
            "⏳ *قيد التطوير...*\n"
            "🎯 سيكون متاحاً في التحديث القادم",
            parse_mode='Markdown'
        )

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إعدادات المستخدم"""
        keyboard = [
            [InlineKeyboardButton("🎨 تفضيلات العرض", callback_data="display_settings")],
            [InlineKeyboardButton("🔔 إعدادات التنبيهات", callback_data="alert_settings")],
            [InlineKeyboardButton("📊 المؤشرات المفضلة", callback_data="indicator_settings")],
            [InlineKeyboardButton("🌐 اللغة والوقت", callback_data="locale_settings")],
            [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ *الإعدادات الشخصية*\n\n"
            "🎛️ اختر الإعداد الذي تريد تخصيصه:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        """أمر المساعدة المتقدم"""
        help_text = """
📚 *دليل البوت الاحترافي:*

🎯 *الأوامر الرئيسية:*
• `/pro BTC` - تحليل احترافي شامل
• `/quick ETH` - تحليل سريع
• `BTC` - اختيار نوع التحليل

📊 *ميزات التحليل الاحترافي:*
• 5 إطارات زمنية (15m - 1w)
• 15+ مؤشر فني متقدم
• حساب نقاط الدخول والخروج
• تحليل المخاطرة والعائد
• مستويات فيبوناتشي

📈 *المؤشرات المغطاة:*
• RSI (14, 21)
• MACD + إشارة + هيستوغرام
• ADX + DI+ + DI-
• Stochastic %K %D
• Bollinger Bands
• EMA/SMA متعددة

🎯 *رموز التوصيات:*
🚀 صاعد قوي | 📈 صاعد | ➡️ محايد
📉 هابط | 💥 هابط قوي

🔔 قريباً: التنبيهات الذكية والمقارنات

⚠️ *تحذير مهم:*
جميع التحاليل للأغراض التعليمية
قم بإجراء بحثك الخاص قبل التداول

🤖 *البوت يعمل 24/7 مع تحديثات لحظية*
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء العام"""
        logger.error(f"Exception while handling an update: {context.error}")

    def process_update_sync(self, update_json):
        """معالجة التحديث بشكل متزامن"""
        try:
            update = Update.de_json(update_json, self.application.bot)
            if update:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.application.process_update(update))
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Error processing update: {e}")

    def run(self):
        """تشغيل البوت"""
        try:
            self.application = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("pro", self.pro_analysis_command))
            self.application.add_handler(CommandHandler("quick", self.quick_analysis_command))
            self.application.add_handler(CommandHandler("compare", self.compare_command))
            self.application.add_handler(CommandHandler("alerts", self.alerts_command))
            self.application.add_handler(CommandHandler("settings", self.settings_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CallbackQueryHandler(self.button_callback))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            
            # إضافة معالج الأخطاء
            self.application.add_error_handler(self.error_handler)
            
            # تحديد وضع التشغيل
            if self.webhook_url and self.webhook_url.strip():
                self.run_webhook()
            else:
                self.run_polling()
                
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    def run_webhook(self):
        """تشغيل البوت في وضع الـ webhook"""
        try:
            app = Flask(__name__)
            
            @app.route('/')
            def health_check():
                return "🤖 Professional Crypto Analysis Bot is running!", 200

            @app.route('/health')
            def health():
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

            @app.route('/webhook', methods=['POST'])
            def webhook():
                try:
                    json_data = request.get_json(force=True)
                    logger.info(f"📩 Received webhook data")
                    
                    if json_data:
                        thread = threading.Thread(
                            target=self.process_update_sync,
                            args=(json_data,)
                        )
                        thread.daemon = True
                        thread.start()
                        return "OK", 200
                    
                    return "No data", 400
                    
                except Exception as e:
                    logger.error(f"❌ Webhook error: {e}")
                    return "Error", 500

            async def setup_webhook():
                try:
                    await self.application.initialize()
                    
                    webhook_endpoint = f"{self.webhook_url}/webhook"
                    
                    # حذف webhook القديم أولاً
                    await self.application.bot.delete_webhook()
                    await asyncio.sleep(1)
                    
                    # إعداد webhook جديد
                    webhook_set = await self.application.bot.set_webhook(
                        url=webhook_endpoint,
                        allowed_updates=["message", "callback_query"],
                        drop_pending_updates=True
                    )
                    
                    if webhook_set:
                        logger.info(f"✅ Webhook set successfully to: {webhook_endpoint}")
                    else:
                        logger.error("❌ Failed to set webhook")
                    
                    # التحقق من إعدادات webhook
                    webhook_info = await self.application.bot.get_webhook_info()
                    logger.info(f"Webhook info: URL={webhook_info.url}, Pending={webhook_info.pending_update_count}")
                    
                    # إرسال تنبيه للإدمن
                    if self.admin_id:
                        try:
                            await self.application.bot.send_message(
                                chat_id=self.admin_id,
                                text=f"🚀 *البوت الاحترافي يعمل بنجاح!*\n\n"
                                     f"⏰ وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                     f"🌐 Webhook: Active\n"
                                     f"📊 نظام التحليل: مُفعّل\n"
                                     f"🔧 الإطارات الزمنية: 5\n"
                                     f"📈 المؤشرات: 15+",
                                parse_mode='Markdown'
                            )
                        except Exception as e:
                            logger.error(f"Error sending startup message: {e}")
                            
                except Exception as e:
                    logger.error(f"Error setting up webhook: {e}")
            
            # تشغيل الإعداد
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup_webhook())
            
            # تشغيل Flask
            port = int(os.environ.get('PORT', 10000))
            logger.info(f"🤖 Professional Bot running on webhook mode, port: {port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            
        except Exception as e:
            logger.error(f"Error in webhook mode: {e}")
            raise

    def run_polling(self):
        """تشغيل البوت في وضع الـ polling"""
        try:
            async def main():
                await self.application.initialize()
                
                # إرسال تنبيه للإدمن
                if self.admin_id:
                    try:
                        await self.application.bot.send_message(
                            chat_id=self.admin_id,
                            text=f"🚀 *البوت الاحترافي يعمل بنجاح!*\n\n"
                                 f"⏰ وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                 f"🔄 وضع: Polling\n"
                                 f"📊 نظام التحليل: مُفعّل\n"
                                 f"🔧 الإطارات الزمنية: 5\n"
                                 f"📈 المؤشرات: 15+",
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        logger.error(f"Error sending startup message: {e}")
                
                logger.info("🤖 Professional Bot running in polling mode...")
                await self.application.run_polling(drop_pending_updates=True)

            asyncio.run(main())
            
        except Exception as e:
            logger.error(f"Error in polling mode: {e}")
            raise

# التشغيل الرئيسي
if __name__ == "__main__":
    try:
        # التحقق من متغيرات البيئة
        if not os.getenv('BOT_TOKEN'):
            print("❌ BOT_TOKEN غير موجود في متغيرات البيئة")
            exit(1)
            
        print("🚀 بدء تشغيل البوت الاحترافي للتحليل الفني...")
        print("📊 تحميل نظام التحليل المتقدم...")
        print("🔧 إعداد المؤشرات الفنية...")
        print("⚡ جاهز للعمل!")
        
        # إنشاء وتشغيل البوت
        bot = ProfessionalCryptoBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل البوت: {e}")
        exit(1)
