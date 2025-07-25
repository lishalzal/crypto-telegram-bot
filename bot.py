import os
import asyncio
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import Dict, List, Optional, Tuple
from flask import Flask, request
import threading
import json
import ta  # المكتبة التقنية للمؤشرات الفنية

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
        """جلب بيانات الأسعار من Binance API مع تعامل أفضل مع الأخطاء"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code == 429:  # Rate limit
                logger.warning("Rate limit reached, waiting...")
                return None
            elif response.status_code != 200:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
            data = response.json()
            if not data:
                return None
                
            # تحويل البيانات مع validation
            processed_data = []
            for row in data:
                try:
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
                    if candle['high'] >= candle['low'] and candle['high'] >= candle['close'] and candle['high'] >= candle['open']:
                        processed_data.append(candle)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid candle data: {e}")
                    continue
            
            return processed_data if len(processed_data) > 50 else None
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, data: List[Dict]) -> Dict:
        """حساب المؤشرات الفنية المتقدمة"""
        try:
            if len(data) < 50:
                return {}
            
            # تحويل البيانات إلى pandas DataFrame للتحليل
            import pandas as pd
            
            df = pd.DataFrame(data)
            
            # أسعار الإغلاق والأحجام
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            indicators = {}
            
            # المتوسطات المتحركة
            indicators['ema_9'] = close.ewm(span=9).mean().iloc[-1]
            indicators['ema_21'] = close.ewm(span=21).mean().iloc[-1]
            indicators['ema_50'] = close.ewm(span=50).mean().iloc[-1]
            indicators['ema_200'] = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else indicators['ema_50']
            
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
            indicators['sma_200'] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else indicators['sma_50']
            
            # RSI مع فترات متعددة
            indicators['rsi_14'] = self.calculate_rsi(close, 14)
            indicators['rsi_21'] = self.calculate_rsi(close, 21)
            
            # MACD
            macd_data = self.calculate_macd(close)
            indicators.update(macd_data)
            
            # نطاقات بولينجر
            bb_data = self.calculate_bollinger_bands(close, 20, 2)
            indicators.update(bb_data)
            
            # Stochastic
            stoch_data = self.calculate_stochastic(high, low, close)
            indicators.update(stoch_data)
            
            # ADX (Average Directional Index)
            adx_data = self.calculate_adx(high, low, close)
            indicators.update(adx_data)
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
            
            # Support and Resistance
            support_resistance = self.calculate_support_resistance(data[-50:])
            indicators.update(support_resistance)
            
            # Fibonacci levels
            fib_levels = self.calculate_fibonacci_levels(data[-100:])
            indicators.update(fib_levels)
            
            # Current price data
            current = data[-1]
            indicators['current_price'] = current['close']
            indicators['current_volume'] = current['volume']
            indicators['price_change_24h'] = ((current['close'] - data[-24]['close']) / data[-24]['close'] * 100) if len(data) >= 24 else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def calculate_rsi(self, prices, period=14):
        """حساب RSI محسن"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """حساب MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1],
                'macd_signal': signal_line.iloc[-1],
                'macd_histogram': histogram.iloc[-1]
            }
        except:
            return {'macd': 0, 'macd_signal': 0, 'macd_histogram': 0}

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """حساب نطاقات بولينجر"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            return {
                'bb_upper': (sma + (std * std_dev)).iloc[-1],
                'bb_middle': sma.iloc[-1],
                'bb_lower': (sma - (std * std_dev)).iloc[-1],
                'bb_width': ((sma + (std * std_dev)) - (sma - (std * std_dev))).iloc[-1]
            }
        except:
            return {'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'bb_width': 0}

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """حساب Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(d_period).mean()
            
            return {
                'stoch_k': k_percent.iloc[-1],
                'stoch_d': d_percent.iloc[-1]
            }
        except:
            return {'stoch_k': 50, 'stoch_d': 50}

    def calculate_adx(self, high, low, close, period=14):
        """حساب ADX"""
        try:
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / true_range.ewm(alpha=1/period).mean())
            minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / true_range.ewm(alpha=1/period).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period).mean()
            
            return {
                'adx': adx.iloc[-1],
                'plus_di': plus_di.iloc[-1],
                'minus_di': minus_di.iloc[-1]
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
        """تحليل شامل متقدم"""
        analysis_results = {}
        
        for timeframe, config in self.timeframes.items():
            try:
                logger.info(f"Analyzing {symbol} on {timeframe}")
                
                data = self.get_price_data(symbol, timeframe, config['limit'])
                if not data or len(data) < 50:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue
                
                # حساب المؤشرات المتقدمة
                indicators = self.calculate_advanced_indicators(data)
                if not indicators:
                    continue
                
                # تحليل هيكل السوق
                market_analysis = self.analyze_market_structure(indicators)
                
                # توليد إشارات التداول
                trading_signals = self.generate_trading_signals(indicators, market_analysis)
                
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
            def run_pro_analysis():
                return self.analyzer.comprehensive_analysis(symbol)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_pro_analysis)
                analysis = future.result(timeout=45)
            
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
            
            # تحليل الإطار الساعي فقط
            data = self.analyzer.get_price_data(symbol, '1h', 100)
            if not data:
                await waiting_msg.edit_text(f"❌ لا توجد بيانات لـ {symbol}")
                return
            
            indicators = self.analyzer.calculate_advanced_indicators(data)
            market_analysis = self.analyzer.analyze_market_structure(indicators)
            trading_signals = self.analyzer.generate_trading_signals(indicators, market_analysis)
            
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
        """معالج الأزرار المحسن"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith("pro_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                await self.pro_analysis_command(update, context)
                
            elif data.startswith("refresh_pro_"):
                symbol = data.split("_")[2]
                context.args = [symbol]
                await self.pro_analysis_command(update, context)
                
            elif data.startswith("refresh_quick_"):
                symbol = data.split("_")[2]
                context.args = [symbol]
                await self.quick_analysis_command(update, context)
                
            elif data == "pro_analysis":
                await query.edit_message_text(
                    "📊 *التحليل الاحترافي*\n\nأرسل رمز العملة للتحليل الشامل:\n\nمثال: BTC أو ETH أو ADA",
                    parse_mode='Markdown'
                )
                
            elif data == "quick_analysis":
                await query.edit_message_text(
                    "⚡ *التحليل السريع*\n\nأرسل رمز العملة للتحليل السريع:\n\nمثال: BTC أو ETH أو ADA",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in button callback: {e}")

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

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
