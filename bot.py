import os
import asyncio
import logging
import aiohttp
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import Dict, List, Optional, Tuple
from flask import Flask
import threading

# إعداد الـ logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# إنشاء Flask app للـ health check
app = Flask(__name__)

@app.route('/')
def health_check():
    return "🤖 Multi-Timeframe Crypto Bot is running!", 200

@app.route('/health')
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

def run_flask():
    """تشغيل Flask في thread منفصل"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

class TechnicalAnalyzer:
    """محلل فني متعدد الأطر الزمنية"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        # الأطر الزمنية المدعومة
        self.timeframes = {
            '1h': {'name': 'ساعة', 'emoji': '⏰', 'interval': '1h', 'limit': 100},
            '4h': {'name': '4 ساعات', 'emoji': '🕐', 'interval': '4h', 'limit': 100},
            '1d': {'name': 'يومي', 'emoji': '📅', 'interval': '1d', 'limit': 100},
            '3d': {'name': '3 أيام', 'emoji': '📆', 'interval': '3d', 'limit': 100},
            '1w': {'name': 'أسبوعي', 'emoji': '📊', 'interval': '1w', 'limit': 100}
        }
        
    async def get_price_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السعر الحالي"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    async def get_kline_data(self, symbol: str, timeframe: str) -> Optional[List]:
        """جلب بيانات الشموع لإطار زمني محدد"""
        try:
            tf_config = self.timeframes.get(timeframe)
            if not tf_config:
                return None
                
            url = f"{self.base_url}/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': tf_config['interval'],
                'limit': tf_config['limit']
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching {timeframe} kline data: {e}")
            return None

    def calculate_indicators(self, kline_data: List, timeframe: str) -> Dict:
        """حساب المؤشرات الفنية للإطار الزمني"""
        try:
            if not kline_data or len(kline_data) < 20:
                return {}
            
            closes = [float(kline[4]) for kline in kline_data]
            highs = [float(kline[2]) for kline in kline_data]
            lows = [float(kline[3]) for kline in kline_data]
            volumes = [float(kline[5]) for kline in kline_data]
            
            current_price = closes[-1]
            
            # تحديد فترات المتوسطات حسب الإطار الزمني
            if timeframe in ['1h']:
                ma_periods = [20, 50, 100]
            elif timeframe in ['4h']:
                ma_periods = [12, 24, 50]
            elif timeframe in ['1d']:
                ma_periods = [7, 20, 50]
            elif timeframe in ['3d']:
                ma_periods = [5, 10, 20]
            else:  # 1w
                ma_periods = [4, 8, 16]
            
            # حساب المتوسطات المتحركة
            sma_short = sum(closes[-ma_periods[0]:]) / ma_periods[0] if len(closes) >= ma_periods[0] else current_price
            sma_medium = sum(closes[-ma_periods[1]:]) / ma_periods[1] if len(closes) >= ma_periods[1] else current_price
            sma_long = sum(closes[-ma_periods[2]:]) / ma_periods[2] if len(closes) >= ma_periods[2] else current_price
            
            # RSI متكيف مع الإطار الزمني
            rsi_period = 14 if timeframe in ['1h', '4h'] else 10 if timeframe == '1d' else 7
            rsi = self.calculate_rsi(closes, rsi_period)
            
            # MACD متكيف
            if timeframe in ['1h', '4h']:
                macd_fast, macd_slow, macd_signal_period = 12, 26, 9
            elif timeframe == '1d':
                macd_fast, macd_slow, macd_signal_period = 8, 16, 6
            else:
                macd_fast, macd_slow, macd_signal_period = 5, 10, 4
                
            macd_line, macd_signal = self.calculate_macd(closes, macd_fast, macd_slow, macd_signal_period)
            
            # الدعم والمقاومة
            lookback = min(len(closes), 50 if timeframe in ['1h', '4h'] else 30)
            support = min(lows[-lookback:])
            resistance = max(highs[-lookback:])
            
            # تحليل الحجم
            vol_period = min(len(volumes), 20)
            avg_volume = sum(volumes[-vol_period:]) / vol_period
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # مستويات فيبوناتشي
            price_range = resistance - support
            fib_levels = {
                '23.6': resistance - (price_range * 0.236),
                '38.2': resistance - (price_range * 0.382),
                '50.0': resistance - (price_range * 0.500),
                '61.8': resistance - (price_range * 0.618),
                '78.6': resistance - (price_range * 0.786)
            }
            
            return {
                'timeframe': timeframe,
                'current_price': current_price,
                'sma_short': sma_short,
                'sma_medium': sma_medium,
                'sma_long': sma_long,
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'support': support,
                'resistance': resistance,
                'volume_ratio': volume_ratio,
                'fib_levels': fib_levels,
                'ma_periods': ma_periods
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe}: {e}")
            return {}

    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """حساب RSI"""
        try:
            if len(closes) < period + 1:
                return 50
                
            gains = []
            losses = []
            
            for i in range(1, min(len(closes), period + 1)):
                change = closes[-i] - closes[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if not gains or not losses:
                return 50
                
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception:
            return 50

    def calculate_macd(self, closes: List[float], fast: int, slow: int, signal: int) -> Tuple[float, float]:
        """حساب MACD"""
        try:
            if len(closes) < slow:
                return 0, 0
                
            # EMA calculation
            def ema(data, period):
                multiplier = 2 / (period + 1)
                ema_values = [data[0]]
                for price in data[1:]:
                    ema_values.append((price * multiplier) + (ema_values[-1] * (1 - multiplier)))
                return ema_values[-1]
            
            ema_fast = ema(closes[-fast:], fast)
            ema_slow = ema(closes[-slow:], slow)
            macd_line = ema_fast - ema_slow
            
            # للبساطة، نستخدم SMA للإشارة
            if len(closes) >= slow + signal:
                recent_macd = []
                for i in range(signal):
                    if len(closes) >= slow + i + 1:
                        fast_val = ema(closes[-(fast+i):-i] if i > 0 else closes[-fast:], fast)
                        slow_val = ema(closes[-(slow+i):-i] if i > 0 else closes[-slow:], slow)
                        recent_macd.append(fast_val - slow_val)
                
                macd_signal = sum(recent_macd) / len(recent_macd) if recent_macd else macd_line
            else:
                macd_signal = macd_line
                
            return macd_line, macd_signal
            
        except Exception:
            return 0, 0

    def analyze_timeframe(self, indicators: Dict, timeframe: str) -> Dict:
        """تحليل إطار زمني واحد"""
        try:
            if not indicators:
                return {'trend': 'غير محدد', 'strength': 0, 'recommendation': '⚪ انتظار'}
            
            current_price = indicators.get('current_price', 0)
            sma_short = indicators.get('sma_short', 0)
            sma_medium = indicators.get('sma_medium', 0)
            sma_long = indicators.get('sma_long', 0)
            rsi = indicators.get('rsi', 50)
            macd_line = indicators.get('macd_line', 0)
            macd_signal = indicators.get('macd_signal', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            signals = []
            
            # إشارات المتوسطات المتحركة (وزن أكبر)
            if current_price > sma_short > sma_medium > sma_long:
                signals.extend([2, 2])  # إشارة قوية جداً
            elif current_price > sma_short > sma_medium:
                signals.extend([1.5, 1])
            elif current_price > sma_medium > sma_long:
                signals.append(1)
            elif current_price < sma_short < sma_medium < sma_long:
                signals.extend([-2, -2])
            elif current_price < sma_short < sma_medium:
                signals.extend([-1.5, -1])
            elif current_price < sma_medium < sma_long:
                signals.append(-1)
            else:
                signals.append(0)
            
            # إشارات RSI (متكيفة مع الإطار الزمني)
            if timeframe in ['1h', '4h']:
                # أطر قصيرة - حساسية أعلى
                if rsi < 25:
                    signals.append(1.5)
                elif rsi < 35:
                    signals.append(1)
                elif rsi > 75:
                    signals.append(-1.5)
                elif rsi > 65:
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                # أطر طويلة - حساسية أقل
                if rsi < 30:
                    signals.append(1)
                elif rsi > 70:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # إشارات MACD
            if macd_line > macd_signal and macd_line > 0:
                signals.append(1.5)
            elif macd_line > macd_signal:
                signals.append(1)
            elif macd_line < macd_signal and macd_line < 0:
                signals.append(-1.5)
            elif macd_line < macd_signal:
                signals.append(-1)
            else:
                signals.append(0)
            
            # إشارات الحجم
            if volume_ratio > 2:
                signals.append(1)
            elif volume_ratio > 1.5:
                signals.append(0.5)
            elif volume_ratio < 0.5:
                signals.append(-0.5)
            else:
                signals.append(0)
            
            # حساب متوسط الإشارات
            avg_signal = sum(signals) / len(signals) if signals else 0
            strength = abs(avg_signal)
            
            # تحديد التوصية
            if avg_signal > 1.5:
                trend = "صاعد قوي جداً"
                recommendation = "🟢 شراء قوي"
            elif avg_signal > 1:
                trend = "صاعد قوي"
                recommendation = "🟢 شراء قوي"
            elif avg_signal > 0.5:
                trend = "صاعد"
                recommendation = "🔵 شراء"
            elif avg_signal > 0.2:
                trend = "صاعد ضعيف"
                recommendation = "🔵 شراء ضعيف"
            elif avg_signal < -1.5:
                trend = "هابط قوي جداً"
                recommendation = "🔴 بيع قوي"
            elif avg_signal < -1:
                trend = "هابط قوي"
                recommendation = "🔴 بيع قوي"
            elif avg_signal < -0.5:
                trend = "هابط"
                recommendation = "🟠 بيع"
            elif avg_signal < -0.2:
                trend = "هابط ضعيف"
                recommendation = "🟠 بيع ضعيف"
            else:
                trend = "محايد"
                recommendation = "⚪ انتظار"
            
            return {
                'trend': trend,
                'strength': strength,
                'recommendation': recommendation,
                'signal_score': avg_signal,
                'signals_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            return {'trend': 'غير محدد', 'strength': 0, 'recommendation': '⚪ انتظار'}

    async def multi_timeframe_analysis(self, symbol: str, selected_timeframes: List[str] = None) -> Dict:
        """تحليل متعدد الأطر الزمنية"""
        try:
            if selected_timeframes is None:
                selected_timeframes = list(self.timeframes.keys())
            
            # جلب بيانات السعر العامة
            price_data = await self.get_price_data(symbol)
            if not price_data:
                return {}
            
            results = {
                'symbol': symbol,
                'price_data': price_data,
                'timeframe_analysis': {},
                'overall_consensus': {},
                'timestamp': datetime.now()
            }
            
            # تحليل كل إطار زمني
            for tf in selected_timeframes:
                if tf not in self.timeframes:
                    continue
                    
                # جلب البيانات
                kline_data = await self.get_kline_data(symbol, tf)
                if not kline_data:
                    continue
                
                # حساب المؤشرات
                indicators = self.calculate_indicators(kline_data, tf)
                if not indicators:
                    continue
                
                # تحليل الإطار الزمني
                analysis = self.analyze_timeframe(indicators, tf)
                
                # حساب الأهداف ووقف الخسارة
                current_price = indicators['current_price']
                support = indicators['support']
                resistance = indicators['resistance']
                
                targets = self.calculate_targets(current_price, support, resistance, tf)
                
                results['timeframe_analysis'][tf] = {
                    'config': self.timeframes[tf],
                    'indicators': indicators,
                    'analysis': analysis,
                    'targets': targets
                }
                
                # إضافة تأخير صغير بين الطلبات
                await asyncio.sleep(0.1)
            
            # حساب الإجماع العام
            results['overall_consensus'] = self.calculate_consensus(results['timeframe_analysis'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}

    def calculate_targets(self, current_price: float, support: float, resistance: float, timeframe: str) -> Dict:
        """حساب الأهداف ووقف الخسارة حسب الإطار الزمني"""
        try:
            # تحديد النسب حسب الإطار الزمني
            if timeframe == '1h':
                # أهداف قصيرة للإطار الساعي
                targets_up = [1.005, 1.015, 1.03]  # 0.5%, 1.5%, 3%
                targets_down = [0.995, 0.985, 0.97]
                sl_margin = 0.02  # 2%
            elif timeframe == '4h':
                targets_up = [1.01, 1.025, 1.05]  # 1%, 2.5%, 5%
                targets_down = [0.99, 0.975, 0.95]
                sl_margin = 0.03  # 3%
            elif timeframe == '1d':
                targets_up = [1.03, 1.07, 1.15]  # 3%, 7%, 15%
                targets_down = [0.97, 0.93, 0.85]
                sl_margin = 0.05  # 5%
            elif timeframe == '3d':
                targets_up = [1.05, 1.12, 1.25]  # 5%, 12%, 25%
                targets_down = [0.95, 0.88, 0.75]
                sl_margin = 0.08  # 8%
            else:  # 1w
                targets_up = [1.08, 1.18, 1.35]  # 8%, 18%, 35%
                targets_down = [0.92, 0.82, 0.65]
                sl_margin = 0.12  # 12%
            
            return {
                'buy_targets': [current_price * t for t in targets_up],
                'sell_targets': [current_price * t for t in targets_down],
                'stop_loss': {
                    'buy': max(support * 0.98, current_price * (1 - sl_margin)),
                    'sell': min(resistance * 1.02, current_price * (1 + sl_margin))
                },
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            logger.error(f"Error calculating targets: {e}")
            return {}

    def calculate_consensus(self, timeframe_analysis: Dict) -> Dict:
        """حساب الإجماع العام من جميع الأطر الزمنية"""
        try:
            if not timeframe_analysis:
                return {'trend': 'غير محدد', 'confidence': 0, 'recommendation': '⚪ انتظار'}
            
            # جمع الإشارات من جميع الأطر مع أوزان مختلفة
            weighted_signals = []
            recommendations = []
            
            # أوزان الأطر الزمنية
            weights = {
                '1h': 1,      # وزن أقل للأطر القصيرة
                '4h': 1.5,    # وزن متوسط
                '1d': 2,      # وزن أعلى للإطار اليومي
                '3d': 1.8,    # وزن عالي
                '1w': 2.2     # وزن أعلى للإطار الأسبوعي
            }
            
            for tf, data in timeframe_analysis.items():
                analysis = data.get('analysis', {})
                signal_score = analysis.get('signal_score', 0)
                recommendation = analysis.get('recommendation', '⚪ انتظار')
                
                weight = weights.get(tf, 1)
                weighted_signals.append(signal_score * weight)
                recommendations.append(recommendation)
            
            if not weighted_signals:
                return {'trend': 'غير محدد', 'confidence': 0, 'recommendation': '⚪ انتظار'}
            
            # حساب المتوسط المرجح
            avg_signal = sum(weighted_signals) / sum(weights[tf] for tf in timeframe_analysis.keys())
            confidence = min(abs(avg_signal) * 50, 100)  # تحويل لنسبة مئوية
            
            # تحديد الإجماع
            bullish_count = sum(1 for r in recommendations if '🟢' in r or '🔵' in r)
            bearish_count = sum(1 for r in recommendations if '🔴' in r or '🟠' in r)
            neutral_count = len(recommendations) - bullish_count - bearish_count
            
            # تحديد التوصية الإجمالية
            if avg_signal > 1:
                trend = "صاعد قوي"
                recommendation = "🟢 شراء قوي"
            elif avg_signal > 0.5:
                trend = "صاعد"
                recommendation = "🔵 شراء"
            elif avg_signal < -1:
                trend = "هابط قوي"
                recommendation = "🔴 بيع قوي"
            elif avg_signal < -0.5:
                trend = "هابط"
                recommendation = "🟠 بيع"
            else:
                trend = "محايد"
                recommendation = "⚪ انتظار"
            
            return {
                'trend': trend,
                'confidence': confidence,
                'recommendation': recommendation,
                'signal_score': avg_signal,
                'agreement': {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count,
                    'total': len(recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return {'trend': 'غير محدد', 'confidence': 0, 'recommendation': '⚪ انتظار'}

class CryptoBot:
    """بوت التوصيات متعدد الأطر الزمنية"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.analyzer = TechnicalAnalyzer()
        self.user_preferences = {}  # تفضيلات المستخدمين للأطر الزمنية
        
        if not self.token:
            raise ValueError("BOT_TOKEN is required")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        welcome = """
🤖 *بوت التحليل متعدد الأطر الزمنية!*

📊 *الأطر الزمنية المدعومة:*
⏰ ساعة | 🕐 4 ساعات | 📅 يومي
📆 3 أيام | 📊 أسبوعي

🎯 *المميزات:*
• تحليل شامل لجميع الأطر
• إجماع ذكي من الأطر المختلفة
• أهداف مخصصة لكل إطار زمني
• مؤشرات متكيفة حسب الإطار

📝 *الاستخدام:*
• تحليل سريع: `BTC`
• تحليل مخصص: `/analyze BTC`
• اختيار الأطر: `/timeframes`

🚀 *جرب الآن!*
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل سريع", callback_data="quick_BTC")],
            [InlineKeyboardButton("🕐 اختيار الأطر الزمنية", callback_data="timeframes")],
            [InlineKeyboardButton("❓ مساعدة", callback_data="help")]
        ]
        
        await update.message.reply_text(
            welcome, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def timeframes_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """قائمة اختيار الأطر الزمنية"""
        keyboard = [
            [InlineKeyboardButton("⏰ ساعة", callback_data="tf_1h"),
             InlineKeyboardButton("🕐 4 ساعات", callback_data="tf_4h")],
            [InlineKeyboardButton("📅 يومي", callback_data="tf_1d"),
             InlineKeyboardButton("📆 3 أيام", callback_data="tf_3d")],
            [InlineKeyboardButton("📊 أسبوعي", callback_data="tf_1w")],
            [InlineKeyboardButton("✅ جميع الأطر", callback_data="tf_all")],
            [InlineKeyboardButton("🔙 رجوع", callback_data="back_main")]
        ]
        
        text = """
🕐 *اختر الأطر الزمنية للتحليل:*

⏰ **ساعة** - للتداول السريع
🕐 **4 ساعات** - للتداول اليومي  
📅 **يومي** - للاستثمار قصير المدى
📆 **3 أيام** - للاستثمار متوسط المدى
📊 **أسبوعي** - للاستثمار طويل المدى

💡 *أو اختر جميع الأطر للتحليل الشامل*
        """
        
        try:
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard)
                )
            else:
                await update.message.reply_text(
                    text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard)
                )
        except Exception as e:
            logger.error(f"Error in timeframes menu: {e}")

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                     symbol: str = None, timeframes: List[str] = None):
        """تحليل العملة على الأطر المحددة"""
        try:
            # تحديد الرمز
            if not symbol:
                if context.args:
                    symbol = context.args[0].upper()
                else:
                    await update.message.reply_text("❌ أرسل رمز العملة\nمثال: /analyze BTC")
                    return
            
            # تحديد الأطر الزمنية
            if not timeframes:
                user_id = update.effective_user.id
                timeframes = self.user_preferences.get(user_id, list(self.analyzer.timeframes.keys()))
            
            # رسالة انتظار
            msg = await update.message.reply_text(
                f"🔍 جاري التحليل الشامل لـ {symbol}...\n"
                f"📊 الأطر: {', '.join([self.analyzer.timeframes[tf]['emoji'] + tf for tf in timeframes])}\n"
                f"⏳ يرجى الانتظار..."
            )
            
            # إجراء التحليل
            analysis = await self.analyzer.multi_timeframe_analysis(symbol, timeframes)
            
            if not analysis or not analysis.get('timeframe_analysis'):
                await msg.edit_text(f"❌ لم أجد بيانات لـ {symbol}")
                return
            
            # عرض النتائج
            await self.show_analysis_results(msg, symbol, analysis)
            
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            await update.message.reply_text(f"❌ خطأ في تحليل {symbol}")

    async def show_analysis_results(self, message, symbol: str, analysis: Dict):
        """عرض نتائج التحليل"""
        try:
            # الإجماع العام
            consensus = analysis.get('overall_consensus', {})
            price_data = analysis.get('price_data', {})
            timeframe_analysis = analysis.get('timeframe_analysis', {})
            
            current_price = float(price_data.get('lastPrice', 0))
            change_24h = float(price_data.get('priceChangePercent', 0))
            volume = float(price_data.get('volume', 0))
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.6f}"
            elif current_price < 1:
                price_str = f"${current_price:.4f}"
            else:
                price_str = f"${current_price:.2f}"
            
            change_emoji = "🟢" if change_24h >= 0 else "🔴"
            
            # التقرير الرئيسي
            main_report = f"""
🎯 *تحليل {symbol}/USDT شامل*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر:* {price_str}
{change_emoji} *24س:* {change_24h:+.2f}%
📊 *الحجم:* {volume:,.0f}

🏆 *الإجماع العام:*
📈 *الاتجاه:* {consensus.get('trend', 'غير محدد')}
🎲 *التوصية:* {consensus.get('recommendation', '⚪ انتظار')}
💪 *الثقة:* {consensus.get('confidence', 0):.1f}%

━━━━━━━━━━━━━━━━━━━━
🕐 *تحليل الأطر الزمنية:*
            """
            
            # تحليل كل إطار زمني
            for tf, data in timeframe_analysis.items():
                tf_config = data['config']
                tf_analysis = data['analysis']
                
                main_report += f"""
{tf_config['emoji']} *{tf_config['name']}:* {tf_analysis['recommendation']}
            """
            
            # الأزرار للتفاصيل
            keyboard = []
            
            # أزرار الأطر الزمنية
            tf_buttons = []
            for tf, data in timeframe_analysis.items():
                tf_config = data['config']
                tf_buttons.append(
                    InlineKeyboardButton(
                        f"{tf_config['emoji']} {tf}", 
                        callback_data=f"detail_{symbol}_{tf}"
                    )
                )
                if len(tf_buttons) == 2:
                    keyboard.append(tf_buttons)
                    tf_buttons = []
            
            if tf_buttons:
                keyboard.append(tf_buttons)
            
            # أزرار إضافية
            keyboard.extend([
                [InlineKeyboardButton("📊 الإجماع التفصيلي", callback_data=f"consensus_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_{symbol}"),
                 InlineKeyboardButton("⚙️ تغيير الأطر", callback_data="timeframes")]
            ])
            
            await message.edit_text(
                main_report, 
                parse_mode='Markdown', 
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error showing results: {e}")
            await message.edit_text(f"❌ خطأ في عرض نتائج {symbol}")

    async def show_timeframe_detail(self, query, symbol: str, timeframe: str, analysis: Dict):
        """عرض تفاصيل إطار زمني محدد"""
        try:
            tf_data = analysis['timeframe_analysis'].get(timeframe)
            if not tf_data:
                await query.answer("❌ لا توجد بيانات لهذا الإطار")
                return
            
            tf_config = tf_data['config']
            indicators = tf_data['indicators']
            tf_analysis = tf_data['analysis']
            targets = tf_data['targets']
            
            current_price = indicators['current_price']
            
            detail_report = f"""
{tf_config['emoji']} *تحليل {symbol} - {tf_config['name']}*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر:* ${current_price:.6f}
📈 *الاتجاه:* {tf_analysis['trend']}
🎲 *التوصية:* {tf_analysis['recommendation']}
💪 *قوة الإشارة:* {tf_analysis['strength']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━
📊 *المؤشرات الفنية:*

🔸 *RSI:* {indicators['rsi']:.1f}
🔸 *MACD:* {'إيجابي' if indicators['macd_line'] > indicators['macd_signal'] else 'سلبي'}
🔸 *MA القصير:* ${indicators['sma_short']:.6f}
🔸 *MA المتوسط:* ${indicators['sma_medium']:.6f}
🔸 *MA الطويل:* ${indicators['sma_long']:.6f}

━━━━━━━━━━━━━━━━━━━━
🎯 *مستويات التداول:*

🟢 *الدعم:* ${targets['support']:.6f}
🔴 *المقاومة:* ${targets['resistance']:.6f}
            """
            
            # إضافة الأهداف حسب التوصية
            if "شراء" in tf_analysis['recommendation']:
                buy_targets = targets['buy_targets']
                detail_report += f"""
🏆 *أهداف الشراء:*
🥇 ${buy_targets[0]:.6f} ({((buy_targets[0]/current_price-1)*100):+.1f}%)
🥈 ${buy_targets[1]:.6f} ({((buy_targets[1]/current_price-1)*100):+.1f}%)
🥉 ${buy_targets[2]:.6f} ({((buy_targets[2]/current_price-1)*100):+.1f}%)

🛑 *وقف الخسارة:* ${targets['stop_loss']['buy']:.6f}
                """
            elif "بيع" in tf_analysis['recommendation']:
                sell_targets = targets['sell_targets']
                detail_report += f"""
🎯 *أهداف البيع:*
🥇 ${sell_targets[0]:.6f} ({((sell_targets[0]/current_price-1)*100):+.1f}%)
🥈 ${sell_targets[1]:.6f} ({((sell_targets[1]/current_price-1)*100):+.1f}%)
🥉 ${sell_targets[2]:.6f} ({((sell_targets[2]/current_price-1)*100):+.1f}%)

🛑 *وقف الخسارة:* ${targets['stop_loss']['sell']:.6f}
                """
            else:
                detail_report += """
⚪ *وضع انتظار:*
راقب كسر الدعم أو المقاومة
                """
            
            detail_report += f"""
━━━━━━━━━━━━━━━━━━━━
📏 *مستويات فيبوناتشي:*
🔸 23.6%: ${indicators['fib_levels']['23.6']:.6f}
🔸 38.2%: ${indicators['fib_levels']['38.2']:.6f}
🔸 50.0%: ${indicators['fib_levels']['50.0']:.6f}
🔸 61.8%: ${indicators['fib_levels']['61.8']:.6f}

⏰ *الإطار الزمني:* {tf_config['name']}
📊 *فترات المتوسطات:* {indicators['ma_periods']}
            """
            
            keyboard = [
                [InlineKeyboardButton("🔙 رجوع للتحليل الشامل", callback_data=f"back_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث هذا الإطار", callback_data=f"refresh_{symbol}_{timeframe}")]
            ]
            
            await query.edit_message_text(
                detail_report,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error showing timeframe detail: {e}")
            await query.answer("❌ خطأ في عرض التفاصيل")

    async def show_consensus_detail(self, query, symbol: str, analysis: Dict):
        """عرض تفاصيل الإجماع"""
        try:
            consensus = analysis.get('overall_consensus', {})
            agreement = consensus.get('agreement', {})
            timeframe_analysis = analysis.get('timeframe_analysis', {})
            
            consensus_report = f"""
🏆 *الإجماع التفصيلي لـ {symbol}*
━━━━━━━━━━━━━━━━━━━━

📊 *النتيجة الإجمالية:*
📈 *الاتجاه:* {consensus.get('trend', 'غير محدد')}
🎲 *التوصية:* {consensus.get('recommendation', '⚪ انتظار')}
💪 *مستوى الثقة:* {consensus.get('confidence', 0):.1f}%
🎯 *نقاط الإشارة:* {consensus.get('signal_score', 0):.2f}

━━━━━━━━━━━━━━━━━━━━
📋 *توزيع الآراء:*

🟢 *صاعد:* {agreement.get('bullish', 0)} أطر
🔴 *هابط:* {agreement.get('bearish', 0)} أطر
⚪ *محايد:* {agreement.get('neutral', 0)} أطر
📊 *المجموع:* {agreement.get('total', 0)} أطر

━━━━━━━━━━━━━━━━━━━━
🕐 *تفصيل كل إطار:*
            """
            
            # تفاصيل كل إطار زمني
            for tf, data in timeframe_analysis.items():
                tf_config = data['config']
                tf_analysis = data['analysis']
                
                consensus_report += f"""
{tf_config['emoji']} *{tf_config['name']}:*
   📈 {tf_analysis['trend']}
   🎲 {tf_analysis['recommendation']}
   💪 {tf_analysis['strength']*100:.1f}%

            """
            
            # توصية نهائية
            if consensus.get('confidence', 0) > 70:
                confidence_level = "عالية جداً 🔥"
            elif consensus.get('confidence', 0) > 50:
                confidence_level = "عالية ✅"
            elif consensus.get('confidence', 0) > 30:
                confidence_level = "متوسطة ⚡"
            else:
                confidence_level = "منخفضة ⚠️"
            
            consensus_report += f"""
━━━━━━━━━━━━━━━━━━━━
💡 *التوصية النهائية:*

📊 *مستوى الثقة:* {confidence_level}
🎯 *قرار الاستثمار:* {consensus.get('recommendation', '⚪ انتظار')}

⚠️ *نصائح:*
• ثقة عالية = فرصة قوية
• ثقة متوسطة = حذر مطلوب  
• ثقة منخفضة = انتظار أفضل
            """
            
            keyboard = [
                [InlineKeyboardButton("🔙 رجوع للتحليل الشامل", callback_data=f"back_{symbol}")],
                [InlineKeyboardButton("📊 إعادة التحليل", callback_data=f"refresh_{symbol}")]
            ]
            
            await query.edit_message_text(
                consensus_report,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error showing consensus detail: {e}")
            await query.answer("❌ خطأ في عرض الإجماع")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص - تحليل سريع"""
        text = update.message.text.upper().strip()
        
        if len(text) <= 10 and text.isalpha():
            # تحليل سريع على جميع الأطر
            await self.analyze(update, context, symbol=text)
        else:
            await update.message.reply_text(
                "💡 أرسل رمز العملة للتحليل السريع\n"
                "أو استخدم /analyze للتحليل المخصص"
            )

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأزرار"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        try:
            if data == "timeframes":
                await self.timeframes_menu(update, context)
                
            elif data.startswith("tf_"):
                # تحديد الأطر الزمنية
                tf = data.split("_")[1]
                user_id = update.effective_user.id
                
                if tf == "all":
                    self.user_preferences[user_id] = list(self.analyzer.timeframes.keys())
                    await query.edit_message_text("✅ تم اختيار جميع الأطر الزمنية")
                else:
                    self.user_preferences[user_id] = [tf]
                    tf_name = self.analyzer.timeframes[tf]['name']
                    await query.edit_message_text(f"✅ تم اختيار إطار {tf_name}")
                    
            elif data.startswith("detail_"):
                # عرض تفاصيل إطار زمني
                parts = data.split("_")
                if len(parts) >= 3:
                    symbol = parts[1]
                    timeframe = parts[2]
                    
                    # نحتاج لإعادة التحليل للحصول على البيانات
                    analysis = await self.analyzer.multi_timeframe_analysis(symbol, [timeframe])
                    if analysis:
                        await self.show_timeframe_detail(query, symbol, timeframe, analysis)
                    
            elif data.startswith("consensus_"):
                # عرض تفاصيل الإجماع
                symbol = data.split("_")[1]
                analysis = await self.analyzer.multi_timeframe_analysis(symbol)
                if analysis:
                    await self.show_consensus_detail(query, symbol, analysis)
                    
            elif data.startswith("refresh_"):
                # تحديث التحليل
                parts = data.split("_")
                symbol = parts[1]
                
                msg = await query.edit_message_text(f"🔄 جاري تحديث تحليل {symbol}...")
                
                if len(parts) > 2:
                    # تحديث إطار واحد
                    timeframe = parts[2]
                    analysis = await self.analyzer.multi_timeframe_analysis(symbol, [timeframe])
                    if analysis:
                        await self.show_timeframe_detail(query, symbol, timeframe, analysis)
                else:
                    # تحديث شامل
                    analysis = await self.analyzer.multi_timeframe_analysis(symbol)
                    if analysis:
                        await self.show_analysis_results(msg, symbol, analysis)
                        
            elif data.startswith("back_"):
                # العودة للتحليل الشامل
                symbol = data.split("_")[1]
                analysis = await self.analyzer.multi_timeframe_analysis(symbol)
                if analysis:
                    await self.show_analysis_results(query.message, symbol, analysis)
                    
            elif data == "help":
                help_text = """
📚 *دليل البوت متعدد الأطر:*

🔍 *التحليل:*
• `BTC` - تحليل شامل سريع
• `/analyze BTC` - تحليل مخصص
• `/timeframes` - اختيار الأطر

🕐 *الأطر الزمنية:*
⏰ ساعة - للسكالبنغ
🕐 4 ساعات - للتداول اليومي
📅 يومي - للاستثمار قصير
📆 3 أيام - للاستثمار متوسط
📊 أسبوعي - للاستثمار طويل

🎯 *المميزات:*
• إجماع ذكي من جميع الأطر
• أهداف مخصصة لكل إطار
• مؤشرات متكيفة
• مستويات فيبوناتشي

⚠️ للأغراض التعليمية فقط
                """
                await query.edit_message_text(help_text, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.answer("❌ حدث خطأ")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء"""
        logger.error(f"Update {update} caused error {context.error}")

    def run(self):
        """تشغيل البوت"""
        try:
            # بدء Flask
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            logger.info("Flask server started")
            
            # إنشاء التطبيق
            app = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            app.add_handler(CommandHandler("start", self.start))
            app.add_handler(CommandHandler("analyze", self.analyze))
            app.add_handler(CommandHandler("timeframes", self.timeframes_menu))
            app.add_handler(CallbackQueryHandler(self.button_handler))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            app.add_error_handler(self.error_handler)
            
            logger.info("🤖 Starting Multi-Timeframe Crypto Bot...")
            
            # تشغيل البوت
            app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False
            )
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

# التشغيل الرئيسي
if __name__ == "__main__":
    try:
        if not os.getenv('BOT_TOKEN'):
            print("❌ BOT_TOKEN required")
            exit(1)
            
        bot = CryptoBot()
        bot.run()
        
    except Exception as e:
        print(f"❌ Bot error: {e}")
        exit(1)
