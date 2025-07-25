import os
import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from flask import Flask, request
import threading

# إعداد الـ logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """محلل فني متقدم للعملات المشفرة"""
    
    def __init__(self):
        self.timeframes = ['1h', '4h', '1d']
        
    async def get_price_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """جلب بيانات الأسعار من Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': timeframe,
                'limit': limit
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            return None
                            
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                        ])
                        
                        # تحويل البيانات للنوع المناسب
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                    else:
                        logger.error(f"API Error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """حساب المؤشرات الفنية"""
        try:
            if len(df) < 50:
                return {}
                
            # المتوسطات المتحركة
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            
            # Support and Resistance levels
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            return df.iloc[-1].to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def analyze_trend(self, indicators: Dict) -> Tuple[str, float]:
        """تحليل الاتجاه العام"""
        if not indicators:
            return "غير محدد", 0
            
        signals = []
        
        try:
            # تحليل المتوسطات المتحركة
            close = indicators.get('close', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            if close and sma_20:
                if close > sma_20:
                    signals.append(1)
                else:
                    signals.append(-1)
                    
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    signals.append(1)
                else:
                    signals.append(-1)
                    
            # تحليل MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if macd and macd_signal:
                if macd > macd_signal:
                    signals.append(1)
                else:
                    signals.append(-1)
                    
            # تحليل RSI
            rsi = indicators.get('rsi', 50)
            if rsi:
                if 30 < rsi < 70:
                    signals.append(0)  # محايد
                elif rsi <= 30:
                    signals.append(1)  # oversold - فرصة شراء
                else:
                    signals.append(-1)  # overbought - فرصة بيع
                    
            if not signals:
                return "غير محدد", 0
                
            # حساب متوسط الإشارات
            avg_signal = sum(signals) / len(signals)
            
            if avg_signal > 0.3:
                return "صاعد", avg_signal
            elif avg_signal < -0.3:
                return "هابط", avg_signal
            else:
                return "محايد", avg_signal
                
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return "غير محدد", 0

    def get_entry_exit_points(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """تحديد نقاط الدخول والخروج"""
        try:
            current_price = indicators.get('close', 0)
            if not current_price:
                return {}
            
            # حساب نقاط الدعم والمقاومة
            support_levels = []
            resistance_levels = []
            
            if len(df) >= 20:
                # آخر 20 قيعان وقمم
                lows = df['low'].tail(20).tolist()
                highs = df['high'].tail(20).tolist()
                
                support_levels = sorted(set([x for x in lows if x > 0]), reverse=True)[:3]
                resistance_levels = sorted(set([x for x in highs if x > 0]))[-3:]
            
            # نقاط الدخول
            bb_lower = indicators.get('bb_lower', current_price * 0.95)
            bb_upper = indicators.get('bb_upper', current_price * 1.05)
            support = indicators.get('support', current_price * 0.95)
            resistance = indicators.get('resistance', current_price * 1.05)
            
            entry_points = {
                'buy_zones': [
                    bb_lower,
                    support,
                    current_price * 0.95
                ],
                'sell_zones': [
                    bb_upper,
                    resistance,
                    current_price * 1.05
                ]
            }
            
            # نقاط وقف الخسارة والأهداف
            stop_loss_buy = min(support_levels) if support_levels else current_price * 0.92
            stop_loss_sell = max(resistance_levels) if resistance_levels else current_price * 1.08
            
            targets = {
                'buy_targets': [
                    current_price * 1.03,
                    current_price * 1.07,
                    current_price * 1.15
                ],
                'sell_targets': [
                    current_price * 0.97,
                    current_price * 0.93,
                    current_price * 0.85
                ]
            }
            
            return {
                'entry_points': entry_points,
                'stop_loss': {'buy': stop_loss_buy, 'sell': stop_loss_sell},
                'targets': targets,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in entry/exit points: {e}")
            return {}

    async def comprehensive_analysis(self, symbol: str) -> Dict:
        """تحليل شامل للعملة"""
        analysis_results = {}
        
        for timeframe in self.timeframes:
            try:
                df = await self.get_price_data(symbol, timeframe)
                if df is not None and len(df) > 50:
                    indicators = self.calculate_technical_indicators(df)
                    if indicators:
                        trend, strength = self.analyze_trend(indicators)
                        entry_exit = self.get_entry_exit_points(df, indicators)
                        
                        analysis_results[timeframe] = {
                            'trend': trend,
                            'strength': strength,
                            'indicators': indicators,
                            'entry_exit': entry_exit,
                            'price': indicators.get('close', 0),
                            'volume': indicators.get('volume', 0)
                        }
                        
                await asyncio.sleep(0.1)  # تجنب الضغط على API
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                continue
        
        return analysis_results

class CryptoTelegramBot:
    """بوت تليغرام للتوصيات المشفرة"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.webhook_url = os.getenv('WEBHOOK_URL')  # رابط الـ webhook
        self.analyzer = TechnicalAnalyzer()
        self.user_watchlists = {}
        self.application = None
        
        if not self.token:
            raise ValueError("BOT_TOKEN environment variable is required")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        welcome_message = """
🤖 *أهلاً بك في بوت التوصيات المشفرة المتقدم!*

📊 *الخدمات المتاحة:*
• تحليل فني شامل للعملات
• نقاط دخول وخروج دقيقة
• توصيات مبنية على مؤشرات متعددة
• مراقبة العملات المفضلة

📝 *الأوامر الأساسية:*
/analyze BTC - تحليل عملة
/watch BTC - إضافة للمراقبة
/watchlist - عرض قائمة المراقبة
/remove BTC - إزالة من المراقبة
/help - المساعدة

💡 *مثال:* أرسل BTC للتحليل السريع

🌟 *البوت يعمل 24/7 مجاناً!*
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل سريع", callback_data="quick_analysis")],
            [InlineKeyboardButton("📋 قائمة المراقبة", callback_data="show_watchlist")],
            [InlineKeyboardButton("❓ المساعدة", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown', reply_markup=reply_markup)

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تحليل عملة محددة"""
        try:
            if not context.args:
                await update.message.reply_text("❌ يرجى تحديد رمز العملة\nمثال: /analyze BTC")
                return
                
            symbol = context.args[0].upper()
            
            # إرسال رسالة انتظار
            waiting_msg = await update.message.reply_text(f"🔍 جاري تحليل {symbol}...\nيرجى الانتظار...")
            
            # إجراء التحليل الشامل
            analysis = await self.analyzer.comprehensive_analysis(symbol)
            
            if not analysis:
                await waiting_msg.edit_text(f"❌ لم أتمكن من العثور على بيانات لـ {symbol}\nتأكد من صحة رمز العملة")
                return
                
            # تنسيق التقرير
            report = await self.format_analysis_report(symbol, analysis)
            
            # إنشاء لوحة الأزرار
            keyboard = [
                [InlineKeyboardButton("👁️ إضافة للمراقبة", callback_data=f"watch_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            try:
                await waiting_msg.edit_text(f"❌ حدث خطأ أثناء تحليل {symbol}")
            except:
                await update.message.reply_text(f"❌ حدث خطأ أثناء تحليل {symbol}")

    async def format_analysis_report(self, symbol: str, analysis: Dict) -> str:
        """تنسيق تقرير التحليل"""
        try:
            if not analysis:
                return "❌ لا توجد بيانات متاحة"
                
            # الحصول على بيانات الإطار الزمني اليومي
            daily_data = analysis.get('1d', {})
            hourly_data = analysis.get('1h', {})
            
            if not daily_data and not hourly_data:
                return "❌ لا توجد بيانات كافية للتحليل"
                
            # استخدام البيانات المتاحة
            data = daily_data if daily_data else hourly_data
            
            current_price = data.get('price', 0)
            trend = data.get('trend', 'غير محدد')
            strength = data.get('strength', 0)
            indicators = data.get('indicators', {})
            entry_exit = data.get('entry_exit', {})
            
            # تحديد التوصية الرئيسية
            if strength > 0.5:
                recommendation = "🟢 شراء قوي"
                confidence = "عالية"
            elif strength > 0.2:
                recommendation = "🔵 شراء"
                confidence = "متوسطة"
            elif strength < -0.5:
                recommendation = "🔴 بيع قوي"
                confidence = "عالية"
            elif strength < -0.2:
                recommendation = "🟠 بيع"
                confidence = "متوسطة"
            else:
                recommendation = "⚪ انتظار"
                confidence = "منخفضة"
            
            # نقاط الدخول والخروج
            entry_points = entry_exit.get('entry_points', {})
            targets = entry_exit.get('targets', {})
            stop_loss = entry_exit.get('stop_loss', {})
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.6f}"
            elif current_price < 1:
                price_str = f"${current_price:.4f}"
            else:
                price_str = f"${current_price:.2f}"
            
            report = f"""
🎯 *تحليل {symbol}/USDT*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* {price_str}
📊 *الاتجاه:* {trend}
💪 *القوة:* {abs(strength)*100:.1f}%
🎲 *التوصية:* {recommendation}
🔮 *الثقة:* {confidence}

━━━━━━━━━━━━━━━━━━━━
📈 *المؤشرات الفنية:*

🔸 *RSI:* {indicators.get('rsi', 0):.1f}
🔸 *MACD:* {'إيجابي' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'سلبي'}
🔸 *MA20:* ${indicators.get('sma_20', 0):.6f}
🔸 *MA50:* ${indicators.get('sma_50', 0):.6f}

━━━━━━━━━━━━━━━━━━━━
🎯 *نقاط التداول:*
            """
            
            if recommendation.startswith("🟢") or recommendation.startswith("🔵"):
                # توصية شراء
                buy_zones = entry_points.get('buy_zones', [current_price])
                buy_targets = targets.get('buy_targets', [current_price * 1.03, current_price * 1.07, current_price * 1.15])
                
                report += f"""
📌 *مناطق الشراء:*
🔸 الأولى: ${min(buy_zones):.6f}
🔸 الثانية: ${max(buy_zones):.6f}

🎯 *الأهداف:*
🥇 الأول: ${buy_targets[0]:.6f} (+3%)
🥈 الثاني: ${buy_targets[1]:.6f} (+7%)
🥉 الثالث: ${buy_targets[2]:.6f} (+15%)

🛑 *وقف الخسارة:* ${stop_loss.get('buy', current_price * 0.92):.6f}
                """
                
            elif recommendation.startswith("🔴") or recommendation.startswith("🟠"):
                # توصية بيع
                sell_zones = entry_points.get('sell_zones', [current_price])
                sell_targets = targets.get('sell_targets', [current_price * 0.97, current_price * 0.93, current_price * 0.85])
                
                report += f"""
📌 *مناطق البيع:*
🔸 الأولى: ${min(sell_zones):.6f}
🔸 الثانية: ${max(sell_zones):.6f}

🎯 *الأهداف:*
🥇 الأول: ${sell_targets[0]:.6f} (-3%)
🥈 الثاني: ${sell_targets[1]:.6f} (-7%)
🥉 الثالث: ${sell_targets[2]:.6f} (-15%)

🛑 *وقف الخسارة:* ${stop_loss.get('sell', current_price * 1.08):.6f}
                """
            else:
                report += """
⚪ *توصية الانتظار:*
السوق في حالة محايدة حالياً
انتظر إشارات أوضح قبل الدخول
                """
            
            report += f"""
━━━━━━━━━━━━━━━━━━━━
⚠️ *تنبيه:* هذا التحليل للأغراض التعليمية فقط
⏰ *التحديث:* {datetime.now().strftime('%H:%M:%S')}
🤖 *البوت متاح 24/7*
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"❌ حدث خطأ في تنسيق التقرير لـ {symbol}"

    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض قائمة المراقبة"""
        try:
            user_id = update.effective_user.id
            watchlist = self.user_watchlists.get(user_id, [])
            
            if not watchlist:
                await update.message.reply_text("📋 قائمة المراقبة فارغة\nاستخدم /watch [SYMBOL] لإضافة عملة")
                return
                
            # إنشاء لوحة أزرار للعملات
            keyboard = []
            for i in range(0, len(watchlist), 2):
                row = []
                for j in range(2):
                    if i + j < len(watchlist):
                        coin = watchlist[i + j]
                        row.append(InlineKeyboardButton(f"📊 {coin}", callback_data=f"analyze_{coin}"))
                keyboard.append(row)
                
            keyboard.append([InlineKeyboardButton("🔄 تحديث الكل", callback_data="update_all_watchlist")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"📋 *قائمة المراقبة ({len(watchlist)} عملة):*\n\n"
            message += " • ".join(watchlist)
            
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in watchlist: {e}")
            await update.message.reply_text("❌ حدث خطأ في عرض قائمة المراقبة")

    async def watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إضافة عملة للمراقبة"""
        try:
            if not context.args:
                await update.message.reply_text("❌ يرجى تحديد رمز العملة\nمثال: /watch BTC")
                return
                
            symbol = context.args[0].upper()
            user_id = update.effective_user.id
            
            if user_id not in self.user_watchlists:
                self.user_watchlists[user_id] = []
                
            if symbol in self.user_watchlists[user_id]:
                await update.message.reply_text(f"👁️ {symbol} موجود بالفعل في قائمة المراقبة")
                return
                
            # التحقق من صحة العملة
            test_data = await self.analyzer.get_price_data(symbol)
            if test_data is None:
                await update.message.reply_text(f"❌ لم أتمكن من العثور على {symbol}")
                return
                
            self.user_watchlists[user_id].append(symbol)
            await update.message.reply_text(f"✅ تم إضافة {symbol} لقائمة المراقبة")
            
        except Exception as e:
            logger.error(f"Error in watch command: {e}")
            await update.message.reply_text("❌ حدث خطأ في إضافة العملة")

    async def remove_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إزالة عملة من المراقبة"""
        try:
            if not context.args:
                await update.message.reply_text("❌ يرجى تحديد رمز العملة\nمثال: /remove BTC")
                return
                
            symbol = context.args[0].upper()
            user_id = update.effective_user.id
            
            if user_id in self.user_watchlists and symbol in self.user_watchlists[user_id]:
                self.user_watchlists[user_id].remove(symbol)
                await update.message.reply_text(f"✅ تم إزالة {symbol} من قائمة المراقبة")
            else:
                await update.message.reply_text(f"❌ {symbol} غير موجود في قائمة المراقبة")
                
        except Exception as e:
            logger.error(f"Error in remove command: {e}")
            await update.message.reply_text("❌ حدث خطأ في إزالة العملة")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأزرار"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith("analyze_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                await self.analyze_command(update, context)
                
            elif data.startswith("watch_"):
                symbol = data.split("_")[1]
                user_id = update.effective_user.id
                
                if user_id not in self.user_watchlists:
                    self.user_watchlists[user_id] = []
                    
                if symbol not in self.user_watchlists[user_id]:
                    self.user_watchlists[user_id].append(symbol)
                    await query.edit_message_text(f"✅ تم إضافة {symbol} لقائمة المراقبة")
                else:
                    await query.edit_message_text(f"👁️ {symbol} موجود بالفعل في قائمة المراقبة")
                    
            elif data.startswith("refresh_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                await self.analyze_command(update, context)
                
            elif data == "show_watchlist":
                await self.watchlist_command(update, context)
                
            elif data == "help":
                await self.help_command(update, context)
                
        except Exception as e:
            logger.error(f"Error in button callback: {e}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر المساعدة"""
        help_text = """
📚 *دليل استخدام البوت:*

🔍 *أوامر التحليل:*
• `/analyze BTC` - تحليل فني شامل
• أرسل `BTC` مباشرة - تحليل سريع

👁️ *أوامر المراقبة:*
• `/watch BTC` - إضافة للمراقبة
• `/watchlist` - عرض القائمة
• `/remove BTC` - إزالة من القائمة

📊 *معلومات التحليل:*
• RSI: مؤشر القوة النسبية (30-70 طبيعي)
• MACD: تقارب وتباعد المتوسطات
• MA: المتوسطات المتحركة (20, 50)
• BB: نطاقات بولينجر

🎯 *رموز التوصيات:*
🟢 شراء قوي | 🔵 شراء
🔴 بيع قوي | 🟠 بيع
⚪ انتظار

⚠️ *تحذير مهم:*
التحليل للأغراض التعليمية فقط
قم بإجراء بحثك الخاص قبل الاستثمار

🌟 *البوت متاح 24/7 مجاناً!*
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص العادية"""
        try:
            text = update.message.text.upper().strip()
            
            # التحقق من رمز العملة
            if len(text) <= 10 and text.isalpha():
                context.args = [text]
                await self.analyze_command(update, context)
            else:
                await update.message.reply_text("💡 أرسل رمز العملة للتحليل (مثل: BTC)\nأو استخدم /help للمساعدة")
                
        except Exception as e:
            logger.error(f"Error in text handler: {e}")
            await update.message.reply_text("❌ حدث خطأ في معالجة الرسالة")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء العام"""
        logger.error(f"Exception while handling an update: {context.error}")

    async def setup_application(self):
        """إعداد التطبيق"""
        try:
            # إنشاء التطبيق
            self.application = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("analyze", self.analyze_command))
            self.application.add_handler(CommandHandler("watch", self.watch_command))
            self.application.add_handler(CommandHandler("watchlist", self.watchlist_command))
            self.application.add_handler(CommandHandler("remove", self.remove_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CallbackQueryHandler(self.button_callback))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            
            # إضافة معالج الأخطاء
            self.application.add_error_handler(self.error_handler)
            
            # تهيئة التطبيق
            await self.application.initialize()
            await self.application.start()
            
            return self.application
            
        except Exception as e:
            logger.error(f"Error setting up application: {e}")
            raise

    def run_webhook(self):
        """تشغيل البوت في وضع الـ webhook"""
        try:
            # إنشاء Flask app
            app = Flask(__name__)
            
            @app.route('/')
            def health_check():
                return "Bot is running! 🤖", 200

            @app.route('/health')
            def health():
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

            @app.route('/webhook', methods=['POST'])
            def webhook():
                try:
                    if not self.application:
                        logger.error("Application not initialized")
                        return "Application not ready", 500
                        
                    # الحصول على البيانات
                    json_data = request.get_json()
                    if not json_data:
                        return "No data", 400
                    
                    # إنشاء Update object
                    update = Update.de_json(json_data, self.application.bot)
                    if not update:
                        return "Invalid update", 400
                    
                    # معالجة التحديث
                    asyncio.create_task(self.application.process_update(update))
                    
                    return "OK", 200
                    
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
                    return "Error", 500

            # إعداد التطبيق بشكل متزامن
            async def setup():
                await self.setup_application()
                
                # إعداد الـ webhook
                if self.webhook_url:
                    webhook_endpoint = f"{self.webhook_url}/webhook"
                    await self.application.bot.set_webhook(webhook_endpoint)
                    logger.info(f"Webhook set to: {webhook_endpoint}")
                
                # إرسال تنبيه للإدمن
                if self.admin_id:
                    try:
                        await self.application.bot.send_message(
                            chat_id=self.admin_id,
                            text="🚀 تم تشغيل البوت بنجاح على Render!\n⏰ الوقت: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                    except Exception as e:
                        logger.error(f"Error sending startup message: {e}")
            
            # تشغيل الإعداد
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup())
            
            # تشغيل Flask
            port = int(os.environ.get('PORT', 10000))
            logger.info(f"🤖 Bot running on webhook mode, port: {port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            
        except Exception as e:
            logger.error(f"Error in webhook mode: {e}")
            raise

    def run_polling(self):
        """تشغيل البوت في وضع الـ polling"""
        try:
            async def main():
                await self.setup_application()
                
                # إرسال تنبيه للإدمن
                if self.admin_id:
                    try:
                        await self.application.bot.send_message(
                            chat_id=self.admin_id,
                            text="🚀 تم تشغيل البوت بنجاح في وضع Polling!\n⏰ الوقت: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                    except Exception as e:
                        logger.error(f"Error sending startup message: {e}")
                
                logger.info("🤖 Bot running in polling mode...")
                
                # بدء التشغيل
                await self.application.updater.start_polling(drop_pending_updates=True)
                await self.application.updater.idle()

            # تشغيل الحلقة الرئيسية
            asyncio.run(main())
            
        except Exception as e:
            logger.error(f"Error in polling mode: {e}")
            raise

    def run(self):
        """تشغيل البوت"""
        try:
            # تحديد وضع التشغيل
            if self.webhook_url:
                # وضع الـ webhook للنشر على Render
                self.run_webhook()
            else:
                # وضع الـ polling للتطوير المحلي
                self.run_polling()
                
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

# التشغيل الرئيسي
if __name__ == "__main__":
    try:
        # التحقق من متغيرات البيئة
        if not os.getenv('BOT_TOKEN'):
            print("❌ BOT_TOKEN غير موجود في متغيرات البيئة")
            exit(1)
            
        # إنشاء وتشغيل البوت
        bot = CryptoTelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل البوت: {e}")
        exit(1)
