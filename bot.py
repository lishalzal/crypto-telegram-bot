import os
import asyncio
import logging
import aiohttp
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import Dict, List, Optional, Tuple
from flask import Flask, request
import threading

# إعداد الـ logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """محلل فني مبسط للعملات المشفرة"""
    
    def __init__(self):
        self.timeframes = ['1h', '4h', '1d']
        
    async def get_price_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List]:
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
                            
                        # تحويل البيانات للنوع المناسب
                        processed_data = []
                        for row in data:
                            processed_data.append({
                                'timestamp': int(row[0]),
                                'open': float(row[1]),
                                'high': float(row[2]),
                                'low': float(row[3]),
                                'close': float(row[4]),
                                'volume': float(row[5])
                            })
                        
                        return processed_data
                    else:
                        logger.error(f"API Error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    def calculate_sma(self, data: List[Dict], period: int) -> float:
        """حساب المتوسط المتحرك البسيط"""
        if len(data) < period:
            return 0
        prices = [item['close'] for item in data[-period:]]
        return sum(prices) / len(prices)

    def calculate_rsi(self, data: List[Dict], period: int = 14) -> float:
        """حساب مؤشر القوة النسبية"""
        if len(data) < period + 1:
            return 50
            
        gains = []
        losses = []
        
        for i in range(1, len(data)):
            change = data[i]['close'] - data[i-1]['close']
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_technical_indicators(self, data: List[Dict]) -> Dict:
        """حساب المؤشرات الفنية"""
        try:
            if len(data) < 20:
                return {}
                
            current = data[-1]
            
            # المتوسطات المتحركة
            sma_20 = self.calculate_sma(data, 20)
            sma_50 = self.calculate_sma(data, 50) if len(data) >= 50 else sma_20
            
            # RSI
            rsi = self.calculate_rsi(data)
            
            # الدعم والمقاومة
            recent_data = data[-20:]
            support = min(item['low'] for item in recent_data)
            resistance = max(item['high'] for item in recent_data)
            
            # نطاقات بولينجر المبسطة
            prices = [item['close'] for item in data[-20:]]
            variance = sum((price - sma_20) ** 2 for price in prices) / 20
            std_dev = variance ** 0.5
            
            bb_upper = sma_20 + (2 * std_dev)
            bb_lower = sma_20 - (2 * std_dev)
            
            return {
                'close': current['close'],
                'volume': current['volume'],
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'support': support,
                'resistance': resistance,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': sma_20
            }
            
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
                    
            # تحليل RSI
            rsi = indicators.get('rsi', 50)
            if 30 < rsi < 70:
                signals.append(0)  # محايد
            elif rsi <= 30:
                signals.append(1)  # oversold
            else:
                signals.append(-1)  # overbought
                    
            # تحليل نطاقات بولينجر
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            if close and bb_upper and bb_lower:
                if close <= bb_lower:
                    signals.append(1)
                elif close >= bb_upper:
                    signals.append(-1)
                    
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

    async def comprehensive_analysis(self, symbol: str) -> Dict:
        """تحليل شامل للعملة"""
        analysis_results = {}
        
        for timeframe in self.timeframes:
            try:
                data = await self.get_price_data(symbol, timeframe)
                if data and len(data) > 50:
                    indicators = self.calculate_technical_indicators(data)
                    if indicators:
                        trend, strength = self.analyze_trend(indicators)
                        
                        current_price = indicators.get('close', 0)
                        support = indicators.get('support', current_price * 0.95)
                        resistance = indicators.get('resistance', current_price * 1.05)
                        
                        entry_exit = {
                            'entry_points': {
                                'buy_zones': [support, current_price * 0.98],
                                'sell_zones': [resistance, current_price * 1.02]
                            },
                            'stop_loss': {
                                'buy': current_price * 0.92,
                                'sell': current_price * 1.08
                            },
                            'targets': {
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
                        }
                        
                        analysis_results[timeframe] = {
                            'trend': trend,
                            'strength': strength,
                            'indicators': indicators,
                            'entry_exit': entry_exit,
                            'price': current_price,
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
        self.webhook_url = os.getenv('WEBHOOK_URL')
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
🔸 *MA20:* ${indicators.get('sma_20', 0):.6f}
🔸 *MA50:* ${indicators.get('sma_50', 0):.6f}
🔸 *الدعم:* ${indicators.get('support', 0):.6f}
🔸 *المقاومة:* ${indicators.get('resistance', 0):.6f}

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
• MA: المتوسطات المتحركة (20, 50)
• الدعم والمقاومة

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

    def run(self):
        """تشغيل البوت"""
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
            
            if self.webhook_url:
                # وضع الـ webhook للنشر على Render
                self.run_webhook()
            else:
                # وضع الـ polling للتطوير المحلي
                self.run_polling()
                
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
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
                    # الحصول على البيانات
                    json_data = request.get_json()
                    logger.info(f"Received webhook data: {json_data}")
                    
                    if not json_data:
                        logger.error("No JSON data received")
                        return "No data", 400
                    
                    # إنشاء Update object
                    update = Update.de_json(json_data, self.application.bot)
                    if not update:
                        logger.error("Failed to create Update object")
                        return "Invalid update", 400
                    
                    # معالجة التحديث بشكل متزامن
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.application.process_update(update))
                    finally:
                        loop.close()
                    
                    return "OK", 200
                    
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
                    return "Error", 500

            # إعداد الـ webhook
            async def setup_webhook():
                await self.application.initialize()
                await self.application.start()
                
                webhook_endpoint = f"{self.webhook_url}/webhook"
                
                # حذف webhook القديم أولاً
                await self.application.bot.delete_webhook()
                await asyncio.sleep(1)
                
                # إعداد webhook جديد
                webhook_set = await self.application.bot.set_webhook(
                    url=webhook_endpoint,
                    allowed_updates=["message", "callback_query"]
                )
                
                if webhook_set:
                    logger.info(f"✅ Webhook set successfully to: {webhook_endpoint}")
                else:
                    logger.error("❌ Failed to set webhook")
                
                # التحقق من إعدادات webhook
                webhook_info = await self.application.bot.get_webhook_info()
                logger.info(f"Webhook info: {webhook_info}")
                
                # إرسال تنبيه للإدمن
                if self.admin_id:
                    try:
                        await self.application.bot.send_message(
                            chat_id=self.admin_id,
                            text=f"🚀 تم تشغيل البوت بنجاح على Render!\n⏰ الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n🌐 Webhook: {webhook_endpoint}"
                        )
                    except Exception as e:
                        logger.error(f"Error sending startup message: {e}")
            
            # تشغيل الإعداد
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup_webhook())
            
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
                await self.application.initialize()
                
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
                await self.application.run_polling(drop_pending_updates=True)

            # تشغيل الحلقة الرئيسية
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
            
        # إنشاء وتشغيل البوت
        bot = CryptoTelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل البوت: {e}")
        exit(1)
