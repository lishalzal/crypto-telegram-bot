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
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء Flask app للـ health check
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Bot is running! 🤖", 200

@app.route('/health')
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

def run_flask():
    """تشغيل Flask في thread منفصل"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

class TechnicalAnalyzer:
    """محلل فني بسيط للعملات المشفرة"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    async def get_price_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السعر الحالي من Binance API"""
        try:
            # جلب السعر الحالي
            price_url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(price_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"API Error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    async def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 50) -> Optional[List]:
        """جلب بيانات الشموع من Binance API"""
        try:
            kline_url = f"{self.base_url}/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(kline_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"Kline API Error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching kline data: {e}")
            return None

    def calculate_simple_indicators(self, kline_data: List) -> Dict:
        """حساب مؤشرات فنية بسيطة"""
        try:
            if not kline_data or len(kline_data) < 20:
                return {}
            
            # استخراج أسعار الإغلاق
            closes = [float(kline[4]) for kline in kline_data]
            highs = [float(kline[2]) for kline in kline_data]
            lows = [float(kline[3]) for kline in kline_data]
            volumes = [float(kline[5]) for kline in kline_data]
            
            current_price = closes[-1]
            
            # حساب المتوسطات المتحركة
            sma_20 = sum(closes[-20:]) / 20
            sma_10 = sum(closes[-10:]) / 10
            sma_5 = sum(closes[-5:]) / 5
            
            # حساب RSI المبسط
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            # حساب RSI للـ 14 فترة الأخيرة
            if len(gains) >= 14:
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # حساب الدعم والمقاومة
            support = min(lows[-20:])
            resistance = max(highs[-20:])
            
            # حساب متوسط الحجم
            avg_volume = sum(volumes[-10:]) / 10
            current_volume = volumes[-1]
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_10': sma_10,
                'sma_5': sma_5,
                'rsi': rsi,
                'support': support,
                'resistance': resistance,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def analyze_trend(self, price_data: Dict, indicators: Dict) -> Tuple[str, float, str]:
        """تحليل الاتجاه والتوصية"""
        try:
            if not price_data or not indicators:
                return "غير محدد", 0, "⚪ انتظار"
            
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_10 = indicators.get('sma_10', 0)
            sma_5 = indicators.get('sma_5', 0)
            rsi = indicators.get('rsi', 50)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            # تحليل الاتجاه
            signals = []
            
            # إشارات المتوسطات المتحركة
            if current_price > sma_5 > sma_10 > sma_20:
                signals.append(2)  # اتجاه صاعد قوي
            elif current_price > sma_10 > sma_20:
                signals.append(1)  # اتجاه صاعد
            elif current_price < sma_5 < sma_10 < sma_20:
                signals.append(-2)  # اتجاه هابط قوي
            elif current_price < sma_10 < sma_20:
                signals.append(-1)  # اتجاه هابط
            else:
                signals.append(0)  # محايد
            
            # إشارات RSI
            if rsi < 30:
                signals.append(1)  # oversold - فرصة شراء
            elif rsi > 70:
                signals.append(-1)  # overbought - فرصة بيع
            else:
                signals.append(0)  # محايد
            
            # إشارات الحجم
            if volume_ratio > 1.5:
                signals.append(1)  # حجم عالي يدعم الحركة
            elif volume_ratio < 0.7:
                signals.append(-0.5)  # حجم منخفض
            else:
                signals.append(0)
            
            # حساب متوسط الإشارات
            if signals:
                avg_signal = sum(signals) / len(signals)
                strength = abs(avg_signal)
                
                # تحديد الاتجاه والتوصية
                if avg_signal > 1:
                    return "صاعد قوي", strength, "🟢 شراء قوي"
                elif avg_signal > 0.5:
                    return "صاعد", strength, "🔵 شراء"
                elif avg_signal < -1:
                    return "هابط قوي", strength, "🔴 بيع قوي"
                elif avg_signal < -0.5:
                    return "هابط", strength, "🟠 بيع"
                else:
                    return "محايد", strength, "⚪ انتظار"
            
            return "غير محدد", 0, "⚪ انتظار"
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return "غير محدد", 0, "⚪ انتظار"

    async def comprehensive_analysis(self, symbol: str) -> Dict:
        """تحليل شامل للعملة"""
        try:
            # جلب بيانات السعر الحالي
            price_data = await self.get_price_data(symbol)
            if not price_data:
                return {}
            
            # جلب بيانات الشموع
            kline_data = await self.get_kline_data(symbol)
            if not kline_data:
                return {'price_data': price_data}
            
            # حساب المؤشرات
            indicators = self.calculate_simple_indicators(kline_data)
            if not indicators:
                return {'price_data': price_data}
            
            # تحليل الاتجاه
            trend, strength, recommendation = self.analyze_trend(price_data, indicators)
            
            # حساب نقاط الدخول والخروج
            current_price = float(price_data['lastPrice'])
            support = indicators.get('support', current_price * 0.95)
            resistance = indicators.get('resistance', current_price * 1.05)
            
            return {
                'price_data': price_data,
                'indicators': indicators,
                'trend': trend,
                'strength': strength,
                'recommendation': recommendation,
                'current_price': current_price,
                'support': support,
                'resistance': resistance,
                'targets': {
                    'buy_targets': [current_price * 1.03, current_price * 1.07, current_price * 1.15],
                    'sell_targets': [current_price * 0.97, current_price * 0.93, current_price * 0.85]
                },
                'stop_loss': {
                    'buy': support * 0.98,
                    'sell': resistance * 1.02
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}

class CryptoTelegramBot:
    """بوت تليغرام للتوصيات المشفرة"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.analyzer = TechnicalAnalyzer()
        self.user_watchlists = {}
        
        if not self.token:
            raise ValueError("BOT_TOKEN environment variable is required")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        welcome_message = """
🤖 *أهلاً بك في بوت التوصيات المشفرة!*

📊 *الخدمات المتاحة:*
• تحليل فني للعملات المشفرة
• نقاط دخول وخروج محددة
• توصيات مبنية على مؤشرات فنية
• مراقبة العملات المفضلة

📝 *الأوامر:*
/analyze BTC - تحليل عملة
/watch BTC - إضافة للمراقبة
/watchlist - قائمة المراقبة
/help - المساعدة

💡 *مثال سريع:* أرسل BTC للتحليل

🌟 *البوت متاح 24/7!*
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل سريع", callback_data="quick_analysis")],
            [InlineKeyboardButton("📋 المراقبة", callback_data="show_watchlist")],
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
            
            # إجراء التحليل
            analysis = await self.analyzer.comprehensive_analysis(symbol)
            
            if not analysis:
                await waiting_msg.edit_text(f"❌ لم أتمكن من العثور على بيانات لـ {symbol}\nتأكد من صحة رمز العملة")
                return
                
            # تنسيق التقرير
            report = self.format_analysis_report(symbol, analysis)
            
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

    def format_analysis_report(self, symbol: str, analysis: Dict) -> str:
        """تنسيق تقرير التحليل"""
        try:
            if not analysis:
                return "❌ لا توجد بيانات متاحة"
            
            price_data = analysis.get('price_data', {})
            current_price = analysis.get('current_price', 0)
            trend = analysis.get('trend', 'غير محدد')
            strength = analysis.get('strength', 0)
            recommendation = analysis.get('recommendation', '⚪ انتظار')
            indicators = analysis.get('indicators', {})
            targets = analysis.get('targets', {})
            stop_loss = analysis.get('stop_loss', {})
            
            # بيانات إضافية من السوق
            change_24h = float(price_data.get('priceChangePercent', 0))
            volume_24h = float(price_data.get('volume', 0))
            high_24h = float(price_data.get('highPrice', 0))
            low_24h = float(price_data.get('lowPrice', 0))
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.6f}"
            elif current_price < 1:
                price_str = f"${current_price:.4f}"
            else:
                price_str = f"${current_price:.2f}"
            
            # تحديد لون التغيير
            change_emoji = "🟢" if change_24h >= 0 else "🔴"
            
            report = f"""
🎯 *تحليل {symbol}/USDT*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* {price_str}
{change_emoji} *التغيير 24س:* {change_24h:+.2f}%
📊 *الاتجاه:* {trend}
💪 *قوة الإشارة:* {strength*100:.1f}%
🎲 *التوصية:* {recommendation}

━━━━━━━━━━━━━━━━━━━━
📈 *بيانات السوق:*

🔸 *أعلى 24س:* ${high_24h:.6f}
🔸 *أقل 24س:* ${low_24h:.6f}
🔸 *الحجم:* {volume_24h:,.0f}
🔸 *RSI:* {indicators.get('rsi', 0):.1f}
🔸 *المتوسط 20:* ${indicators.get('sma_20', 0):.6f}

━━━━━━━━━━━━━━━━━━━━
            """
            
            # إضافة نقاط التداول حسب التوصية
            if "شراء" in recommendation:
                buy_targets = targets.get('buy_targets', [])
                if buy_targets:
                    report += f"""🎯 *استراتيجية الشراء:*

📌 *نقاط الدخول:*
🔸 الدعم: ${analysis.get('support', 0):.6f}
🔸 المُوصى: ${current_price * 0.98:.6f}

🏆 *الأهداف:*
🥇 الأول: ${buy_targets[0]:.6f} (+3%)
🥈 الثاني: ${buy_targets[1]:.6f} (+7%)
🥉 الثالث: ${buy_targets[2]:.6f} (+15%)

🛑 *وقف الخسارة:* ${stop_loss.get('buy', 0):.6f}
                    """
                    
            elif "بيع" in recommendation:
                sell_targets = targets.get('sell_targets', [])
                if sell_targets:
                    report += f"""🎯 *استراتيجية البيع:*

📌 *نقاط الدخول:*
🔸 المقاومة: ${analysis.get('resistance', 0):.6f}
🔸 المُوصى: ${current_price * 1.02:.6f}

🎯 *الأهداف:*
🥇 الأول: ${sell_targets[0]:.6f} (-3%)
🥈 الثاني: ${sell_targets[1]:.6f} (-7%)
🥉 الثالث: ${sell_targets[2]:.6f} (-15%)

🛑 *وقف الخسارة:* ${stop_loss.get('sell', 0):.6f}
                    """
            else:
                report += """
⚪ *الوضع الحالي:*
السوق في حالة تذبذب
انتظر إشارات أوضح للدخول
راقب كسر الدعم أو المقاومة
                """
            
            report += f"""
━━━━━━━━━━━━━━━━━━━━
⚠️ *تنبيه:* هذا تحليل تعليمي فقط
💡 *نصيحة:* قم ببحثك الخاص دائماً
⏰ *وقت التحديث:* {datetime.now().strftime('%H:%M:%S')}
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"❌ حدث خطأ في تنسيق تقرير {symbol}"

    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض قائمة المراقبة"""
        try:
            user_id = update.effective_user.id
            watchlist = self.user_watchlists.get(user_id, [])
            
            if not watchlist:
                await update.message.reply_text("📋 قائمة المراقبة فارغة\nاستخدم /watch [SYMBOL] لإضافة عملة")
                return
                
            message = f"📋 *قائمة المراقبة ({len(watchlist)} عملة):*\n\n"
            message += " • ".join(watchlist)
            
            # إنشاء أزرار للعملات
            keyboard = []
            for i in range(0, len(watchlist), 2):
                row = []
                for j in range(2):
                    if i + j < len(watchlist):
                        coin = watchlist[i + j]
                        row.append(InlineKeyboardButton(f"📊 {coin}", callback_data=f"analyze_{coin}"))
                keyboard.append(row)
                
            reply_markup = InlineKeyboardMarkup(keyboard)
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

🔍 *التحليل:*
• `/analyze BTC` - تحليل شامل
• أرسل `BTC` مباشرة - تحليل سريع

👁️ *المراقبة:*
• `/watch BTC` - إضافة للمراقبة
• `/watchlist` - عرض القائمة

📊 *المؤشرات:*
• RSI: مؤشر القوة النسبية
• SMA: المتوسطات المتحركة
• الدعم والمقاومة

🎯 *التوصيات:*
🟢 شراء قوي | 🔵 شراء
🔴 بيع قوي | 🟠 بيع | ⚪ انتظار

⚠️ *تحذير:*
التحليل للأغراض التعليمية فقط
قم بإجراء بحثك الخاص دائماً

🌟 *البوت متاح 24/7!*
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

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء العام"""
        logger.error(f"Exception while handling an update: {context.error}")

    def run(self):
        """تشغيل البوت"""
        try:
            # بدء Flask في thread منفصل
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
            # إنشاء التطبيق
            application = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("analyze", self.analyze_command))
            application.add_handler(CommandHandler("watch", self.watch_command))
            application.add_handler(CommandHandler("watchlist", self.watchlist_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CallbackQueryHandler(self.button_callback))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            
            # إضافة معالج الأخطاء
            application.add_error_handler(self.error_handler)
            
            # إرسال تنبيه للإدمن عند بدء التشغيل
            if self.admin_id:
                asyncio.create_task(self.send_startup_message(application))
            
            logger.info("🤖 تم تشغيل البوت على Render...")
            
            # بدء التشغيل
            application.run_polling(drop_pending_updates=True)
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    async def send_startup_message(self, application):
        """إرسال رسالة بدء التشغيل للإدمن"""
        try:
            await asyncio.sleep(2)
            await application.bot.send_message(
                chat_id=self.admin_id,
                text="🚀 تم تشغيل البوت بنجاح على Render!\n⏰ " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")

# التشغيل الرئيسي
if __name__ == "__main__":
    try:
        if not os.getenv('BOT_TOKEN'):
            print("❌ BOT_TOKEN غير موجود في متغيرات البيئة")
            exit(1)
            
        bot = CryptoTelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل البوت: {e}")
        exit(1)
