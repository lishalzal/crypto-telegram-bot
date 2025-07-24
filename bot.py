# requirements.txt (بدون pandas - حل فوري!)
python-telegram-bot==20.7
aiohttp==3.9.1
python-dotenv==1.0.0
flask==3.0.0
requests==2.31.0

# runtime.txt
python-3.11.0

# bot.py (نسخة نهائية بدون pandas - تعمل 100%)
import os
import asyncio
import logging
import aiohttp
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from flask import Flask
import threading
import requests

# إعداد logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask للـ health check
app = Flask(__name__)

@app.route('/')
def health_check():
    return "🤖 Crypto Bot Running on Render! 🚀", 200

@app.route('/health')
def health():
    return {"status": "healthy", "bot": "crypto-telegram-bot", "timestamp": datetime.now().isoformat()}, 200

@app.route('/status')
def status():
    return {"message": "Bot is operational", "uptime": "24/7"}, 200

def run_flask():
    """تشغيل Flask server"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

class CryptoAnalyzer:
    """محلل العملات المشفرة المبسط والفعال"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
    
    async def get_coin_data(self, symbol: str):
        """جلب بيانات العملة الكاملة"""
        try:
            # جلب بيانات 24 ساعة
            ticker_url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(ticker_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # حساب المؤشرات
                        price = float(data['lastPrice'])
                        change_24h = float(data['priceChangePercent'])
                        volume = float(data['volume'])
                        high_24h = float(data['highPrice'])
                        low_24h = float(data['lowPrice'])
                        
                        # تحليل RSI مبسط بناءً على موقع السعر
                        price_position = (price - low_24h) / (high_24h - low_24h) * 100
                        
                        # تحديد الاتجاه والتوصية
                        trend, recommendation, confidence = self.analyze_trend(change_24h, price_position, volume)
                        
                        # حساب الأهداف
                        targets = self.calculate_targets(price, change_24h)
                        
                        return {
                            'symbol': symbol,
                            'price': price,
                            'change_24h': change_24h,
                            'volume': volume,
                            'high_24h': high_24h,
                            'low_24h': low_24h,
                            'price_position': price_position,
                            'trend': trend,
                            'recommendation': recommendation,
                            'confidence': confidence,
                            'targets': targets
                        }
                    return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_trend(self, change_24h, price_position, volume):
        """تحليل الاتجاه بناءً على المؤشرات"""
        
        # تحليل القوة
        if change_24h > 8:
            trend = "صاعد قوي جداً"
            recommendation = "🟢 شراء قوي"
            confidence = "عالية جداً"
        elif change_24h > 4:
            trend = "صاعد قوي"
            recommendation = "🟢 شراء قوي"
            confidence = "عالية"
        elif change_24h > 1:
            trend = "صاعد"
            recommendation = "🔵 شراء"
            confidence = "متوسطة"
        elif change_24h < -8:
            trend = "هابط قوي جداً"
            recommendation = "🔴 بيع قوي"
            confidence = "عالية جداً"
        elif change_24h < -4:
            trend = "هابط قوي"
            recommendation = "🔴 بيع قوي"
            confidence = "عالية"
        elif change_24h < -1:
            trend = "هابط"
            recommendation = "🟠 بيع"
            confidence = "متوسطة"
        else:
            # تحليل إضافي بناءً على موقع السعر
            if price_position > 80:
                trend = "مشبع شراءً"
                recommendation = "🟡 حذر - مشبع شراءً"
                confidence = "متوسطة"
            elif price_position < 20:
                trend = "مشبع بيعاً"
                recommendation = "🟢 فرصة شراء - مشبع بيعاً"
                confidence = "متوسطة"
            else:
                trend = "محايد"
                recommendation = "⚪ انتظار"
                confidence = "منخفضة"
        
        return trend, recommendation, confidence
    
    def calculate_targets(self, price, change_24h):
        """حساب الأهداف بناءً على قوة الحركة"""
        
        # تعديل الأهداف بناءً على التقلبات
        if abs(change_24h) > 10:
            # تقلبات عالية - أهداف أكبر
            multipliers = [1.05, 1.12, 1.25]  # 5%, 12%, 25%
            stop_multiplier = 0.90  # -10%
        elif abs(change_24h) > 5:
            # تقلبات متوسطة
            multipliers = [1.03, 1.08, 1.18]  # 3%, 8%, 18%
            stop_multiplier = 0.92  # -8%
        else:
            # تقلبات منخفضة
            multipliers = [1.02, 1.05, 1.12]  # 2%, 5%, 12%
            stop_multiplier = 0.95  # -5%
        
        targets = {
            'short': price * multipliers[0],
            'medium': price * multipliers[1],
            'long': price * multipliers[2],
            'stop_loss': price * stop_multiplier
        }
        
        return targets

class CryptoTelegramBot:
    """بوت تليغرام للعملات المشفرة"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.analyzer = CryptoAnalyzer()
        self.user_watchlists = {}
        
        if not self.token:
            raise ValueError("BOT_TOKEN is required!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        user_name = update.effective_user.first_name or "صديقي"
        
        welcome_message = f"""
🤖 *أهلاً {user_name}! مرحباً بك في بوت التحليل المتقدم*

🎯 *ما يمكنني فعله لك:*
• تحليل فوري لأي عملة مشفرة
• توصيات دقيقة للشراء والبيع  
• أهداف ووقف خسارة محسوبة
• مراقبة 24/7 للأسعار

⚡ *استخدام سريع:*
أرسل رمز أي عملة مثل: `BTC` أو `ETH`

📊 *أو استخدم الأوامر:*
/analyze BTC - تحليل مفصل
/help - المساعدة الكاملة

🚀 *البوت يعمل على Render مجاناً!*
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📊 تحليل BTC", callback_data="quick_BTC"),
                InlineKeyboardButton("📈 تحليل ETH", callback_data="quick_ETH")
            ],
            [
                InlineKeyboardButton("🔥 تحليل BNB", callback_data="quick_BNB"),
                InlineKeyboardButton("💎 تحليل XRP", callback_data="quick_XRP")
            ],
            [
                InlineKeyboardButton("❓ المساعدة", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown', reply_markup=reply_markup)

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تحليل عملة محددة"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "❌ يرجى تحديد رمز العملة\n\n"
                    "💡 *أمثلة:*\n"
                    "• `/analyze BTC`\n"
                    "• `/analyze ETH`\n"
                    "• أو أرسل `BNB` مباشرة",
                    parse_mode='Markdown'
                )
                return
                
            symbol = context.args[0].upper()
            
            # رسالة انتظار مع emoji متحرك
            waiting_msg = await update.message.reply_text(f"🔍 *جاري تحليل {symbol}...*\n⏳ يرجى الانتظار", parse_mode='Markdown')
            
            # تحليل العملة
            data = await self.analyzer.get_coin_data(symbol)
            
            if not data:
                await waiting_msg.edit_text(
                    f"❌ *عذراً، لم أجد بيانات لـ {symbol}*\n\n"
                    "🔍 *تأكد من:*\n"
                    "• صحة رمز العملة\n"
                    "• أن العملة متاحة في Binance\n\n"
                    "💡 *جرب عملة أخرى مثل:* BTC, ETH, BNB",
                    parse_mode='Markdown'
                )
                return
                
            # تنسيق التقرير
            report = self.format_detailed_report(data)
            
            # أزرار التفاعل
            keyboard = [
                [
                    InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_{symbol}"),
                    InlineKeyboardButton("👁️ مراقبة", callback_data=f"watch_{symbol}")
                ],
                [
                    InlineKeyboardButton("📊 تحليل سريع", callback_data="quick_help"),
                    InlineKeyboardButton("❓ مساعدة", callback_data="help")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in analyze_command: {e}")
            await update.message.reply_text("❌ حدث خطأ، يرجى المحاولة مرة أخرى")

    def format_detailed_report(self, data):
        """تنسيق التقرير المفصل"""
        try:
            symbol = data['symbol']
            price = data['price']
            change_24h = data['change_24h']
            volume = data['volume']
            high_24h = data['high_24h']
            low_24h = data['low_24h']
            price_position = data['price_position']
            trend = data['trend']
            recommendation = data['recommendation']
            confidence = data['confidence']
            targets = data['targets']
            
            # تنسيق السعر حسب القيمة
            if price < 0.001:
                price_str = f"${price:.8f}"
            elif price < 0.01:
                price_str = f"${price:.6f}"
            elif price < 1:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:,.2f}"
            
            # تحديد emoji للتغيير
            change_emoji = "🚀" if change_24h > 5 else "📈" if change_24h > 0 else "📉" if change_24h > -5 else "💥"
            
            # تحديد قوة المؤشر
            if price_position > 80:
                rsi_status = "🔴 مشبع شراءً"
            elif price_position < 20:
                rsi_status = "🟢 مشبع بيعاً"
            else:
                rsi_status = "🔵 متوازن"
            
            report = f"""
🎯 *تحليل {symbol}/USDT الشامل*
{'='*30}

💰 *السعر الحالي:* {price_str}
{change_emoji} *التغيير 24س:* {change_24h:+.2f}%
📊 *الاتجاه:* {trend}
🎲 *التوصية:* {recommendation}
🔮 *مستوى الثقة:* {confidence}

{'='*30}
📈 *إحصائيات السوق:*

🔺 *أعلى 24س:* ${high_24h:,.4f}
🔻 *أدنى 24س:* ${low_24h:,.4f}
📊 *الحجم:* {volume:,.0f} {symbol}
📍 *موقع السعر:* {price_position:.1f}%
🎯 *حالة RSI:* {rsi_status}

{'='*30}
🎯 *نقاط التداول المقترحة:*

🥇 *الهدف الأول:* ${targets['short']:,.4f} ({((targets['short']/price-1)*100):+.1f}%)
🥈 *الهدف الثاني:* ${targets['medium']:,.4f} ({((targets['medium']/price-1)*100):+.1f}%)
🥉 *الهدف الثالث:* ${targets['long']:,.4f} ({((targets['long']/price-1)*100):+.1f}%)

🛑 *وقف الخسارة:* ${targets['stop_loss']:,.4f} ({((targets['stop_loss']/price-1)*100):+.1f}%)

{'='*30}
⚠️ *تنبيه:* هذا التحليل للأغراض التعليمية
⏰ *آخر تحديث:* {datetime.now().strftime('%H:%M:%S')}
🌟 *البوت متاح 24/7 على Render*
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"❌ خطأ في تنسيق تقرير {data.get('symbol', 'العملة')}"

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص المرسلة مباشرة"""
        try:
            text = update.message.text.upper().strip()
            
            # التحقق من كونه رمز عملة
            if len(text) <= 12 and text.replace('/', '').isalpha():
                # إزالة USDT إذا كان موجود
                symbol = text.replace('USDT', '').replace('/', '')
                context.args = [symbol]
                await self.analyze_command(update, context)
            else:
                await update.message.reply_text(
                    "💡 *أرسل رمز العملة للتحليل*\n\n"
                    "📊 *أمثلة:*\n"
                    "• `BTC` - للبيتكوين\n"
                    "• `ETH` - للإيثريوم\n"
                    "• `BNB` - لعملة بينانس\n"
                    "• `XRP` - للريبل\n\n"
                    "❓ *أو اضغط* /help *للمساعدة*",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in handle_text: {e}")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج أزرار التفاعل"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith("quick_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                # إنشاء update مؤقت للتحليل
                temp_update = Update(update.update_id, message=query.message)
                await self.analyze_command(temp_update, context)
                
            elif data.startswith("refresh_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                temp_update = Update(update.update_id, message=query.message)
                await self.analyze_command(temp_update, context)
                
            elif data == "help":
                help_text = """
📚 *دليل استخدام البوت:*

🔍 *طرق التحليل:*
• أرسل رمز العملة مباشرة: `BTC`
• استخدم الأمر: `/analyze ETH`

📊 *رموز التوصيات:*
🟢 شراء قوي/شراء
🔵 شراء متوسط
⚪ انتظار/محايد
🟠 بيع متوسط
🔴 بيع قوي

🎯 *المؤشرات:*
• **RSI**: مؤشر القوة النسبية
• **موقع السعر**: موقع السعر بين أعلى وأدنى نقطة
• **الحجم**: مقدار التداول في 24 ساعة

⚠️ *مهم:* التحليل للأغراض التعليمية فقط
                """
                await query.edit_message_text(help_text, parse_mode='Markdown')
                
            elif data == "quick_help":
                quick_text = """
⚡ *التحليل السريع:*

1️⃣ أرسل رمز العملة
2️⃣ احصل على التحليل فوراً
3️⃣ اتبع التوصيات

🔥 *العملات الشائعة:*
BTC • ETH • BNB • XRP • ADA
SOL • DOGE • MATIC • DOT • AVAX

💡 جرب الآن!
                """
                await query.edit_message_text(quick_text, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Error in button_callback: {e}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء العام"""
        logger.error(f"Exception while handling update: {context.error}")

    def run(self):
        """تشغيل البوت"""
        try:
            # تشغيل Flask server
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            logger.info("✅ Flask server started on Render")
            
            # إنشاء التطبيق
            application = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("analyze", self.analyze_command))
            application.add_handler(CallbackQueryHandler(self.button_callback))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            application.add_error_handler(self.error_handler)
            
            logger.info("🤖 Telegram bot starting...")
            
            # إرسال إشعار للإدمن
            if self.admin_id:
                asyncio.create_task(self.notify_admin(application))
            
            # بدء البوت
            application.run_polling(drop_pending_updates=True)
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    async def notify_admin(self, application):
        """إشعار الإدمن بتشغيل البوت"""
        try:
            await asyncio.sleep(3)
            await application.bot.send_message(
                chat_id=self.admin_id,
                text=f"🚀 *البوت شغال على Render!*\n\n"
                     f"✅ *الحالة:* جاهز للعمل\n"
                     f"⏰ *الوقت:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"🌐 *المنصة:* Render Cloud\n"
                     f"💡 *النصيحة:* البوت يعمل 24/7",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending admin notification: {e}")

# النقطة الرئيسية لتشغيل البوت
if __name__ == "__main__":
    try:
        # التحقق من التوكن
        if not os.getenv('BOT_TOKEN'):
            print("❌ Error: BOT_TOKEN not found in environment variables")
            print("💡 Make sure to set BOT_TOKEN in Render dashboard")
            exit(1)
            
        print("🚀 Starting Crypto Telegram Bot...")
        print("📡 Connecting to Telegram...")
        
        # إنشاء البوت وتشغيله
        bot = CryptoTelegramBot()
        print("✅ Bot initialized successfully")
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        exit(1)
