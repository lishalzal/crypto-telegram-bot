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

# إعداد الـ logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء Flask app للـ health check
app = Flask(__name__)

@app.route('/')
def health_check():
    return "🤖 Bot is running perfectly! ✅", 200

@app.route('/health')
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

def run_flask():
    """تشغيل Flask في thread منفصل"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

class SimpleCryptoAnalyzer:
    """محلل مبسط للعملات المشفرة"""
    
    async def get_simple_price(self, symbol: str):
        """جلب السعر الحالي والتغيير"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'symbol': symbol,
                            'price': float(data['lastPrice']),
                            'change_24h': float(data['priceChangePercent']),
                            'high_24h': float(data['highPrice']),
                            'low_24h': float(data['lowPrice']),
                            'volume': float(data['volume'])
                        }
                    return None
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def get_basic_analysis(self, symbol: str):
        """تحليل بسيط بناءً على السعر والتغيير"""
        price_data = await self.get_simple_price(symbol)
        if not price_data:
            return None
            
        change_24h = price_data['change_24h']
        price = price_data['price']
        
        # تحليل بسيط بناءً على التغيير
        if change_24h > 5:
            trend = "صاعد قوي"
            recommendation = "🟢 شراء قوي"
            confidence = "عالية"
        elif change_24h > 2:
            trend = "صاعد"
            recommendation = "🔵 شراء"
            confidence = "متوسطة"
        elif change_24h < -5:
            trend = "هابط قوي"
            recommendation = "🔴 بيع قوي"
            confidence = "عالية"
        elif change_24h < -2:
            trend = "هابط"
            recommendation = "🟠 بيع"
            confidence = "متوسطة"
        else:
            trend = "محايد"
            recommendation = "⚪ انتظار"
            confidence = "منخفضة"
        
        # حساب أهداف بسيطة
        targets = {
            'short': price * 1.03,  # +3%
            'medium': price * 1.07, # +7%
            'long': price * 1.15    # +15%
        }
        
        stop_loss = price * 0.92  # -8%
        
        return {
            'price_data': price_data,
            'trend': trend,
            'recommendation': recommendation,
            'confidence': confidence,
            'targets': targets,
            'stop_loss': stop_loss
        }

class SimpleTelegramBot:
    """بوت تليغرام مبسط للعملات المشفرة"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.admin_id = int(os.getenv('ADMIN_ID', '0'))
        self.analyzer = SimpleCryptoAnalyzer()
        self.user_watchlists = {}
        
        if not self.token:
            raise ValueError("BOT_TOKEN environment variable is required")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        welcome_message = """
🤖 *أهلاً بك في بوت التوصيات المشفرة!*

📊 *الخدمات المتاحة:*
• تحليل أسعار العملات الحية
• توصيات شراء وبيع
• مراقبة العملات المفضلة
• أهداف ووقف خسارة

📝 *طريقة الاستخدام:*
• أرسل رمز أي عملة: `BTC`
• أو استخدم: `/analyze BTC`

💡 *مثال:* أرسل `BNB` للحصول على تحليل فوري

🚀 *البوت يعمل على Render مجاناً 24/7!*
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل سريع", callback_data="quick_help")],
            [InlineKeyboardButton("📋 المساعدة", callback_data="help")]
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
            waiting_msg = await update.message.reply_text(f"🔍 جاري تحليل {symbol}...")
            
            # إجراء التحليل
            analysis = await self.analyzer.get_basic_analysis(symbol)
            
            if not analysis:
                await waiting_msg.edit_text(f"❌ لم أتمكن من العثور على {symbol}\nتأكد من صحة رمز العملة")
                return
                
            # تنسيق التقرير
            report = self.format_analysis_report(symbol, analysis)
            
            # إنشاء أزرار
            keyboard = [
                [InlineKeyboardButton("👁️ إضافة للمراقبة", callback_data=f"watch_{symbol}")],
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await waiting_msg.edit_text(report, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            await update.message.reply_text(f"❌ حدث خطأ أثناء تحليل العملة")

    def format_analysis_report(self, symbol: str, analysis: dict) -> str:
        """تنسيق تقرير التحليل"""
        try:
            price_data = analysis['price_data']
            
            # تنسيق السعر
            price = price_data['price']
            if price < 0.01:
                price_str = f"${price:.6f}"
            elif price < 1:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:.2f}"
            
            change_24h = price_data['change_24h']
            change_emoji = "📈" if change_24h >= 0 else "📉"
            
            targets = analysis['targets']
            
            report = f"""
🎯 *تحليل {symbol}/USDT*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر الحالي:* {price_str}
{change_emoji} *التغيير 24س:* {change_24h:+.2f}%
📊 *الاتجاه:* {analysis['trend']}
🎲 *التوصية:* {analysis['recommendation']}
🔮 *مستوى الثقة:* {analysis['confidence']}

━━━━━━━━━━━━━━━━━━━━
📈 *بيانات السوق:*

🔺 *أعلى 24س:* ${price_data['high_24h']:.4f}
🔻 *أدنى 24س:* ${price_data['low_24h']:.4f}
📊 *الحجم:* {price_data['volume']:,.0f}

━━━━━━━━━━━━━━━━━━━━
🎯 *الأهداف المقترحة:*

🥇 *الهدف الأول:* ${targets['short']:.4f} (+3%)
🥈 *الهدف الثاني:* ${targets['medium']:.4f} (+7%)
🥉 *الهدف الثالث:* ${targets['long']:.4f} (+15%)

🛑 *وقف الخسارة:* ${analysis['stop_loss']:.4f} (-8%)

━━━━━━━━━━━━━━━━━━━━
⚠️ *تنبيه:* للأغراض التعليمية فقط
⏰ *الوقت:* {datetime.now().strftime('%H:%M:%S')}
🌟 *متاح 24/7 على Render*
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"❌ حدث خطأ في تنسيق التقرير"

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص العادية"""
        try:
            text = update.message.text.upper().strip()
            
            # التحقق من رمز العملة
            if len(text) <= 10 and text.isalpha():
                context.args = [text]
                await self.analyze_command(update, context)
            else:
                await update.message.reply_text(
                    "💡 أرسل رمز العملة للتحليل\n"
                    "مثال: BTC, ETH, BNB\n"
                    "أو استخدم /start للمساعدة"
                )
                
        except Exception as e:
            logger.error(f"Error in text handler: {e}")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأزرار"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith("refresh_"):
                symbol = data.split("_")[1]
                context.args = [symbol]
                await self.analyze_command(update, context)
            elif data == "help":
                await query.edit_message_text(
                    "📚 *المساعدة:*\n\n"
                    "🔍 للتحليل: أرسل رمز العملة مباشرة\n"
                    "📊 مثال: BTC, ETH, BNB, XRP\n"
                    "⚡ النتائج فورية ودقيقة!\n\n"
                    "🤖 البوت متاح 24/7",
                    parse_mode='Markdown'
                )
            elif data == "quick_help":
                await query.edit_message_text(
                    "⚡ *تحليل سريع:*\n\n"
                    "1️⃣ أرسل رمز العملة: `BTC`\n"
                    "2️⃣ احصل على تحليل فوري\n"
                    "3️⃣ شاهد الأهداف والتوصيات\n\n"
                    "💡 جرب الآن: أرسل `ETH`",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in button callback: {e}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأخطاء"""
        logger.error(f"Exception: {context.error}")

    def run(self):
        """تشغيل البوت"""
        try:
            # بدء Flask
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            logger.info("✅ Flask server started")
            
            # إنشاء التطبيق
            application = Application.builder().token(self.token).build()
            
            # إضافة المعالجات
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("analyze", self.analyze_command))
            application.add_handler(CallbackQueryHandler(self.button_callback))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            application.add_error_handler(self.error_handler)
            
            logger.info("🤖 Starting Telegram bot...")
            
            # إرسال رسالة للإدمن
            if self.admin_id:
                asyncio.create_task(self.send_startup_message(application))
            
            # بدء التشغيل
            application.run_polling(drop_pending_updates=True)
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    async def send_startup_message(self, application):
        """إرسال رسالة بدء التشغيل"""
        try:
            await asyncio.sleep(3)
            await application.bot.send_message(
                chat_id=self.admin_id,
                text=f"🚀 البوت شغال على Render!\n"
                     f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"✅ جاهز لاستقبال الطلبات"
            )
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")

# التشغيل الرئيسي
if __name__ == "__main__":
    try:
        if not os.getenv('BOT_TOKEN'):
            print("❌ BOT_TOKEN مطلوب في متغيرات البيئة")
            exit(1)
            
        print("🚀 بدء تشغيل البوت...")
        bot = SimpleTelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت")
    except Exception as e:
        print(f"❌ خطأ: {e}")
        exit(1)
