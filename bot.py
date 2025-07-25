import os
import asyncio
import logging
import aiohttp
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, Response
import threading

# إعداد الـ logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# متغيرات البيئة
BOT_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_ID = int(os.getenv('ADMIN_ID', '0'))
PORT = int(os.getenv('PORT', 10000))
WEBHOOK_URL = os.getenv('WEBHOOK_URL', f'https://crypto-telegram-bot.onrender.com')

# إنشاء Flask app
app = Flask(__name__)

# متغير عام للتطبيق
telegram_app = None

class TechnicalAnalyzer:
    """محلل فني للعملات المشفرة"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    async def get_price_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السعر من Binance API"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"API Error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return None

    async def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 50) -> Optional[List]:
        """جلب بيانات الشموع"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching kline data: {e}")
            return None

    def calculate_indicators(self, kline_data: List) -> Dict:
        """حساب المؤشرات الفنية"""
        try:
            if not kline_data or len(kline_data) < 20:
                return {}
            
            closes = [float(kline[4]) for kline in kline_data]
            highs = [float(kline[2]) for kline in kline_data]
            lows = [float(kline[3]) for kline in kline_data]
            volumes = [float(kline[5]) for kline in kline_data]
            
            current_price = closes[-1]
            
            # المتوسطات المتحركة
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
            sma_10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else current_price
            sma_5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current_price
            
            # RSI المبسط
            gains = []
            losses = []
            for i in range(1, min(len(closes), 15)):
                change = closes[-i] - closes[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if gains and losses:
                avg_gain = sum(gains) / len(gains)
                avg_loss = sum(losses) / len(losses)
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # الدعم والمقاومة
            support = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            resistance = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            
            # الحجم
            avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_10': sma_10,
                'sma_5': sma_5,
                'rsi': rsi,
                'support': support,
                'resistance': resistance,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def analyze_trend(self, price_data: Dict, indicators: Dict) -> Tuple[str, str]:
        """تحليل الاتجاه والتوصية"""
        try:
            if not indicators:
                return "غير محدد", "⚪ انتظار"
            
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_10 = indicators.get('sma_10', 0)
            sma_5 = indicators.get('sma_5', 0)
            rsi = indicators.get('rsi', 50)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            signals = []
            
            # إشارات المتوسطات
            if current_price > sma_5 > sma_10 > sma_20:
                signals.append(2)
            elif current_price > sma_10 > sma_20:
                signals.append(1)
            elif current_price < sma_5 < sma_10 < sma_20:
                signals.append(-2)
            elif current_price < sma_10 < sma_20:
                signals.append(-1)
            else:
                signals.append(0)
            
            # إشارات RSI
            if rsi < 30:
                signals.append(1)
            elif rsi > 70:
                signals.append(-1)
            else:
                signals.append(0)
            
            # إشارات الحجم
            if volume_ratio > 1.5:
                signals.append(0.5)
            elif volume_ratio < 0.7:
                signals.append(-0.3)
            else:
                signals.append(0)
            
            avg_signal = sum(signals) / len(signals) if signals else 0
            
            if avg_signal > 1:
                return "صاعد قوي", "🟢 شراء قوي"
            elif avg_signal > 0.3:
                return "صاعد", "🔵 شراء"
            elif avg_signal < -1:
                return "هابط قوي", "🔴 بيع قوي"
            elif avg_signal < -0.3:
                return "هابط", "🟠 بيع"
            else:
                return "محايد", "⚪ انتظار"
                
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return "غير محدد", "⚪ انتظار"

    async def analyze_symbol(self, symbol: str) -> Dict:
        """تحليل شامل للعملة"""
        try:
            # جلب البيانات
            price_data = await self.get_price_data(symbol)
            if not price_data:
                return {}
            
            kline_data = await self.get_kline_data(symbol)
            if not kline_data:
                return {'price_data': price_data}
            
            # حساب المؤشرات
            indicators = self.calculate_indicators(kline_data)
            if not indicators:
                return {'price_data': price_data}
            
            # تحليل الاتجاه
            trend, recommendation = self.analyze_trend(price_data, indicators)
            
            current_price = float(price_data['lastPrice'])
            
            return {
                'price_data': price_data,
                'indicators': indicators,
                'trend': trend,
                'recommendation': recommendation,
                'current_price': current_price,
                'targets': {
                    'buy': [current_price * 1.03, current_price * 1.07, current_price * 1.15],
                    'sell': [current_price * 0.97, current_price * 0.93, current_price * 0.85]
                },
                'stop_loss': {
                    'buy': indicators.get('support', current_price * 0.95),
                    'sell': indicators.get('resistance', current_price * 1.05)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {}

class CryptoBot:
    """بوت التوصيات المشفرة"""
    
    def __init__(self):
        self.token = BOT_TOKEN
        self.admin_id = ADMIN_ID
        self.analyzer = TechnicalAnalyzer()
        self.watchlists = {}
        
        if not self.token:
            raise ValueError("BOT_TOKEN is required")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر البداية"""
        welcome = """
🤖 *مرحباً بك في بوت التوصيات المشفرة!*

📊 *ما أستطيع فعله:*
• تحليل فني دقيق للعملات
• نقاط دخول وخروج محددة  
• توصيات مبنية على مؤشرات فنية
• مراقبة العملات المفضلة

🚀 *جرب الآن:*
أرسل رمز أي عملة مثل: `BTC`

📝 *أوامر أخرى:*
/analyze BTC - تحليل مفصل
/help - المساعدة
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 تحليل BTC", callback_data="analyze_BTC")],
            [InlineKeyboardButton("📊 تحليل ETH", callback_data="analyze_ETH")],
            [InlineKeyboardButton("❓ مساعدة", callback_data="help")]
        ]
        
        await update.message.reply_text(
            welcome, 
            parse_mode='Markdown', 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تحليل عملة"""
        try:
            if not context.args:
                await update.message.reply_text("❌ أرسل رمز العملة\nمثال: /analyze BTC")
                return
                
            symbol = context.args[0].upper()
            
            # رسالة انتظار
            msg = await update.message.reply_text(f"🔍 جاري تحليل {symbol}...")
            
            # التحليل
            analysis = await self.analyzer.analyze_symbol(symbol)
            
            if not analysis:
                await msg.edit_text(f"❌ لم أجد بيانات لـ {symbol}")
                return
                
            # تنسيق التقرير
            report = self.format_report(symbol, analysis)
            
            # الأزرار
            keyboard = [
                [InlineKeyboardButton("🔄 تحديث", callback_data=f"refresh_{symbol}")],
                [InlineKeyboardButton("📊 تحليل آخر", callback_data="quick")]
            ]
            
            await msg.edit_text(
                report, 
                parse_mode='Markdown', 
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            await update.message.reply_text(f"❌ خطأ في تحليل {symbol if 'symbol' in locals() else 'العملة'}")

    def format_report(self, symbol: str, analysis: Dict) -> str:
        """تنسيق تقرير التحليل"""
        try:
            price_data = analysis.get('price_data', {})
            indicators = analysis.get('indicators', {})
            
            current_price = analysis.get('current_price', 0)
            trend = analysis.get('trend', 'غير محدد')
            recommendation = analysis.get('recommendation', '⚪ انتظار')
            
            change_24h = float(price_data.get('priceChangePercent', 0))
            volume = float(price_data.get('volume', 0))
            high_24h = float(price_data.get('highPrice', 0))
            low_24h = float(price_data.get('lowPrice', 0))
            
            # تنسيق السعر
            if current_price < 0.01:
                price_str = f"${current_price:.6f}"
            elif current_price < 1:
                price_str = f"${current_price:.4f}"
            else:
                price_str = f"${current_price:.2f}"
            
            change_emoji = "🟢" if change_24h >= 0 else "🔴"
            
            report = f"""
🎯 *تحليل {symbol}/USDT*
━━━━━━━━━━━━━━━━━━━━

💰 *السعر:* {price_str}
{change_emoji} *24س:* {change_24h:+.2f}%
📊 *الاتجاه:* {trend}
🎲 *التوصية:* {recommendation}

━━━━━━━━━━━━━━━━━━━━
📈 *بيانات السوق:*

🔸 *أعلى:* ${high_24h:.6f}
🔸 *أقل:* ${low_24h:.6f}
🔸 *الحجم:* {volume:,.0f}
🔸 *RSI:* {indicators.get('rsi', 0):.1f}

━━━━━━━━━━━━━━━━━━━━
            """
            
            # إضافة الاستراتيجية
            targets = analysis.get('targets', {})
            stop_loss = analysis.get('stop_loss', {})
            
            if "شراء" in recommendation:
                buy_targets = targets.get('buy', [])
                if buy_targets:
                    report += f"""🎯 *استراتيجية الشراء:*

🏆 *الأهداف:*
🥇 ${buy_targets[0]:.6f} (+3%)
🥈 ${buy_targets[1]:.6f} (+7%)  
🥉 ${buy_targets[2]:.6f} (+15%)

🛑 *وقف الخسارة:* ${stop_loss.get('buy', 0):.6f}
                    """
            elif "بيع" in recommendation:
                sell_targets = targets.get('sell', [])
                if sell_targets:
                    report += f"""🎯 *استراتيجية البيع:*

🎯 *الأهداف:*
🥇 ${sell_targets[0]:.6f} (-3%)
🥈 ${sell_targets[1]:.6f} (-7%)
🥉 ${sell_targets[2]:.6f} (-15%)

🛑 *وقف الخسارة:* ${stop_loss.get('sell', 0):.6f}
                    """
            else:
                report += """
⚪ *وضع الانتظار:*
السوق محايد حالياً
انتظر إشارة أوضح
                """
            
            report += f"""
━━━━━━━━━━━━━━━━━━━━
⚠️ *للأغراض التعليمية فقط*
⏰ *التحديث:* {datetime.now().strftime('%H:%M')}
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"❌ خطأ في تنسيق تقرير {symbol}"

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج النصوص"""
        text = update.message.text.upper().strip()
        
        if len(text) <= 10 and text.isalpha():
            context.args = [text]
            await self.analyze(update, context)
        else:
            await update.message.reply_text("💡 أرسل رمز العملة (مثل: BTC)")

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """المساعدة"""
        help_text = """
📚 *دليل الاستخدام:*

🔍 *التحليل:*
• أرسل رمز العملة: `BTC`
• أو: `/analyze BTC`

📊 *التوصيات:*
🟢 شراء قوي | 🔵 شراء
🔴 بيع قوي | 🟠 بيع | ⚪ انتظار

💡 *أمثلة:*
BTC, ETH, BNB, XRP, ADA, SOL

⚠️ *تحذير:*
للأغراض التعليمية فقط
قم ببحثك الخاص قبل الاستثمار

🚀 *البوت متاح 24/7!*
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالج الأزرار"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "help":
            await self.help_cmd(update, context)
        elif data == "quick":
            await query.edit_message_text(
                "📊 أرسل رمز العملة للتحليل:\nمثال: BTC, ETH, BNB, XRP"
            )
        elif data.startswith("analyze_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await self.analyze(update, context)
        elif data.startswith("refresh_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await self.analyze(update, context)

# إنشاء البوت
bot = CryptoBot()

# Flask Routes
@app.route('/')
def home():
    return """
    <h1>🤖 Crypto Telegram Bot</h1>
    <p>✅ Bot is running!</p>
    <p>⏰ Time: {}</p>
    <p>🔗 <a href="/health">Health Check</a></p>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 200

@app.route('/health')
def health():
    return {
        "status": "healthy",
        "bot": "running",
        "timestamp": datetime.now().isoformat()
    }, 200

@app.route('/webhook', methods=['POST'])
def webhook():
    """استقبال التحديثات من تليغرام"""
    try:
        if telegram_app is None:
            return "Bot not ready", 503
            
        json_data = request.get_json()
        if json_data:
            update = Update.de_json(json_data, telegram_app.bot)
            asyncio.run(telegram_app.process_update(update))
            
        return "OK", 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return "Error", 500

async def setup_bot():
    """إعداد البوت"""
    global telegram_app
    
    try:
        # إنشاء التطبيق
        telegram_app = Application.builder().token(BOT_TOKEN).build()
        
        # إضافة المعالجات
        telegram_app.add_handler(CommandHandler("start", bot.start))
        telegram_app.add_handler(CommandHandler("analyze", bot.analyze))
        telegram_app.add_handler(CommandHandler("help", bot.help_cmd))
        telegram_app.add_handler(CallbackQueryHandler(bot.button_handler))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
        
        # إعداد webhook
        webhook_url = f"{WEBHOOK_URL}/webhook"
        await telegram_app.bot.set_webhook(webhook_url)
        
        logger.info(f"✅ Webhook set to: {webhook_url}")
        logger.info("🤖 Bot is ready!")
        
        # إرسال رسالة للإدمن
        if ADMIN_ID:
            try:
                await telegram_app.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=f"🚀 البوت يعمل على Render!\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                )
            except Exception as e:
                logger.error(f"Could not send admin message: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up bot: {e}")
        return False

def run_app():
    """تشغيل التطبيق"""
    try:
        # إعداد البوت
        setup_success = asyncio.run(setup_bot())
        
        if not setup_success:
            logger.error("Failed to setup bot")
            return
        
        logger.info(f"🌐 Starting Flask on port {PORT}")
        
        # تشغيل Flask
        app.run(
            host='0.0.0.0',
            port=PORT,
            debug=False,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Error running app: {e}")

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN required")
        exit(1)
    
    print("🚀 Starting Crypto Bot with Webhook...")
    run_app()
