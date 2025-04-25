import pandas as pd
import numpy as np
import yfinance as yf
import ta
import telegram
import asyncio
import schedule
import time
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Stock Configuration
STOCK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']  # Default stocks to analyze
TIME_PERIOD = '1y'  # Data period to fetch

# Technical Analysis Parameters
SHORT_WINDOW = 20
MEDIUM_WINDOW = 50
LONG_WINDOW = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Emojis for signals
EMOJI_MAP = {
    'strong_buy': '🚀',
    'buy': '✅',
    'neutral': '⚖️',
    'sell': '🔴',
    'strong_sell': '⛔',
    'bullish': '📈',
    'bearish': '📉',
    'volume_high': '💰',
    'volume_low': '💸',
    'volatility_high': '⚡',
    'volatility_low': '🌊',
    'uptrend': '↗️',
    'downtrend': '↘️',
    'consolidation': '↔️',
    'breakout': '💥',
    'support': '🛡️',
    'resistance': '🧱',
    'warning': '⚠️',
    'clock': '🕒',
    'chart': '📊',
    'profit': '💹',
    'loss': '📉',
    'calendar': '📅',
    'globe': '🌐'
}

class StockSignalBot:
    def __init__(self):
        self.bot = telegram.Bot(token=TELEGRAM_TOKEN)
    
    def get_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        try:
            data = yf.download(symbol, period=period, interval=interval)
            if data.empty:
                logger.error(f"No data found for {symbol}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate various technical indicators"""
        if data is None or data.empty:
            return None
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Moving Averages
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=SHORT_WINDOW)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=MEDIUM_WINDOW)
        df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=LONG_WINDOW)
        df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=SHORT_WINDOW)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=RSI_PERIOD)
        
        # MACD
        macd = ta.trend.MACD(df['Close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # ADX (Trend Strength)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR (Volatility)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        return df
    
    def generate_signals(self, df, symbol):
        """Generate trading signals based on technical indicators"""
        if df is None or df.empty:
            return None
        
        # Get the most recent values
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None
        
        signals = {
            'symbol': symbol,
            'price': round(current['Close'], 2),
            'change_pct': round(((current['Close'] / df.iloc[-2]['Close']) - 1) * 100, 2) if previous is not None else 0,
            'indicators': {},
            'overall_signal': 'neutral',
            'signal_strength': 0
        }
        
        # Score system: -2 (strong sell) to +2 (strong buy)
        score = 0
        signal_count = 0
        
        # Moving Average Signals
        if current['Close'] > current['SMA20']:
            signals['indicators']['sma20'] = {'signal': 'bullish', 'value': round(current['SMA20'], 2)}
            score += 0.5
        else:
            signals['indicators']['sma20'] = {'signal': 'bearish', 'value': round(current['SMA20'], 2)}
            score -= 0.5
        signal_count += 1
        
        if current['Close'] > current['SMA50']:
            signals['indicators']['sma50'] = {'signal': 'bullish', 'value': round(current['SMA50'], 2)}
            score += 1
        else:
            signals['indicators']['sma50'] = {'signal': 'bearish', 'value': round(current['SMA50'], 2)}
            score -= 1
        signal_count += 1
        
        if current['Close'] > current['SMA200']:
            signals['indicators']['sma200'] = {'signal': 'bullish', 'value': round(current['SMA200'], 2)}
            score += 1.5
        else:
            signals['indicators']['sma200'] = {'signal': 'bearish', 'value': round(current['SMA200'], 2)}
            score -= 1.5
        signal_count += 1
        
        # Golden/Death Cross
        if previous is not None:
            if previous['SMA50'] <= previous['SMA200'] and current['SMA50'] > current['SMA200']:
                signals['indicators']['golden_cross'] = {'signal': 'bullish', 'value': True}
                score += 2
                signal_count += 1
            elif previous['SMA50'] >= previous['SMA200'] and current['SMA50'] < current['SMA200']:
                signals['indicators']['death_cross'] = {'signal': 'bearish', 'value': True}
                score -= 2
                signal_count += 1
        
        # RSI Signal
        signals['indicators']['rsi'] = {'signal': 'neutral', 'value': round(current['RSI'], 2)}
        if current['RSI'] < RSI_OVERSOLD:
            signals['indicators']['rsi']['signal'] = 'oversold'
            score += 1.5
        elif current['RSI'] > RSI_OVERBOUGHT:
            signals['indicators']['rsi']['signal'] = 'overbought'
            score -= 1.5
        signal_count += 1
        
        # MACD Signal
        if current['MACD'] > current['MACD_Signal']:
            signals['indicators']['macd'] = {'signal': 'bullish', 'value': round(current['MACD'], 2)}
            score += 1
        else:
            signals['indicators']['macd'] = {'signal': 'bearish', 'value': round(current['MACD'], 2)}
            score -= 1
        signal_count += 1
        
        # MACD Crossover
        if previous is not None:
            if previous['MACD'] <= previous['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
                signals['indicators']['macd_crossover'] = {'signal': 'bullish', 'value': True}
                score += 1.5
                signal_count += 1
            elif previous['MACD'] >= previous['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
                signals['indicators']['macd_crossover'] = {'signal': 'bearish', 'value': True}
                score -= 1.5
                signal_count += 1
        
        # Bollinger Bands
        bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower']) if (current['BB_Upper'] - current['BB_Lower']) > 0 else 0.5
        signals['indicators']['bollinger'] = {
            'signal': 'neutral',
            'value': round(bb_position, 2),
            'width': round(current['BB_Width'], 2)
        }
        
        if bb_position > 0.95:
            signals['indicators']['bollinger']['signal'] = 'overbought'
            score -= 1
        elif bb_position < 0.05:
            signals['indicators']['bollinger']['signal'] = 'oversold'
            score += 1
        signal_count += 1
        
        # ADX (Trend Strength)
        signals['indicators']['adx'] = {'signal': 'neutral', 'value': round(current['ADX'], 2)}
        if current['ADX'] > 25:
            signals['indicators']['adx']['signal'] = 'strong_trend'
        elif current['ADX'] < 20:
            signals['indicators']['adx']['signal'] = 'weak_trend'
        
        # Stochastic
        signals['indicators']['stochastic'] = {
            'signal': 'neutral', 
            'k_value': round(current['Stoch_K'], 2),
            'd_value': round(current['Stoch_D'], 2)
        }
        
        if current['Stoch_K'] < 20 and current['Stoch_D'] < 20:
            signals['indicators']['stochastic']['signal'] = 'oversold'
            score += 1
        elif current['Stoch_K'] > 80 and current['Stoch_D'] > 80:
            signals['indicators']['stochastic']['signal'] = 'overbought'
            score -= 1
        signal_count += 1
        
        # Volume Analysis
        recent_volume_avg = df['Volume'].tail(5).mean()
        volume_ratio = current['Volume'] / recent_volume_avg if recent_volume_avg > 0 else 1
        
        signals['indicators']['volume'] = {
            'signal': 'neutral',
            'value': int(current['Volume']),
            'ratio': round(volume_ratio, 2)
        }
        
        if volume_ratio > 1.5:
            signals['indicators']['volume']['signal'] = 'high'
            # Volume direction influence
            if current['Close'] > df.iloc[-2]['Close']:
                score += 0.5
            else:
                score -= 0.5
        signal_count += 1
        
        # Calculate the final score normalized to -2 to +2 range
        if signal_count > 0:
            normalized_score = score / (signal_count * 0.75)  # Scale factor to keep within -2 to +2
            normalized_score = max(min(normalized_score, 2), -2)  # Clamp to range
        else:
            normalized_score = 0
        
        # Convert score to signal
        if normalized_score > 1.5:
            signals['overall_signal'] = 'strong_buy'
        elif normalized_score > 0.5:
            signals['overall_signal'] = 'buy'
        elif normalized_score > -0.5:
            signals['overall_signal'] = 'neutral'
        elif normalized_score > -1.5:
            signals['overall_signal'] = 'sell'
        else:
            signals['overall_signal'] = 'strong_sell'
        
        signals['signal_strength'] = round(normalized_score, 2)
        
        return signals
    
    def format_message(self, signals):
        """Format signals into a nice Telegram message with emojis"""
        if signals is None:
            return "⚠️ No signals available"
        
        # Signal emoji
        signal_emoji = EMOJI_MAP.get(signals['overall_signal'], EMOJI_MAP['neutral'])
        
        # Price change direction
        change_direction = "↗️" if signals['change_pct'] > 0 else "↘️" if signals['change_pct'] < 0 else "➡️"
        
        message = f"{EMOJI_MAP['globe']} *STOCK SIGNAL: {signals['symbol']}* {signal_emoji}\n\n"
        message += f"{EMOJI_MAP['chart']} *Price:* ${signals['price']} {change_direction} ({signals['change_pct']}%)\n"
        message += f"{EMOJI_MAP['calendar']} *Date:* {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Overall recommendation
        signal_text = signals['overall_signal'].replace('_', ' ').upper()
        message += f"*RECOMMENDATION:* {signal_emoji} {signal_text} {signal_emoji}\n"
        message += f"*Signal Strength:* {'▓' * int((abs(signals['signal_strength']) * 5) // 2)}{'░' * (10 - int((abs(signals['signal_strength']) * 5) // 2))} ({signals['signal_strength']})\n\n"
        
        # Key indicators section
        message += "*KEY INDICATORS:*\n"
        
        # Moving Averages
        message += f"• MA Trend: "
        if 'sma20' in signals['indicators']:
            sma20_signal = signals['indicators']['sma20']['signal']
            message += f"{EMOJI_MAP['bullish'] if sma20_signal == 'bullish' else EMOJI_MAP['bearish']} 20-day "
        if 'sma50' in signals['indicators']:
            sma50_signal = signals['indicators']['sma50']['signal']
            message += f"{EMOJI_MAP['bullish'] if sma50_signal == 'bullish' else EMOJI_MAP['bearish']} 50-day "
        if 'sma200' in signals['indicators']:
            sma200_signal = signals['indicators']['sma200']['signal']
            message += f"{EMOJI_MAP['bullish'] if sma200_signal == 'bullish' else EMOJI_MAP['bearish']} 200-day"
        message += "\n"
        
        # Special Crossovers
        if 'golden_cross' in signals['indicators']:
            message += f"• {EMOJI_MAP['breakout']} GOLDEN CROSS DETECTED! {EMOJI_MAP['bullish']}\n"
        if 'death_cross' in signals['indicators']:
            message += f"• {EMOJI_MAP['warning']} DEATH CROSS DETECTED! {EMOJI_MAP['bearish']}\n"
        
        # RSI
        if 'rsi' in signals['indicators']:
            rsi_value = signals['indicators']['rsi']['value']
            rsi_signal = signals['indicators']['rsi']['signal']
            rsi_emoji = EMOJI_MAP['neutral']
            if rsi_signal == 'oversold':
                rsi_emoji = EMOJI_MAP['buy']
            elif rsi_signal == 'overbought':
                rsi_emoji = EMOJI_MAP['sell']
            message += f"• RSI: {rsi_emoji} {rsi_value} "
            if rsi_signal != 'neutral':
                message += f"({rsi_signal.upper()})"
            message += "\n"
        
        # MACD
        if 'macd' in signals['indicators']:
            macd_signal = signals['indicators']['macd']['signal']
            macd_emoji = EMOJI_MAP['bullish'] if macd_signal == 'bullish' else EMOJI_MAP['bearish']
            message += f"• MACD: {macd_emoji} {macd_signal.upper()}\n"
        
        # MACD Crossover
        if 'macd_crossover' in signals['indicators']:
            cross_signal = signals['indicators']['macd_crossover']['signal']
            cross_emoji = EMOJI_MAP['bullish'] if cross_signal == 'bullish' else EMOJI_MAP['bearish']
            message += f"• {cross_emoji} MACD CROSSOVER ({cross_signal.upper()})\n"
        
        # Bollinger Bands
        if 'bollinger' in signals['indicators']:
            bb_signal = signals['indicators']['bollinger']['signal']
            bb_value = signals['indicators']['bollinger']['value']
            bb_width = signals['indicators']['bollinger']['width']
            
            if bb_signal == 'overbought':
                bb_emoji = EMOJI_MAP['sell']
                bb_text = "OVERBOUGHT"
            elif bb_signal == 'oversold':
                bb_emoji = EMOJI_MAP['buy']
                bb_text = "OVERSOLD"
            else:
                bb_emoji = EMOJI_MAP['neutral']
                bb_text = "NEUTRAL"
                
            message += f"• Bollinger: {bb_emoji} {bb_text} (pos: {bb_value}, width: {bb_width})\n"
        
        # Stochastic
        if 'stochastic' in signals['indicators']:
            stoch_signal = signals['indicators']['stochastic']['signal']
            stoch_k = signals['indicators']['stochastic']['k_value']
            stoch_d = signals['indicators']['stochastic']['d_value']
            
            if stoch_signal == 'overbought':
                stoch_emoji = EMOJI_MAP['sell']
                stoch_text = "OVERBOUGHT"
            elif stoch_signal == 'oversold':
                stoch_emoji = EMOJI_MAP['buy']
                stoch_text = "OVERSOLD"
            else:
                stoch_emoji = EMOJI_MAP['neutral']
                stoch_text = "NEUTRAL"
                
            message += f"• Stochastic: {stoch_emoji} {stoch_text} (K: {stoch_k}, D: {stoch_d})\n"
        
        # Volume
        if 'volume' in signals['indicators']:
            vol_signal = signals['indicators']['volume']['signal']
            vol_ratio = signals['indicators']['volume']['ratio']
            vol_emoji = EMOJI_MAP['volume_high'] if vol_signal == 'high' else EMOJI_MAP['volume_low']
            
            message += f"• Volume: {vol_emoji} {vol_ratio}x average\n"
        
        # ADX (Trend Strength)
        if 'adx' in signals['indicators']:
            adx_value = signals['indicators']['adx']['value']
            adx_signal = signals['indicators']['adx']['signal']
            
            if adx_signal == 'strong_trend':
                adx_emoji = EMOJI_MAP['bullish'] if signals['signal_strength'] > 0 else EMOJI_MAP['bearish']
                adx_text = "STRONG TREND"
            elif adx_signal == 'weak_trend':
                adx_emoji = EMOJI_MAP['consolidation']
                adx_text = "WEAK TREND"
            else:
                adx_emoji = EMOJI_MAP['neutral']
                adx_text = "MODERATE TREND"
                
            message += f"• ADX: {adx_emoji} {adx_value} ({adx_text})\n"
        
        # Disclaimer
        message += f"\n{EMOJI_MAP['warning']} *Disclaimer:* This is algorithmic analysis and should not be considered financial advice. Always do your own research before investing."
        
        return message
    
    async def send_telegram_message(self, message):
        """Send formatted message to Telegram"""
        try:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Message sent successfully to Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def process_stock(self, symbol):
        """Process a single stock and send signal to Telegram"""
        logger.info(f"Processing stock: {symbol}")
        data = self.get_stock_data(symbol)
        if data is None:
            await self.send_telegram_message(f"⚠️ Error: Could not fetch data for {symbol}")
            return
            
        data_with_indicators = self.calculate_technical_indicators(data)
        signals = self.generate_signals(data_with_indicators, symbol)
        message = self.format_message(signals)
        await self.send_telegram_message(message)
    
    async def run_daily_analysis(self):
        """Run analysis on all configured stocks"""
        logger.info(f"Starting daily analysis for {len(STOCK_SYMBOLS)} stocks")
        
        # Header message
        current_date = datetime.now().strftime('%Y-%m-%d')
        header = f"{EMOJI_MAP['chart']} *DAILY STOCK SIGNALS - {current_date}* {EMOJI_MAP['calendar']}\n\n"
        header += f"Analyzing {len(STOCK_SYMBOLS)} stocks for trading opportunities...\n"
        await self.send_telegram_message(header)
        
        # Process each stock
        for symbol in STOCK_SYMBOLS:
            await self.process_stock(symbol)
            await asyncio.sleep(1)  # Small delay between stocks
        
        # Summary message
        summary = f"{EMOJI_MAP['clock']} *ANALYSIS COMPLETE* {EMOJI_MAP['chart']}\n\n"
        summary += f"Analyzed {len(STOCK_SYMBOLS)} stocks on {current_date}.\n"
        summary += "Remember to review signals in context of broader market conditions and your investment strategy."
        await self.send_telegram_message(summary)

def job():
    """Run the daily job"""
    logger.info("Starting scheduled job")
    bot = StockSignalBot()
    asyncio.run(bot.run_daily_analysis())
    logger.info("Job completed")

def add_stock(symbol):
    """Add a stock to the tracking list"""
    if symbol not in STOCK_SYMBOLS:
        STOCK_SYMBOLS.append(symbol)
        logger.info(f"Added {symbol} to tracking list")
        return True
    return False

def remove_stock(symbol):
    """Remove a stock from the tracking list"""
    if symbol in STOCK_SYMBOLS:
        STOCK_SYMBOLS.remove(symbol)
        logger.info(f"Removed {symbol} from tracking list")
        return True
    return False

def main():
    """Main function to run the bot"""
    # Schedule the job to run daily at market close (4:30 PM Eastern Time)
    schedule.every().monday.at("16:30").do(job)
    schedule.every().tuesday.at("16:30").do(job)
    schedule.every().wednesday.at("16:30").do(job)
    schedule.every().thursday.at("16:30").do(job)
    schedule.every().friday.at("16:30").do(job)
    
    logger.info("Stock Signal Bot initialized")
    logger.info(f"Tracking {len(STOCK_SYMBOLS)} stocks")
    logger.info("Scheduled to run at 16:30 ET on weekdays")
    
    # Run the job once to test
    print("Running initial analysis...")
    job()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
