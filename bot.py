import websocket
import json
import threading
import time
import pandas as pd
import pandas_ta as ta
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import numpy as np

from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class LeveragedTradingBot:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=TESTNET)
        self.setup_symbol()
        
        # Data storage
        self.data = {tf: pd.DataFrame() for tf in TIMEFRAMES}
        self.current_price = 0
        self.position = None
        self.trade_history = []
        self.performance_metrics = {}
        
        # WebSocket
        self.ws = None
        self.setup_websocket()
        
        # Initialize data
        self.load_historical_data()
        
        logging.info("🤖 Leveraged Trading Bot Initialized")

    def setup_symbol(self):
        """Setup leverage and symbol info"""
        try:
            # Set leverage
            self.client.futures_change_leverage(
                symbol=SYMBOL, 
                leverage=LEVERAGE
            )
            logging.info(f"✅ Leverage set to {LEVERAGE}x for {SYMBOL}")
        except Exception as e:
            logging.error(f"❌ Error setting leverage: {e}")

    def load_historical_data(self):
        """Load historical data for all timeframes"""
        for timeframe in TIMEFRAMES:
            try:
                klines = self.client.futures_klines(
                    symbol=SYMBOL,
                    interval=timeframe,
                    limit=INITIAL_CANDLES
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                self.data[timeframe] = df
                logging.info(f"✅ Loaded {len(df)} {timeframe} candles")
                
            except Exception as e:
                logging.error(f"❌ Error loading {timeframe} data: {e}")

    def calculate_indicators(self, df):
        """Calculate all technical indicators with weights"""
        if len(df) < 200:
            return 0
        
        try:
            weights = 0
            
            # EMA Signals
            ema_fast = ta.ema(df['close'], length=EMA_FAST)
            ema_slow = ta.ema(df['close'], length=EMA_SLOW)
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                weights += 0.1
            else:
                weights -= 0.1
            
            # MACD
            macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
                weights += 0.1
            else:
                weights -= 0.1
            
            # RSI
            rsi = ta.rsi(df['close'], length=RSI_PERIOD)
            if rsi.iloc[-1] < 30:
                weights += 0.1
            elif rsi.iloc[-1] > 70:
                weights -= 0.1
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=BB_PERIOD)
            current_close = df['close'].iloc[-1]
            if current_close < bb['BBL_20_2.0'].iloc[-1]:
                weights += 0.1
            elif current_close > bb['BBU_20_2.0'].iloc[-1]:
                weights -= 0.1
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)
            if adx['ADX_14'].iloc[-1] > 25:
                # Strong trend
                if df['close'].iloc[-1] > df['close'].iloc[-2]:
                    weights += 0.1
                else:
                    weights -= 0.1
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=STOCH_K, d=STOCH_D)
            if stoch['STOCHk_14_3_3'].iloc[-1] < 20:
                weights += 0.1
            elif stoch['STOCHk_14_3_3'].iloc[-1] > 80:
                weights -= 0.1
            
            # Volume (compare with average)
            volume_avg = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            if current_volume > volume_avg * 1.2:
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    weights += 0.1
                else:
                    weights -= 0.1
            
            # ATR (Volatility)
            atr = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
            atr_percent = atr.iloc[-1] / df['close'].iloc[-1]
            if atr_percent > 0.02:  # High volatility
                # In high volatility, be cautious about new positions
                weights *= 0.8
            
            # Fibonacci (simplified - using recent high/low)
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            current_price = df['close'].iloc[-1]
            
            fib_levels = {
                '0.236': recent_high - (recent_high - recent_low) * 0.236,
                '0.382': recent_high - (recent_high - recent_low) * 0.382,
                '0.5': recent_high - (recent_high - recent_low) * 0.5,
                '0.618': recent_high - (recent_high - recent_low) * 0.618
            }
            
            for level, price in fib_levels.items():
                if abs(current_price - price) / current_price < 0.005:  # Within 0.5%
                    if current_price > price:
                        weights += 0.05
                    else:
                        weights -= 0.05
            
            return round(weights, 3)
            
        except Exception as e:
            logging.error(f"❌ Error calculating indicators: {e}")
            return 0

    def get_combined_weight(self):
        """Get combined weight from all timeframes"""
        total_weight = 0
        timeframe_weights = {}
        
        for timeframe in TIMEFRAMES:
            if len(self.data[timeframe]) >= 200:
                weight = self.calculate_indicators(self.data[timeframe])
                total_weight += weight
                timeframe_weights[timeframe] = weight
                logging.info(f"📊 {timeframe} Weight: {weight}")
        
        avg_weight = total_weight / len(TIMEFRAMES)
        logging.info(f"📈 Combined Weight: {avg_weight}")
        return avg_weight, timeframe_weights

    def calculate_position_size(self):
        """Calculate position size with leverage"""
        try:
            balance_info = self.client.futures_account_balance()
            usdt_balance = next((float(bal['balance']) for bal in balance_info if bal['asset'] == 'USDT'), 0)
            
            position_value = usdt_balance * POSITION_SIZE_PERCENT * LEVERAGE
            quantity = position_value / self.current_price
            
            return min(quantity, QUANTITY)  # Use fixed quantity or calculated
        except Exception as e:
            logging.error(f"❌ Error calculating position size: {e}")
            return QUANTITY

    def open_long_position(self, weight):
        """Open long position"""
        try:
            quantity = self.calculate_position_size()
            
            # Calculate stop loss and take profit
            stop_loss = self.current_price * (1 - STOP_LOSS_PERCENT)
            take_profit = self.current_price * (1 + TAKE_PROFIT_PERCENT)
            
            # Place order
            order = self.client.futures_create_order(
                symbol=SYMBOL,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            
            self.position = {
                'type': 'LONG',
                'entry_price': self.current_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'entry_weight': weight
            }
            
            trade_log = {
                'action': 'OPEN_LONG',
                'timestamp': datetime.now(),
                'price': self.current_price,
                'quantity': quantity,
                'weight': weight,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            self.trade_history.append(trade_log)
            
            logging.info(f"✅ LONG opened: {quantity} {SYMBOL} at ${self.current_price}")
            logging.info(f"🎯 SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Error opening long position: {e}")

    def open_short_position(self, weight):
        """Open short position"""
        try:
            quantity = self.calculate_position_size()
            
            # Calculate stop loss and take profit
            stop_loss = self.current_price * (1 + STOP_LOSS_PERCENT)
            take_profit = self.current_price * (1 - TAKE_PROFIT_PERCENT)
            
            # Place order
            order = self.client.futures_create_order(
                symbol=SYMBOL,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            self.position = {
                'type': 'SHORT',
                'entry_price': self.current_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'entry_weight': weight
            }
            
            trade_log = {
                'action': 'OPEN_SHORT',
                'timestamp': datetime.now(),
                'price': self.current_price,
                'quantity': quantity,
                'weight': weight,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            self.trade_history.append(trade_log)
            
            logging.info(f"✅ SHORT opened: {quantity} {SYMBOL} at ${self.current_price}")
            logging.info(f"🎯 SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Error opening short position: {e}")

    def close_position(self, reason="Strategy"):
        """Close current position"""
        try:
            if not self.position:
                return
            
            side = 'SELL' if self.position['type'] == 'LONG' else 'BUY'
            
            order = self.client.futures_create_order(
                symbol=SYMBOL,
                side=side,
                type='MARKET',
                quantity=self.position['quantity']
            )
            
            # Calculate PnL
            if self.position['type'] == 'LONG':
                pnl = (self.current_price - self.position['entry_price']) * self.position['quantity']
            else:
                pnl = (self.position['entry_price'] - self.current_price) * self.position['quantity']
            
            pnl_percent = (pnl / (self.position['entry_price'] * self.position['quantity'])) * 100 * LEVERAGE
            
            trade_log = {
                'action': f'CLOSE_{self.position["type"]}',
                'timestamp': datetime.now(),
                'price': self.current_price,
                'quantity': self.position['quantity'],
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'reason': reason,
                'entry_price': self.position['entry_price'],
                'entry_weight': self.position['entry_weight']
            }
            self.trade_history.append(trade_log)
            
            logging.info(f"🔒 Position closed: {self.position['type']} at ${self.current_price}")
            logging.info(f"💰 PnL: ${pnl:.2f} ({pnl_percent:.2f}%) - Reason: {reason}")
            
            self.position = None
            
        except Exception as e:
            logging.error(f"❌ Error closing position: {e}")

    def check_position_management(self, current_weight):
        """Check if current position should be managed"""
        if not self.position:
            return
        
        current_pnl = self.calculate_unrealized_pnl()
        
        # Check stop loss and take profit
        if (self.position['type'] == 'LONG' and 
            (self.current_price <= self.position['stop_loss'] or 
             self.current_price >= self.position['take_profit'])):
            reason = "TP" if self.current_price >= self.position['take_profit'] else "SL"
            self.close_position(f"{reason} Hit")
            
        elif (self.position['type'] == 'SHORT' and 
              (self.current_price >= self.position['stop_loss'] or 
               self.current_price <= self.position['take_profit'])):
            reason = "TP" if self.current_price <= self.position['take_profit'] else "SL"
            self.close_position(f"{reason} Hit")
        
        # Check weight-based exit
        elif (self.position['type'] == 'LONG' and current_weight < -EXIT_THRESHOLD):
            self.close_position("Weight Signal")
        elif (self.position['type'] == 'SHORT' and current_weight > EXIT_THRESHOLD):
            self.close_position("Weight Signal")

    def calculate_unrealized_pnl(self):
        """Calculate unrealized PnL for current position"""
        if not self.position:
            return 0
        
        if self.position['type'] == 'LONG':
            return (self.current_price - self.position['entry_price']) * self.position['quantity']
        else:
            return (self.position['entry_price'] - self.current_price) * self.position['quantity']

    def on_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'e' in data and data['e'] == 'kline':
                kline = data['k']
                timeframe = kline['i']
                
                if timeframe in TIMEFRAMES:
                    self.update_data(timeframe, kline)
                    
                    # Only trade on 5m updates for responsiveness
                    if timeframe == '5m':
                        self.trading_logic()
                        
        except Exception as e:
            logging.error(f"❌ WebSocket message error: {e}")

    def update_data(self, timeframe, kline):
        """Update data for specific timeframe"""
        new_data = {
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }
        
        # Update or append new candle
        df = self.data[timeframe]
        if len(df) > 0 and df['timestamp'].iloc[-1] == new_data['timestamp']:
            # Update current candle
            df.iloc[-1] = new_data
        else:
            # Add new candle
            new_df = pd.DataFrame([new_data])
            self.data[timeframe] = pd.concat([df, new_df], ignore_index=True).tail(500)
        
        self.current_price = float(kline['c'])

    def trading_logic(self):
        """Main trading logic"""
        try:
            current_weight, timeframe_weights = self.get_combined_weight()
            
            # Check position management first
            self.check_position_management(current_weight)
            
            # Then check for new entries
            if not self.position:
                if current_weight > LONG_THRESHOLD:
                    self.open_long_position(current_weight)
                elif current_weight < SHORT_THRESHOLD:
                    self.open_short_position(current_weight)
                    
        except Exception as e:
            logging.error(f"❌ Trading logic error: {e}")

    def setup_websocket(self):
        """Setup WebSocket connection"""
        def run_websocket():
            stream_url = "wss://fstream.binance.com/ws" if not TESTNET else "wss://stream.binancefuture.com/ws"
            streams = [f"{SYMBOL.lower()}@kline_{tf}" for tf in TIMEFRAMES]
            full_url = f"{stream_url}/{'/'.join(streams)}"
            
            self.ws = websocket.WebSocketApp(
                full_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.ws.run_forever()
        
        # Start WebSocket in separate thread
        ws_thread = threading.Thread(target=run_websocket)
        ws_thread.daemon = True
        ws_thread.start()

    def on_error(self, ws, error):
        logging.error(f"❌ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info("🔌 WebSocket connection closed")

    def get_performance_summary(self):
        """Calculate performance metrics"""
        if not self.trade_history:
            return {}
        
        closed_trades = [t for t in self.trade_history if 'pnl' in t]
        
        if not closed_trades:
            return {}
        
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in closed_trades if t['pnl'] <= 0])
        
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(t['pnl'] for t in closed_trades)
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in closed_trades if t['pnl'] <= 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

# Backtesting Class
class Backtester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=TESTNET)
        self.bot = LeveragedTradingBot()
        self.backtest_results = []

    def run_backtest(self, start_date, end_date, timeframe='5m'):
        """Run backtest for specified date range"""
        try:
            # Convert dates to milliseconds
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            # Fetch historical data
            klines = self.client.futures_klines(
                symbol=SYMBOL,
                interval=timeframe,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )
            
            if not klines:
                logging.error("❌ No historical data found")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Run backtest
            for i in range(200, len(df)):
                # Update bot data
                current_data = df.iloc[:i+1].copy()
                self.bot.data[timeframe] = current_data
                self.bot.current_price = df['close'].iloc[i]
                
                # Execute trading logic
                self.bot.trading_logic()
                
                # Record snapshot
                if i % 50 == 0:  # Record every 50 candles
                    self.record_snapshot(df.iloc[i])
            
            logging.info("✅ Backtest completed")
            self.generate_backtest_report()
            
        except Exception as e:
            logging.error(f"❌ Backtest error: {e}")

    def record_snapshot(self, current_candle):
        """Record backtest snapshot"""
        snapshot = {
            'timestamp': current_candle['timestamp'],
            'price': current_candle['close'],
            'balance': self.balance,
            'position': self.bot.position,
            'total_trades': len([t for t in self.bot.trade_history if 'pnl' in t])
        }
        self.backtest_results.append(snapshot)

    def generate_backtest_report(self):
        """Generate backtest performance report"""
        performance = self.bot.get_performance_summary()
        
        print("\n" + "="*50)
        print("BACKTEST REPORT")
        print("="*50)
        for key, value in performance.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*50)

# Dashboard
def create_dashboard(bot):
    """Create live trading dashboard"""
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("🚀 Leveraged Trading Bot Dashboard", className="text-center mb-4"), width=12)
        ]),
        
        # Runtime and Balance Info
        dbc.Row([
            dbc.Col([
                html.Div(id="runtime-info", className="card text-white bg-primary mb-3"),
                html.Div(id="balance-info", className="card text-white bg-success mb-3"),
            ], width=6),
            dbc.Col([
                html.Div(id="performance-metrics", className="card text-white bg-info mb-3"),
            ], width=6),
        ]),
        
        # Current Position
        dbc.Row([
            dbc.Col([
                html.Div(id="position-info", className="card text-white bg-warning mb-3"),
            ], width=12),
        ]),
        
        # Trade History
        dbc.Row([
            dbc.Col([
                html.H3("Trade History"),
                html.Div(id="trade-history", className="card text-white bg-secondary"),
            ], width=12),
        ]),
        
        # Update interval
        dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
    ], fluid=True)
    
    @app.callback(
        [Output('runtime-info', 'children'),
         Output('balance-info', 'children'),
         Output('performance-metrics', 'children'),
         Output('position-info', 'children'),
         Output('trade-history', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        # Runtime info
        runtime_info = [
            html.H4("Runtime Info", className="card-header"),
            html.Div([
                html.P(f"Current Price: ${bot.current_price:.6f}"),
                html.P(f"Symbol: {SYMBOL}"),
                html.P(f"Leverage: {LEVERAGE}x"),
                html.P(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ], className="card-body")
        ]
        
        # Balance info
        try:
            balance_info = bot.client.futures_account_balance()
            usdt_balance = next((float(bal['balance']) for bal in balance_info if bal['asset'] == 'USDT'), 0)
            balance_display = f"${usdt_balance:.2f}"
        except:
            balance_display = "Loading..."
        
        balance_info = [
            html.H4("Account Balance", className="card-header"),
            html.Div([
                html.H2(balance_display, className="card-title"),
                html.P(f"Position Size: {POSITION_SIZE_PERCENT*100}%"),
            ], className="card-body")
        ]
        
        # Performance metrics
        performance = bot.get_performance_summary()
        if performance:
            metrics = [
                html.H4("Performance Metrics", className="card-header"),
                html.Div([
                    html.P(f"Total Trades: {performance['total_trades']}"),
                    html.P(f"Win Rate: {performance['win_rate']:.1f}%"),
                    html.P(f"Total PnL: ${performance['total_pnl']:.2f}"),
                    html.P(f"Profit Factor: {performance['profit_factor']:.2f}"),
                ], className="card-body")
            ]
        else:
            metrics = [
                html.H4("Performance Metrics", className="card-header"),
                html.Div([html.P("No trades completed yet")], className="card-body")
            ]
        
        # Position info
        if bot.position:
            unrealized_pnl = bot.calculate_unrealized_pnl()
            position_info = [
                html.H4("Current Position", className="card-header"),
                html.Div([
                    html.P(f"Type: {bot.position['type']}"),
                    html.P(f"Entry Price: ${bot.position['entry_price']:.6f}"),
                    html.P(f"Quantity: {bot.position['quantity']:.0f}"),
                    html.P(f"Unrealized PnL: ${unrealized_pnl:.2f}"),
                    html.P(f"Stop Loss: ${bot.position['stop_loss']:.6f}"),
                    html.P(f"Take Profit: ${bot.position['take_profit']:.6f}"),
                    html.P(f"Entry Weight: {bot.position['entry_weight']}"),
                ], className="card-body")
            ]
        else:
            position_info = [
                html.H4("Current Position", className="card-header"),
                html.Div([html.P("No active position")], className="card-body")
            ]
        
        # Trade history
        recent_trades = bot.trade_history[-10:]  # Last 10 trades
        trade_items = []
        for trade in reversed(recent_trades):
            trade_text = f"{trade['timestamp'].strftime('%H:%M:%S')} - {trade['action']} at ${trade['price']:.6f}"
            if 'pnl' in trade:
                pnl_color = "text-success" if trade['pnl'] > 0 else "text-danger"
                trade_text += f" | PnL: <span class='{pnl_color}'>${trade['pnl']:.2f}</span>"
            trade_items.append(html.P(html.Span(trade_text, dangerously_allow_html=True)))
        
        trade_history = [
            html.H4("Recent Trades", className="card-header"),
            html.Div(trade_items, className="card-body")
        ]
        
        return runtime_info, balance_info, metrics, position_info, trade_history
    
    return app

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # Run backtest
        backtester = Backtester()
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        backtester.run_backtest(start_date, end_date)
    else:
        # Run live trading with dashboard
        bot = LeveragedTradingBot()
        app = create_dashboard(bot)
        print("🚀 Dashboard running on http://localhost:8050")
        app.run_server(debug=False, host='0.0.0.0', port=8050)
