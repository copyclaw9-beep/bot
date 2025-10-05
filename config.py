import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'testnet_api_key_here')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'testnet_secret_here')
TESTNET = True

# Trading Parameters
SYMBOL = 'DOGEUSDT'
LEVERAGE = 10
POSITION_SIZE_PERCENT = 0.05  # 5% of balance
QUANTITY = 1000  # Fixed quantity for DOGE

# Risk Management
STOP_LOSS_PERCENT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT = 0.03  # 3% take profit

# Indicator Parameters
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
BB_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
ATR_PERIOD = 14

# Weight Thresholds
LONG_THRESHOLD = 0.2
SHORT_THRESHOLD = -0.2
EXIT_THRESHOLD = 0.2

# Timeframes
TIMEFRAMES = ['5m', '15m', '1h']
INITIAL_CANDLES = 200
