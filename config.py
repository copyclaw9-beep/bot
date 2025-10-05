# Configuration settings - NEVER commit real API keys!
import os
from dotenv import load_dotenv

load_dotenv()

# Exchange Configuration
EXCHANGE = "binance"  # or "coinbase", "kraken", etc.
TESTNET = True  # Always start with testnet!

# API Keys - Store in .env file, not here!
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Parameters
SYMBOL = "BTCUSDT"
QUANTITY = 0.001  # Small amount for testing
