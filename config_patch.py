# Runtime patch for TradingConfig
import sys

# Remove cached module
if 'ml_trading_core' in sys.modules:
    del sys.modules['ml_trading_core']

# Import and patch
import ml_trading_core

# Save original init
_original_init = ml_trading_core.TradingConfig.__init__

def patched_init(self):
    _original_init(self)
    # Add missing attributes
    self.DATA_LOOKBACK_DAYS = 730
    self.SAMPLE_WEIGHT_HALFLIFE = 60
    self.MODEL_DIR = './data/models'
    print("Applied TradingConfig patch")

# Apply patch
ml_trading_core.TradingConfig.__init__ = patched_init
