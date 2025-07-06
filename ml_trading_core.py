# ml_trading_core.py

"""
Professional ML Trading System - Core Architecture
Hedge fund-quality system with ensemble ML, GPU acceleration, and live trading
"""

import os
import logging
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Central configuration for the trading system"""
    # Data settings
    DATA_START_DATE: str = "2022-01-01"
    CACHE_DIR: str = "./cache"
    DATA_UPDATE_HOUR: int = 2  # 2 AM updates

    # ML settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    ENSEMBLE_MIN_AGREEMENT: float = 0.6
    WALK_FORWARD_TRAIN_MONTHS: int = 12
    WALK_FORWARD_VAL_MONTHS: int = 3
    WALK_FORWARD_TEST_MONTHS: int = 1

    # Trading settings
    MAX_POSITION_SIZE: float = 0.10  # 10% max per position
    MAX_SECTOR_EXPOSURE: float = 0.30  # 30% max per sector
    KELLY_FRACTION: float = 0.25  # Fractional Kelly
    MIN_DAILY_VOLUME: float = 1_000_000  # $1M minimum
    MAX_PORTFOLIO_HEAT: float = 0.08  # 8% max risk

    # Risk settings
    MAX_VIX_THRESHOLD: float = 30.0
    MIN_WIN_PROBABILITY: float = 0.55
    MIN_SHARPE_RATIO: float = 1.0

    # Execution settings
    MARKET_OPEN_BUFFER: int = 15  # Minutes after open
    SIGNAL_GENERATION_TIME: str = "09:00"
    EXECUTION_TIME: str = "09:30"

    # API settings (store in environment variables)
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Feature engineering settings
    FEATURE_LOOKBACK_DAYS: int = 252  # 1 year
    MIN_DATA_POINTS: int = 60  # Minimum days for calculation


# =============================================================================
# WATCHLIST MANAGEMENT
# =============================================================================

class WatchlistManager:
    """Manages the trading universe of symbols"""

    def __init__(self, watchlist_file: str = None):
        """Initialize watchlist from file or defaults"""
        self.watchlist_file = watchlist_file

        # Try to load from file first
        if watchlist_file and os.path.exists(watchlist_file):
            self.symbols = self._load_watchlist_from_file(watchlist_file)
            logger.info(f"Loaded {len(self.symbols)} symbols from {watchlist_file}")
        else:
            self.symbols = self._initialize_watchlist()

        self.sector_map = self._load_sector_mapping()

        # Log summary statistics
        self._log_watchlist_summary()

    def _load_watchlist_from_file(self, filename: str) -> List[str]:
        """Load watchlist from CSV or JSON file"""
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
                # Assume first column is symbol
                return df.iloc[:, 0].tolist()
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    return data.get('symbols', [])
            else:
                logger.warning(f"Unsupported file format: {filename}")
                return self._initialize_watchlist()
        except Exception as e:
            logger.error(f"Error loading watchlist from {filename}: {e}")
            return self._initialize_watchlist()

    def save_watchlist(self, filename: str = None):
        """Save current watchlist to file"""
        if filename is None:
            filename = self.watchlist_file or 'watchlist.json'

        try:
            data = {
                'symbols': self.symbols,
                'count': len(self.symbols),
                'updated': datetime.now().isoformat(),
                'categories': self._get_category_counts()
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved watchlist to {filename}")
        except Exception as e:
            logger.error(f"Error saving watchlist: {e}")

    def _log_watchlist_summary(self):
        """Log summary statistics about the watchlist"""
        category_counts = self._get_category_counts()

        logger.info("Watchlist Summary:")
        logger.info(f"  Total Symbols: {len(self.symbols)}")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count}")

    def _get_category_counts(self) -> Dict[str, int]:
        """Get count of symbols by category"""
        category_counts = {}

        for symbol in self.symbols:
            sector = self.sector_map.get(symbol, 'Unknown')

            # Extract main category
            if 'ETF' in sector:
                if 'Bond' in sector:
                    category = 'Bond ETFs'
                elif 'Volatility' in sector:
                    category = 'Volatility Products'
                elif 'Currency' in sector:
                    category = 'Currency ETFs'
                elif 'Commodity' in sector or any(x in sector for x in ['Gold', 'Silver', 'Oil']):
                    category = 'Commodity ETFs'
                elif 'Inverse' in sector or '3x' in sector:
                    category = 'Leveraged/Inverse ETFs'
                elif 'International' in sector or 'Emerging' in sector:
                    category = 'International ETFs'
                else:
                    category = 'Sector/Index ETFs'
            else:
                category = sector

            category_counts[category] = category_counts.get(category, 0) + 1

        return category_counts

    def _initialize_watchlist(self) -> List[str]:
        """Initialize comprehensive watchlist with 200+ symbols across asset classes"""

        # Core S&P 500 leaders (expanded)
        sp500_leaders = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'ADBE', 'CRM',
            'NFLX', 'CSCO', 'XOM', 'CVX', 'PFE', 'BAC', 'WMT', 'KO', 'PEP',
            'TMO', 'ABT', 'MRK', 'ABBV', 'COST', 'AVGO', 'ACN', 'MCD', 'NKE',
            'LLY', 'ORCL', 'TXN', 'HON', 'QCOM', 'UPS', 'RTX', 'NEE', 'BMY',
            'AMGN', 'SBUX', 'GE', 'CAT', 'BA', 'MMM', 'GS', 'BLK', 'MDLZ',
            'T', 'VZ', 'CMCSA', 'INTC', 'AXP', 'IBM', 'LOW', 'LMT', 'CVS',
            'SCHW', 'AMT', 'C', 'MS', 'WFC', 'USB', 'PNC', 'TGT', 'FDX'
        ]

        # High momentum & growth stocks
        momentum_stocks = [
            'AMD', 'SHOP', 'SQ', 'ROKU', 'SNAP', 'PINS', 'UBER', 'LYFT',
            'DOCU', 'ZM', 'CRWD', 'DDOG', 'NET', 'SNOW', 'PLTR', 'COIN',
            'ABNB', 'DASH', 'RBLX', 'HOOD', 'RIVN', 'LCID', 'SOFI', 'UPST',
            'PATH', 'U', 'BILL', 'HUBS', 'TEAM', 'MDB', 'OKTA', 'TWLO',
            'ZS', 'PANW', 'FTNT', 'CYBR', 'S', 'ESTC', 'SPLK', 'NOW'
        ]

        # Comprehensive Sector ETFs
        sector_etfs = [
            # SPDR Sectors
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU', 'XLC',
            # Vanguard Sectors
            'VGT', 'VFH', 'VHT', 'VDE', 'VIS', 'VCR', 'VDC', 'VAW', 'VNQ', 'VPU',
            # Industry Specific
            'SMH', 'SOXX', 'IGV', 'HACK', 'FINX', 'XBI', 'IBB', 'XPH', 'IHI', 'IHF',
            'XHB', 'ITB', 'JETS', 'IYT', 'XRT', 'XHE', 'KRE', 'KBE', 'KBWB'
        ]

        # Bond & Fixed Income ETFs
        bonds_etfs = [
            'TLT', 'IEF', 'SHY', 'AGG', 'BND', 'LQD', 'HYG', 'JNK', 'EMB',
            'TIP', 'VTIP', 'STIP', 'MUB', 'HYD', 'BAB', 'MINT', 'NEAR', 'SHV',
            'BIL', 'SGOV', 'GOVT', 'MBB', 'VMBS', 'VCSH', 'VCIT', 'VCLT'
        ]

        # Volatility Products
        volatility_products = [
            'VIX', 'VXX', 'UVXY', 'SVXY', 'VIXY', 'VXZ', 'VIXM', 'SVIX',
            'UVIX', 'TVIX', 'ZIV', 'VMIN', 'VMAX'
        ]

        # Currency ETFs
        currency_etfs = [
            'UUP', 'UDN', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA', 'FXF',
            'CYB', 'CEW', 'DBV', 'USDU'
        ]

        # Commodity ETFs (expanded)
        commodities = [
            # Precious Metals
            'GLD', 'SLV', 'IAU', 'GDX', 'GDXJ', 'SIL', 'SILJ', 'PPLT', 'PALL',
            # Energy
            'USO', 'UCO', 'SCO', 'UNG', 'UGA', 'BNO', 'DBO', 'DBE', 'XOP',
            # Agriculture
            'DBA', 'CORN', 'WEAT', 'SOYB', 'CANE', 'JO', 'NIB', 'COW', 'MOO',
            # Base Metals & Materials
            'COPX', 'CPER', 'JJC', 'LIT', 'REMX', 'URA', 'XME', 'PICK', 'SLX'
        ]

        # International & Regional ETFs
        international = [
            # ADRs
            'TSM', 'BABA', 'NVO', 'ASML', 'TM', 'SAP', 'INFY', 'MELI', 'SE',
            'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'TCEHY', 'NTES', 'WB',
            # Country ETFs
            'EWJ', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWH', 'EWS', 'EWT',
            'INDA', 'FXI', 'MCHI', 'ASHR', 'EEM', 'VWO', 'IEMG', 'EFA', 'VEA'
        ]

        # Factor/Style ETFs
        factor_etfs = [
            'IWF', 'IWD', 'IWM', 'IWO', 'IWN', 'VTV', 'VUG', 'VBR', 'VBK',
            'MTUM', 'VLUE', 'QUAL', 'SIZE', 'USMV', 'SPLV', 'EEMV', 'EFAV'
        ]

        # ARK Innovation ETFs
        ark_etfs = [
            'ARKK', 'ARKG', 'ARKQ', 'ARKW', 'ARKF', 'ARKX', 'PRNT', 'IZRL'
        ]

        # Clean Energy & ESG
        clean_energy = [
            'ICLN', 'TAN', 'FAN', 'PBW', 'QCLN', 'ACES', 'SMOG', 'GRID',
            'LIT', 'BATT', 'DRIV', 'IDRV', 'KARS', 'HAIL'
        ]

        # REITs
        reits = [
            'VNQ', 'IYR', 'RWR', 'SCHH', 'XLRE', 'REET', 'USRT', 'REM',
            'MORT', 'KBWY', 'SRVR', 'INDS', 'HOMZ', 'REZ', 'RWX', 'VNQI'
        ]

        # Thematic ETFs
        thematic = [
            'BOTZ', 'ROBO', 'IRBO', 'IBOT', 'CIBR', 'IHAK', 'SNSR', 'PAVE',
            'BLOK', 'BLCN', 'LEGR', 'DAPP', 'FIVG', 'NXTG', 'DTEC', 'TEKK'
        ]

        # Inverse & Leveraged (for hedging)
        inverse_leveraged = [
            'SH', 'PSQ', 'DOG', 'SDS', 'QID', 'DXD', 'SPXU', 'SQQQ', 'SDOW',
            'TZA', 'SRTY', 'FAZ', 'TBT', 'TMV', 'SPXL', 'TQQQ', 'UDOW', 'UPRO'
        ]

        # Market Cap Weighted Indices
        indices = [
            'SPY', 'QQQ', 'DIA', 'IWM', 'MDY', 'OEF', 'RSP', 'SPLG', 'VOO'
        ]

        # Combine all symbols and remove duplicates
        all_symbols = list(set(
            sp500_leaders + momentum_stocks + sector_etfs + bonds_etfs +
            volatility_products + currency_etfs + commodities + international +
            factor_etfs + ark_etfs + clean_energy + reits + thematic +
            inverse_leveraged + indices
        ))

        # Sort alphabetically for consistency
        all_symbols = sorted(all_symbols)

        logger.info(f"Initialized watchlist with {len(all_symbols)} unique symbols")

        return all_symbols

    def _load_sector_mapping(self) -> Dict[str, str]:
        """Load comprehensive sector mapping for all symbols"""
        # In production, this would load from a database or API
        sector_map = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology', 'ADBE': 'Technology',
            'CRM': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology',
            'IBM': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology',
            'TXN': 'Technology', 'QCOM': 'Technology', 'AVGO': 'Technology',
            'NOW': 'Technology', 'PANW': 'Technology', 'FTNT': 'Technology',
            'CRWD': 'Technology', 'DDOG': 'Technology', 'NET': 'Technology',
            'SNOW': 'Technology', 'PLTR': 'Technology', 'ZS': 'Technology',
            'OKTA': 'Technology', 'TWLO': 'Technology', 'MDB': 'Technology',
            'TEAM': 'Technology', 'HUBS': 'Technology', 'BILL': 'Technology',
            'U': 'Technology', 'PATH': 'Technology', 'ESTC': 'Technology',
            'SPLK': 'Technology', 'S': 'Technology', 'CYBR': 'Technology',

            # Consumer Discretionary
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
            'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
            'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
            'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',
            'ABNB': 'Consumer Discretionary', 'UBER': 'Consumer Discretionary',
            'LYFT': 'Consumer Discretionary', 'DASH': 'Consumer Discretionary',
            'SQ': 'Consumer Discretionary', 'SHOP': 'Consumer Discretionary',
            'ROKU': 'Consumer Discretionary', 'NFLX': 'Consumer Discretionary',
            'DIS': 'Consumer Discretionary', 'RBLX': 'Consumer Discretionary',
            'LCID': 'Consumer Discretionary', 'RIVN': 'Consumer Discretionary',
            'NIO': 'Consumer Discretionary', 'LI': 'Consumer Discretionary',
            'XPEV': 'Consumer Discretionary',

            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'V': 'Financials', 'MA': 'Financials', 'GS': 'Financials',
            'MS': 'Financials', 'C': 'Financials', 'USB': 'Financials',
            'PNC': 'Financials', 'SCHW': 'Financials', 'BLK': 'Financials',
            'AXP': 'Financials', 'COF': 'Financials', 'COIN': 'Financials',
            'HOOD': 'Financials', 'SOFI': 'Financials', 'UPST': 'Financials',
            'BRK-B': 'Financials',

            # Healthcare
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
            'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
            'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'CVS': 'Healthcare',
            'AMGN': 'Healthcare', 'BMY': 'Healthcare', 'MDT': 'Healthcare',
            'DHR': 'Healthcare', 'ISRG': 'Healthcare', 'VRTX': 'Healthcare',
            'REGN': 'Healthcare', 'MRNA': 'Healthcare', 'ZM': 'Healthcare',
            'DOCU': 'Healthcare',

            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
            'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy',

            # Industrials
            'BA': 'Industrials', 'CAT': 'Industrials', 'UPS': 'Industrials',
            'HON': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials',
            'GE': 'Industrials', 'MMM': 'Industrials', 'FDX': 'Industrials',
            'UNP': 'Industrials', 'DE': 'Industrials', 'EMR': 'Industrials',

            # Consumer Staples
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
            'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
            'COST': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
            'PM': 'Consumer Staples', 'CL': 'Consumer Staples',

            # Utilities
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
            'D': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities',

            # Real Estate
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
            'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate',

            # Communication Services
            'T': 'Communication Services', 'VZ': 'Communication Services',
            'CMCSA': 'Communication Services', 'TMUS': 'Communication Services',
            'SNAP': 'Communication Services', 'PINS': 'Communication Services',

            # Materials
            'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
            'ECL': 'Materials', 'DD': 'Materials', 'NEM': 'Materials',

            # International ADRs
            'TSM': 'Technology', 'ASML': 'Technology', 'SAP': 'Technology',
            'INFY': 'Technology', 'BABA': 'Consumer Discretionary',
            'SE': 'Consumer Discretionary', 'MELI': 'Consumer Discretionary',
            'JD': 'Consumer Discretionary', 'PDD': 'Consumer Discretionary',
            'BIDU': 'Technology', 'NTES': 'Technology', 'TCEHY': 'Technology',
            'NVO': 'Healthcare', 'TM': 'Consumer Discretionary',
            'WB': 'Technology',

            # ETF Classifications
            'SPY': 'Broad Market ETF', 'QQQ': 'Technology ETF',
            'DIA': 'Broad Market ETF', 'IWM': 'Small Cap ETF',
            'XLK': 'Technology ETF', 'XLF': 'Financials ETF',
            'XLV': 'Healthcare ETF', 'XLE': 'Energy ETF',
            'XLI': 'Industrials ETF', 'XLY': 'Consumer Discretionary ETF',
            'XLP': 'Consumer Staples ETF', 'XLB': 'Materials ETF',
            'XLRE': 'Real Estate ETF', 'XLU': 'Utilities ETF',
            'XLC': 'Communication Services ETF',

            # Bond ETFs
            'TLT': 'Long-Term Bond ETF', 'IEF': 'Intermediate Bond ETF',
            'SHY': 'Short-Term Bond ETF', 'AGG': 'Aggregate Bond ETF',
            'BND': 'Total Bond ETF', 'LQD': 'Corporate Bond ETF',
            'HYG': 'High Yield Bond ETF', 'JNK': 'Junk Bond ETF',
            'EMB': 'Emerging Market Bond ETF', 'TIP': 'TIPS ETF',
            'MUB': 'Municipal Bond ETF',

            # Volatility Products
            'VXX': 'Volatility ETF', 'UVXY': 'Volatility ETF',
            'SVXY': 'Inverse Volatility ETF', 'VIXY': 'Volatility ETF',

            # Commodity ETFs
            'GLD': 'Gold ETF', 'SLV': 'Silver ETF', 'USO': 'Oil ETF',
            'UNG': 'Natural Gas ETF', 'DBA': 'Agriculture ETF',
            'CORN': 'Corn ETF', 'WEAT': 'Wheat ETF', 'SOYB': 'Soybean ETF',
            'GDX': 'Gold Miners ETF', 'GDXJ': 'Junior Gold Miners ETF',
            'XME': 'Metals & Mining ETF', 'COPX': 'Copper Miners ETF',
            'LIT': 'Lithium ETF', 'URA': 'Uranium ETF',

            # Clean Energy ETFs
            'ICLN': 'Clean Energy ETF', 'TAN': 'Solar ETF',
            'FAN': 'Wind Energy ETF', 'PBW': 'Clean Energy ETF',
            'QCLN': 'Clean Energy ETF', 'ACES': 'Clean Energy ETF',

            # Currency ETFs
            'UUP': 'US Dollar ETF', 'FXE': 'Euro ETF',
            'FXY': 'Japanese Yen ETF', 'FXB': 'British Pound ETF',

            # REIT ETFs
            'VNQ': 'Real Estate ETF', 'IYR': 'Real Estate ETF',
            'RWR': 'Real Estate ETF', 'SCHH': 'Real Estate ETF',

            # ARK ETFs
            'ARKK': 'Innovation ETF', 'ARKG': 'Genomics ETF',
            'ARKQ': 'Autonomous Tech ETF', 'ARKW': 'Internet ETF',
            'ARKF': 'Fintech ETF', 'ARKX': 'Space ETF',

            # Factor ETFs
            'MTUM': 'Momentum ETF', 'VLUE': 'Value ETF',
            'QUAL': 'Quality ETF', 'USMV': 'Low Volatility ETF',

            # International ETFs
            'EEM': 'Emerging Markets ETF', 'VWO': 'Emerging Markets ETF',
            'EFA': 'International Developed ETF', 'VEA': 'International Developed ETF',
            'FXI': 'China ETF', 'MCHI': 'China ETF', 'INDA': 'India ETF',
            'EWJ': 'Japan ETF', 'EWZ': 'Brazil ETF', 'EWG': 'Germany ETF',

            # Thematic ETFs
            'BOTZ': 'Robotics ETF', 'ROBO': 'Robotics ETF',
            'CIBR': 'Cybersecurity ETF', 'HACK': 'Cybersecurity ETF',
            'BLOK': 'Blockchain ETF', 'FINX': 'Fintech ETF',
            'PAVE': 'Infrastructure ETF', 'JETS': 'Airlines ETF',

            # Inverse/Leveraged ETFs
            'SH': 'Inverse S&P ETF', 'PSQ': 'Inverse QQQ ETF',
            'SPXU': '3x Inverse S&P ETF', 'SQQQ': '3x Inverse QQQ ETF',
            'SPXL': '3x S&P ETF', 'TQQQ': '3x QQQ ETF',
            'TBT': 'Inverse Bond ETF', 'TMV': '3x Inverse Bond ETF',
        }
        return sector_map

    def filter_by_liquidity(self, data: pd.DataFrame, min_volume: float) -> List[str]:
        """Filter symbols by liquidity requirements with asset-specific thresholds"""
        liquid_symbols = []

        for symbol in self.symbols:
            if symbol in data.columns.levels[1]:
                try:
                    recent_volume = data['Volume'][symbol].tail(20).mean()
                    recent_price = data['Close'][symbol].tail(20).mean()
                    dollar_volume = recent_volume * recent_price

                    # Get asset type
                    sector = self.sector_map.get(symbol, 'Unknown')

                    # Apply different liquidity thresholds based on asset type
                    if 'ETF' in sector:
                        # ETFs generally have good liquidity
                        threshold = min_volume * 0.5  # Lower threshold for ETFs
                    elif 'Bond' in sector:
                        # Bond ETFs might have lower volume
                        threshold = min_volume * 0.3
                    elif symbol in ['VXX', 'UVXY', 'SVXY', 'VIXY']:
                        # Volatility products need higher liquidity
                        threshold = min_volume * 2
                    elif any(x in sector for x in ['3x', 'Inverse']):
                        # Leveraged/Inverse products need higher liquidity
                        threshold = min_volume * 1.5
                    else:
                        # Regular stocks
                        threshold = min_volume

                    if dollar_volume >= threshold:
                        liquid_symbols.append(symbol)
                    else:
                        logger.debug(f"{symbol}: Low liquidity ${dollar_volume:,.0f} < ${threshold:,.0f}")

                except Exception as e:
                    logger.debug(f"Error checking liquidity for {symbol}: {e}")

        logger.info(f"Filtered to {len(liquid_symbols)} liquid symbols from {len(self.symbols)} total")

        return liquid_symbols

    def get_symbols_by_category(self, category: str) -> List[str]:
        """Get symbols by category (e.g., 'Technology', 'ETF', 'Bond ETF')"""
        return [symbol for symbol, sector in self.sector_map.items()
                if category.lower() in sector.lower()]

    def get_trading_universe(self, exclude_categories: List[str] = None) -> List[str]:
        """Get trading universe with optional category exclusions"""
        if exclude_categories is None:
            exclude_categories = []

        universe = []
        for symbol in self.symbols:
            sector = self.sector_map.get(symbol, 'Unknown')

            # Check if symbol should be excluded
            exclude = False
            for category in exclude_categories:
                if category.lower() in sector.lower():
                    exclude = True
                    break

            if not exclude:
                universe.append(symbol)

        return universe

    def get_hedging_instruments(self) -> Dict[str, List[str]]:
        """Get instruments suitable for hedging"""
        return {
            'volatility': self.get_symbols_by_category('Volatility'),
            'inverse_equity': [s for s in self.symbols if 'Inverse' in self.sector_map.get(s, '')],
            'bonds': self.get_symbols_by_category('Bond ETF'),
            'gold': ['GLD', 'IAU', 'GDX', 'GDXJ'],
            'currency': self.get_symbols_by_category('Currency ETF'),
            'defensive_sectors': ['XLP', 'XLU', 'VDC', 'VPU']  # Consumer Staples & Utilities
        }


# =============================================================================
# DATA PIPELINE
# =============================================================================

class DataPipeline:
    """Handles all data fetching, caching, and preprocessing"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, symbols: List[str], start_date: str = None,
                   end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
        """Fetch data with intelligent caching"""
        start_date = start_date or self.config.DATA_START_DATE
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        cache_file = f"{self.cache_dir}/market_data_{start_date}_{end_date}.parquet"

        if use_cache and os.path.exists(cache_file):
            # Check if cache is fresh (less than 1 day old)
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(days=1):
                logger.info(f"Loading cached data from {cache_file}")
                return pd.read_parquet(cache_file)

        logger.info(f"Fetching fresh data for {len(symbols)} symbols")

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for symbol in symbols:
                future = executor.submit(self._fetch_symbol_data, symbol, start_date, end_date)
                futures[future] = symbol

            all_data = {}
            for future in futures:
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")

        # Combine all data into MultiIndex DataFrame
        if all_data:
            combined_data = pd.concat(all_data, axis=1, keys=all_data.keys())
            combined_data = combined_data.swaplevel(axis=1)
            combined_data.sort_index(axis=1, inplace=True)

            # Save to cache
            combined_data.to_parquet(cache_file)
            logger.info(f"Cached data to {cache_file}")

            return combined_data
        else:
            return pd.DataFrame()

    def _fetch_symbol_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Add additional data if available
            try:
                info = ticker.info
                data['MarketCap'] = info.get('marketCap', np.nan)
                data['Sector'] = info.get('sector', 'Unknown')
            except:
                pass

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def update_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch real-time data during market hours"""
        realtime_data = {}

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for symbol in symbols:
                future = executor.submit(self._fetch_realtime_quote, symbol)
                futures[future] = symbol

            for future in futures:
                symbol = futures[future]
                try:
                    quote = future.result()
                    if quote:
                        realtime_data[symbol] = quote
                except Exception as e:
                    logger.error(f"Error fetching realtime data for {symbol}: {e}")

        return pd.DataFrame(realtime_data).T

    def _fetch_realtime_quote(self, symbol: str) -> Dict:
        """Fetch real-time quote for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            quote = ticker.info

            return {
                'price': quote.get('regularMarketPrice', np.nan),
                'volume': quote.get('regularMarketVolume', np.nan),
                'bid': quote.get('bid', np.nan),
                'ask': quote.get('ask', np.nan),
                'bid_size': quote.get('bidSize', np.nan),
                'ask_size': quote.get('askSize', np.nan),
                'timestamp': datetime.now()
            }
        except:
            return None


# =============================================================================
# ENHANCED FEATURE ENGINEERING
# =============================================================================

class EnhancedFeatureEngineer:
    """Comprehensive feature engineering with 30+ methods"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.feature_groups = {
            'price': [],
            'volume': [],
            'volatility': [],
            'technical': [],
            'microstructure': [],
            'pattern': [],
            'statistical': [],
            'ml_derived': [],
            'cross_asset': [],
            'composite': []
        }

    def engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer all features for a single symbol"""
        # Extract symbol data
        symbol_data = self._extract_symbol_data(data, symbol)

        if symbol_data is None or len(symbol_data) < self.config.MIN_DATA_POINTS:
            return pd.DataFrame()

        features = pd.DataFrame(index=symbol_data.index)

        # Price-based features
        features = self._add_price_features(symbol_data, features)

        # Volume features
        features = self._add_volume_features(symbol_data, features)

        # Volatility features
        features = self._add_volatility_features(symbol_data, features)

        # Technical indicators
        features = self._add_technical_indicators(symbol_data, features)

        # Market microstructure
        features = self._add_microstructure_features(symbol_data, features)

        # Pattern recognition
        features = self._add_pattern_features(symbol_data, features)

        # Statistical features
        features = self._add_statistical_features(symbol_data, features)

        # Advanced interaction features
        features = self._add_interaction_features(features)

        # ML-discovered features
        features = self._add_ml_features(features)

        # Cross-asset features
        features = self._add_cross_asset_features(data, symbol, features)

        # Composite scores
        features = self._add_composite_scores(features)

        # Clean and validate features
        features = self._clean_features(features)

        return features

    def _extract_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract data for a specific symbol from MultiIndex DataFrame"""
        try:
            symbol_data = data.xs(symbol, level=1, axis=1)
            return symbol_data.dropna()
        except:
            return None

    def _add_price_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive price-based features"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_price = data['Open']
        volume = data['Volume']

        # Returns at multiple timeframes
        for period in [1, 2, 3, 5, 10, 20, 60]:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'log_return_{period}d'] = np.log(close / close.shift(period))

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']

        # Exponential moving averages
        for period in [8, 12, 21, 26, 50]:
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            features[f'price_to_ema_{period}'] = close / features[f'ema_{period}']

        # VWAP
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['price_to_vwap'] = close / features['vwap']

        # Support and resistance levels
        for lookback in [10, 20, 50, 100]:
            features[f'resistance_{lookback}d'] = high.rolling(lookback).max()
            features[f'support_{lookback}d'] = low.rolling(lookback).min()
            features[f'price_position_{lookback}d'] = (close - features[f'support_{lookback}d']) / \
                                                      (features[f'resistance_{lookback}d'] - features[
                                                          f'support_{lookback}d'])

        # Fibonacci retracements
        for lookback in [20, 50]:
            high_val = high.rolling(lookback).max()
            low_val = low.rolling(lookback).min()
            diff = high_val - low_val

            for fib_level, fib_pct in [(0.236, '236'), (0.382, '382'), (0.5, '500'),
                                       (0.618, '618'), (0.786, '786')]:
                fib_price = high_val - (diff * fib_level)
                features[f'fib_{fib_pct}_{lookback}d'] = fib_price
                features[f'price_to_fib_{fib_pct}_{lookback}d'] = close / fib_price

        # Price channels
        for period in [20, 50]:
            features[f'high_channel_{period}'] = high.rolling(period).max()
            features[f'low_channel_{period}'] = low.rolling(period).min()
            features[f'channel_position_{period}'] = (close - features[f'low_channel_{period}']) / \
                                                     (features[f'high_channel_{period}'] - features[
                                                         f'low_channel_{period}'])

        self.feature_groups['price'].extend(features.columns.tolist())

        return features

    def _add_volume_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        volume = data['Volume']
        close = data['Close']
        high = data['High']
        low = data['Low']

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = volume.rolling(period).mean()
            features[f'volume_ratio_{period}'] = volume / features[f'volume_ma_{period}']

        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv'] = obv
        features['obv_ma_20'] = obv.rolling(20).mean()
        features['obv_divergence'] = obv - features['obv_ma_20']

        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        features['acc_dist'] = ad
        features['acc_dist_ma_20'] = ad.rolling(20).mean()

        # Chaikin Money Flow
        for period in [10, 20]:
            money_flow_vol = clv * volume
            features[f'cmf_{period}'] = money_flow_vol.rolling(period).sum() / volume.rolling(period).sum()

        # Volume Price Trend
        vpt = (close.pct_change() * volume).cumsum()
        features['vpt'] = vpt
        features['vpt_ma_20'] = vpt.rolling(20).mean()

        # Money Flow Index
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        for period in [14, 20]:
            positive_sum = positive_flow.rolling(period).sum()
            negative_sum = negative_flow.rolling(period).sum()
            mfi = 100 - (100 / (1 + positive_sum / negative_sum))
            features[f'mfi_{period}'] = mfi

        # Volume-weighted momentum
        features['volume_weighted_momentum'] = (close.pct_change() * volume).rolling(20).sum() / volume.rolling(
            20).sum()

        self.feature_groups['volume'].extend(
            [col for col in features.columns if col not in self.feature_groups['price']])

        return features

    def _add_volatility_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_price = data['Open']

        # ATR at multiple timeframes
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)

        for period in [10, 14, 20, 30]:
            atr = tr.rolling(period).mean()
            features[f'atr_{period}'] = atr
            features[f'atr_pct_{period}'] = atr / close * 100

        # Bollinger Bands with multiple parameters
        for period, num_std in [(10, 1.5), (20, 2), (20, 2.5), (30, 2)]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}_{int(num_std * 10)}'] = ma + (num_std * std)
            features[f'bb_lower_{period}_{int(num_std * 10)}'] = ma - (num_std * std)
            features[f'bb_width_{period}_{int(num_std * 10)}'] = features[f'bb_upper_{period}_{int(num_std * 10)}'] - \
                                                                 features[f'bb_lower_{period}_{int(num_std * 10)}']
            features[f'bb_position_{period}_{int(num_std * 10)}'] = (close - features[
                f'bb_lower_{period}_{int(num_std * 10)}']) / \
                                                                    features[f'bb_width_{period}_{int(num_std * 10)}']

        # Keltner Channels
        for period, mult in [(20, 1.5), (20, 2), (30, 2)]:
            ma = close.ewm(span=period, adjust=False).mean()
            atr = tr.ewm(span=period, adjust=False).mean()
            features[f'kc_upper_{period}_{int(mult * 10)}'] = ma + (mult * atr)
            features[f'kc_lower_{period}_{int(mult * 10)}'] = ma - (mult * atr)
            features[f'kc_position_{period}_{int(mult * 10)}'] = (close - features[
                f'kc_lower_{period}_{int(mult * 10)}']) / \
                                                                 (features[f'kc_upper_{period}_{int(mult * 10)}'] -
                                                                  features[f'kc_lower_{period}_{int(mult * 10)}'])

        # Historical volatility
        for period in [5, 10, 20, 30, 60]:
            features[f'hist_vol_{period}'] = close.pct_change().rolling(period).std() * np.sqrt(252)

        # Parkinson volatility
        for period in [10, 20, 30]:
            parkinson = np.sqrt(252 / (4 * np.log(2)) * ((np.log(high / low) ** 2).rolling(period).mean()))
            features[f'parkinson_vol_{period}'] = parkinson

        # Garman-Klass volatility
        for period in [10, 20, 30]:
            gk = np.sqrt(252 * (0.5 * (np.log(high / low) ** 2) -
                                (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2)).rolling(period).mean())
            features[f'garman_klass_vol_{period}'] = gk

        # Volatility regime detection
        vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252)
        vol_60 = close.pct_change().rolling(60).std() * np.sqrt(252)
        features['vol_regime'] = vol_20 / vol_60

        self.feature_groups['volatility'].extend([col for col in features.columns
                                                  if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_technical_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # RSI with multiple periods
        for period in [7, 14, 21, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # RSI divergence
            features[f'rsi_{period}_divergence'] = features[f'rsi_{period}'] - features[f'rsi_{period}'].rolling(
                20).mean()

        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            features[f'macd_{fast}_{slow}_{signal}'] = macd_line
            features[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line
            features[f'macd_hist_{fast}_{slow}_{signal}'] = macd_line - signal_line

        # Stochastic oscillator
        for period, smooth in [(14, 3), (21, 5), (5, 3)]:
            lowest_low = low.rolling(period).min()
            highest_high = high.rolling(period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            features[f'stoch_k_{period}'] = k_percent
            features[f'stoch_d_{period}_{smooth}'] = k_percent.rolling(smooth).mean()

        # Williams %R
        for period in [14, 20]:
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            features[f'williams_r_{period}'] = -100 * ((highest_high - close) / (highest_high - lowest_low))

        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(period).mean()
            mad = (typical_price - sma).abs().rolling(period).mean()
            features[f'cci_{period}'] = (typical_price - sma) / (0.015 * mad)

        # Average Directional Index (ADX)
        for period in [14, 20]:
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr_smooth = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr_smooth)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            features[f'adx_{period}'] = dx.ewm(span=period, adjust=False).mean()
            features[f'plus_di_{period}'] = plus_di
            features[f'minus_di_{period}'] = minus_di

        # Aroon
        for period in [20, 25]:
            high_period = high.rolling(period + 1).apply(lambda x: x.argmax())
            low_period = low.rolling(period + 1).apply(lambda x: x.argmin())
            features[f'aroon_up_{period}'] = 100 * (period - high_period) / period
            features[f'aroon_down_{period}'] = 100 * (period - low_period) / period
            features[f'aroon_osc_{period}'] = features[f'aroon_up_{period}'] - features[f'aroon_down_{period}']

        # Ultimate Oscillator
        bp = close - pd.DataFrame([low, close.shift(1)]).min()
        tr = pd.DataFrame([high - low,
                           abs(high - close.shift(1)),
                           abs(low - close.shift(1))]).max()

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        features['ultimate_osc'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

        # PPO (Percentage Price Oscillator)
        for fast, slow in [(12, 26), (10, 20)]:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            features[f'ppo_{fast}_{slow}'] = 100 * (ema_fast - ema_slow) / ema_slow

        # TRIX
        for period in [14, 20]:
            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            features[f'trix_{period}'] = 10000 * ema3.pct_change()

        # CMO (Chande Momentum Oscillator)
        for period in [14, 20]:
            delta = close.diff()
            up = delta.where(delta > 0, 0).rolling(period).sum()
            down = -delta.where(delta < 0, 0).rolling(period).sum()
            features[f'cmo_{period}'] = 100 * (up - down) / (up + down)

        self.feature_groups['technical'].extend([col for col in features.columns
                                                 if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_microstructure_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # Bid-ask spread proxy (using high-low as proxy)
        features['spread_proxy'] = (high - low) / close
        features['spread_proxy_ma_20'] = features['spread_proxy'].rolling(20).mean()

        # Intraday volatility
        features['intraday_vol'] = (high - low) / close
        features['intraday_vol_ma_20'] = features['intraday_vol'].rolling(20).mean()

        # Order flow imbalance estimation
        # Using close position within the day's range as proxy
        features['order_flow_imbalance'] = (close - low) / (high - low)
        features['order_flow_imbalance_ma_10'] = features['order_flow_imbalance'].rolling(10).mean()

        # Amihud illiquidity measure
        features['amihud_illiq'] = abs(close.pct_change()) / (volume * close)
        features['amihud_illiq_ma_20'] = features['amihud_illiq'].rolling(20).mean()

        # Kyle's lambda (simplified version)
        # Price impact = price change / volume
        price_change = close.pct_change()
        features['kyle_lambda'] = abs(price_change) / np.log(1 + volume)
        features['kyle_lambda_ma_20'] = features['kyle_lambda'].rolling(20).mean()

        # Microstructure noise ratio
        # Ratio of short-term to long-term variance
        short_var = close.pct_change().rolling(5).var()
        long_var = close.pct_change().rolling(20).var()
        features['noise_ratio'] = short_var / long_var

        # Effective spread proxy
        mid_price = (high + low) / 2
        features['effective_spread'] = 2 * abs(close - mid_price) / mid_price

        # Roll's implicit spread estimator
        price_changes = close.pct_change()
        features['roll_spread'] = 2 * np.sqrt(-price_changes.rolling(20).cov(price_changes.shift(1)))

        self.feature_groups['microstructure'].extend([col for col in features.columns
                                                      if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_pattern_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_price = data['Open']

        # Candlestick patterns (simplified versions)
        # Doji
        body = abs(close - open_price)
        range_hl = high - low
        features['doji'] = (body / range_hl < 0.1).astype(int)

        # Hammer
        lower_shadow = pd.DataFrame([close, open_price]).min() - low
        features['hammer'] = ((lower_shadow > 2 * body) &
                              (close > open_price) &
                              (high - pd.DataFrame([close, open_price]).max() < body)).astype(int)

        # Shooting star
        upper_shadow = high - pd.DataFrame([close, open_price]).max()
        features['shooting_star'] = ((upper_shadow > 2 * body) &
                                     (close < open_price) &
                                     (pd.DataFrame([close, open_price]).min() - low < body)).astype(int)

        # Engulfing patterns
        prev_body = abs(close.shift(1) - open_price.shift(1))
        features['bullish_engulfing'] = ((close > open_price) &
                                         (close.shift(1) < open_price.shift(1)) &
                                         (body > prev_body) &
                                         (open_price < close.shift(1)) &
                                         (close > open_price.shift(1))).astype(int)

        features['bearish_engulfing'] = ((close < open_price) &
                                         (close.shift(1) > open_price.shift(1)) &
                                         (body > prev_body) &
                                         (open_price > close.shift(1)) &
                                         (close < open_price.shift(1))).astype(int)

        # Pin bar
        features['pin_bar_bullish'] = ((lower_shadow > 2.5 * body) &
                                       (upper_shadow < 0.3 * body)).astype(int)
        features['pin_bar_bearish'] = ((upper_shadow > 2.5 * body) &
                                       (lower_shadow < 0.3 * body)).astype(int)

        # Inside bar
        features['inside_bar'] = ((high < high.shift(1)) &
                                  (low > low.shift(1))).astype(int)

        # Outside bar
        features['outside_bar'] = ((high > high.shift(1)) &
                                   (low < low.shift(1))).astype(int)

        # Consecutive patterns
        returns = close.pct_change()
        features['consecutive_up_days'] = (returns > 0).rolling(5).sum()
        features['consecutive_down_days'] = (returns < 0).rolling(5).sum()

        # Pattern strength scoring
        pattern_cols = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing',
                        'bearish_engulfing', 'pin_bar_bullish', 'pin_bar_bearish']
        features['bullish_pattern_score'] = features[['hammer', 'bullish_engulfing',
                                                      'pin_bar_bullish']].sum(axis=1)
        features['bearish_pattern_score'] = features[['shooting_star', 'bearish_engulfing',
                                                      'pin_bar_bearish']].sum(axis=1)

        # Pattern combination features
        features['pattern_strength'] = features['bullish_pattern_score'] - features['bearish_pattern_score']

        self.feature_groups['pattern'].extend([col for col in features.columns
                                               if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_statistical_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        close = data['Close']
        returns = close.pct_change()

        # Rolling statistics
        for period in [5, 10, 20, 50]:
            # Skewness
            features[f'skew_{period}'] = returns.rolling(period).skew()

            # Kurtosis
            features[f'kurtosis_{period}'] = returns.rolling(period).kurt()

            # Z-score
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'zscore_{period}'] = (close - mean) / std

            # Percentile rank
            features[f'percentile_rank_{period}'] = close.rolling(period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )

        # Jarque-Bera test statistic
        from scipy import stats
        for period in [20, 50]:
            features[f'jarque_bera_{period}'] = returns.rolling(period).apply(
                lambda x: stats.jarque_bera(x)[0] if len(x) == period else np.nan
            )

        # Autocorrelation
        for lag in [1, 5, 10, 20]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) >= lag + 1 else np.nan
            )

        # Hurst exponent (simplified)
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]

        features['hurst_exponent'] = close.rolling(50).apply(
            lambda x: hurst_exponent(x.values) if len(x) == 50 else np.nan
        )

        # Shannon entropy
        def shannon_entropy(ts, bins=10):
            hist, _ = np.histogram(ts, bins=bins)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))

        features['shannon_entropy'] = returns.rolling(50).apply(
            lambda x: shannon_entropy(x) if len(x) == 50 else np.nan
        )

        # Sample entropy (simplified)
        features['sample_entropy'] = returns.rolling(30).std() / returns.rolling(10).std()

        self.feature_groups['statistical'].extend([col for col in features.columns
                                                   if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add advanced interaction features"""
        # Golden Cross / Death Cross
        if 'sma_50' in features and 'sma_200' in features:
            golden_cross = (features['sma_50'] > features['sma_200']) & \
                           (features['sma_50'].shift(1) <= features['sma_200'].shift(1))
            death_cross = (features['sma_50'] < features['sma_200']) & \
                          (features['sma_50'].shift(1) >= features['sma_200'].shift(1))

            features['golden_cross'] = golden_cross.astype(int)
            features['death_cross'] = death_cross.astype(int)
            features['days_since_golden_cross'] = golden_cross.cumsum().groupby(golden_cross.cumsum()).cumcount()
            features['days_since_death_cross'] = death_cross.cumsum().groupby(death_cross.cumsum()).cumcount()

        # MA Crossovers
        if 'sma_5' in features and 'sma_20' in features:
            features['ma_cross_5_20'] = np.sign(features['sma_5'] - features['sma_20'])
            features['ma_cross_5_20_change'] = features['ma_cross_5_20'].diff()

        if 'sma_20' in features and 'sma_50' in features:
            features['ma_cross_20_50'] = np.sign(features['sma_20'] - features['sma_50'])
            features['ma_cross_20_50_change'] = features['ma_cross_20_50'].diff()

        # RSI Divergence
        if 'rsi_14' in features and 'return_5d' in features:
            price_trend = features['return_5d'] > 0
            rsi_trend = features['rsi_14'] > features['rsi_14'].shift(5)
            features['rsi_price_divergence'] = (price_trend != rsi_trend).astype(int)

        # MACD Divergence
        if 'macd_12_26_9' in features and 'return_5d' in features:
            macd_trend = features['macd_12_26_9'] > features['macd_12_26_9'].shift(5)
            features['macd_price_divergence'] = (price_trend != macd_trend).astype(int)

        # Volume-Price Confirmation
        if 'volume_ratio_20' in features and 'return_1d' in features:
            features['volume_price_confirm'] = (
                    (features['volume_ratio_20'] > 1) & (features['return_1d'] > 0)
            ).astype(int)

        # Bollinger/Keltner Squeeze
        if 'bb_width_20_20' in features and 'kc_upper_20_20' in features and 'kc_lower_20_20' in features:
            kc_width = features['kc_upper_20_20'] - features['kc_lower_20_20']
            features['bb_kc_squeeze'] = (features['bb_width_20_20'] < kc_width).astype(int)

        # Support/Resistance with indicators
        if 'support_20d' in features and 'rsi_14' in features:
            features['support_bounce_rsi'] = (
                    (features['price_position_20d'] < 0.2) & (features['rsi_14'] < 30)
            ).astype(int)

        if 'resistance_20d' in features and 'rsi_14' in features:
            features['resistance_test_rsi'] = (
                    (features['price_position_20d'] > 0.8) & (features['rsi_14'] > 70)
            ).astype(int)

        # Multi-timeframe momentum alignment
        momentum_features = []
        for period in [5, 10, 20]:
            if f'return_{period}d' in features:
                momentum_features.append(f'return_{period}d')

        if momentum_features:
            features['momentum_alignment'] = (features[momentum_features] > 0).sum(axis=1) / len(momentum_features)

        self.feature_groups['composite'].extend([col for col in features.columns
                                                 if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_ml_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add ML-discovered features"""
        # Polynomial features for key indicators
        key_features = []
        for col in ['rsi_14', 'macd_12_26_9', 'atr_pct_14', 'volume_ratio_20']:
            if col in features:
                key_features.append(col)
                # Add polynomial features
                features[f'{col}_squared'] = features[col] ** 2
                features[f'{col}_cubed'] = features[col] ** 3
                features[f'{col}_sqrt'] = np.sqrt(np.abs(features[col]))

        # Interaction terms
        if len(key_features) >= 2:
            for i in range(len(key_features)):
                for j in range(i + 1, len(key_features)):
                    features[f'{key_features[i]}_x_{key_features[j]}'] = \
                        features[key_features[i]] * features[key_features[j]]

        # Fourier features for cyclical patterns
        if 'return_1d' in features:
            for period in [5, 10, 20]:
                features[f'fourier_sin_{period}'] = np.sin(2 * np.pi * features.index.day / period)
                features[f'fourier_cos_{period}'] = np.cos(2 * np.pi * features.index.day / period)

        # Fractal dimension (simplified)
        if 'return_1d' in features:
            def fractal_dimension(ts, max_k=5):
                N = len(ts)
                L = []
                for k in range(1, max_k + 1):
                    n_max = int(N / k)
                    L_k = 0
                    for m in range(k):
                        R_m = ts[m::k][:n_max]
                        if len(R_m) > 1:
                            L_k += np.sum(np.abs(np.diff(R_m))) / (n_max - 1)
                    L.append(L_k / k)

                if len(L) > 1:
                    x = np.log(range(1, max_k + 1))
                    y = np.log(L)
                    return -np.polyfit(x, y, 1)[0]
                return np.nan

            features['fractal_dimension'] = features['return_1d'].rolling(50).apply(
                lambda x: fractal_dimension(x.values) if len(x) == 50 else np.nan
            )

        self.feature_groups['ml_derived'].extend([col for col in features.columns
                                                  if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_cross_asset_features(self, data: pd.DataFrame, symbol: str,
                                  features: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset features"""
        # VIX features (if available)
        if 'VIX' in data.columns.levels[1]:
            try:
                vix_data = data.xs('VIX', level=1, axis=1)['Close']
                vix_aligned = vix_data.reindex(features.index, method='ffill')

                features['vix_level'] = vix_aligned
                features['vix_ma_20'] = vix_aligned.rolling(20).mean()
                features['vix_change_5d'] = vix_aligned.pct_change(5)
                features['vix_high_regime'] = (vix_aligned > 20).astype(int)
                features['vix_extreme_regime'] = (vix_aligned > 30).astype(int)
            except:
                pass

        # Dollar index (DXY proxy using UUP if available)
        if 'UUP' in data.columns.levels[1]:
            try:
                dxy_data = data.xs('UUP', level=1, axis=1)['Close']
                dxy_aligned = dxy_data.reindex(features.index, method='ffill')

                features['dollar_strength'] = dxy_aligned.pct_change(20)
                features['dollar_trend'] = np.sign(dxy_aligned - dxy_aligned.rolling(50).mean())
            except:
                pass

        # Market correlation
        if 'SPY' in data.columns.levels[1]:
            try:
                spy_data = data.xs('SPY', level=1, axis=1)['Close']
                spy_returns = spy_data.pct_change().reindex(features.index, method='ffill')
                symbol_returns = data.xs(symbol, level=1, axis=1)['Close'].pct_change()

                features['correlation_spy_20d'] = symbol_returns.rolling(20).corr(spy_returns)
                features['correlation_spy_60d'] = symbol_returns.rolling(60).corr(spy_returns)
                features['beta_spy_60d'] = symbol_returns.rolling(60).cov(spy_returns) / spy_returns.rolling(60).var()
            except:
                pass

        self.feature_groups['cross_asset'].extend([col for col in features.columns
                                                   if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _add_composite_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add composite scoring features"""
        # Bull market score
        bull_factors = []
        if 'sma_50' in features and 'sma_200' in features:
            bull_factors.append((features['sma_50'] > features['sma_200']).astype(int))
        if 'price_to_sma_50' in features:
            bull_factors.append((features['price_to_sma_50'] > 1).astype(int))
        if 'rsi_14' in features:
            bull_factors.append(((features['rsi_14'] > 50) & (features['rsi_14'] < 70)).astype(int))
        if 'macd_hist_12_26_9' in features:
            bull_factors.append((features['macd_hist_12_26_9'] > 0).astype(int))
        if 'adx_14' in features:
            bull_factors.append((features['adx_14'] > 25).astype(int))

        if bull_factors:
            features['bull_market_score'] = sum(bull_factors) / len(bull_factors)

        # Mean reversion setup score
        mr_factors = []
        if 'rsi_14' in features:
            mr_factors.append((features['rsi_14'] < 30).astype(int))
            mr_factors.append((features['rsi_14'] > 70).astype(int))
        if 'bb_position_20_20' in features:
            mr_factors.append((features['bb_position_20_20'] < 0).astype(int))
            mr_factors.append((features['bb_position_20_20'] > 1).astype(int))
        if 'zscore_20' in features:
            mr_factors.append((abs(features['zscore_20']) > 2).astype(int))

        if mr_factors:
            features['mean_reversion_score'] = sum(mr_factors) / len(mr_factors)

        # Breakout setup score
        breakout_factors = []
        if 'volume_ratio_20' in features:
            breakout_factors.append((features['volume_ratio_20'] > 1.5).astype(int))
        if 'atr_pct_14' in features and 'atr_pct_14' in features:
            atr_expansion = features['atr_pct_14'] > features['atr_pct_14'].rolling(20).mean()
            breakout_factors.append(atr_expansion.astype(int))
        if 'bb_width_20_20' in features:
            bb_squeeze = features['bb_width_20_20'] < features['bb_width_20_20'].rolling(50).quantile(0.2)
            breakout_factors.append(bb_squeeze.astype(int))
        if 'price_position_50d' in features:
            near_resistance = features['price_position_50d'] > 0.9
            breakout_factors.append(near_resistance.astype(int))

        if breakout_factors:
            features['breakout_setup_score'] = sum(breakout_factors) / len(breakout_factors)

        # Trend exhaustion score
        exhaustion_factors = []
        if 'consecutive_up_days' in features:
            exhaustion_factors.append((features['consecutive_up_days'] > 7).astype(int))
        if 'consecutive_down_days' in features:
            exhaustion_factors.append((features['consecutive_down_days'] > 7).astype(int))
        if 'rsi_14_divergence' in features:
            exhaustion_factors.append((abs(features['rsi_14_divergence']) > 20).astype(int))
        if 'volume_ratio_20' in features and 'return_1d' in features:
            volume_divergence = (features['volume_ratio_20'] < 0.7) & (abs(features['return_1d']) > 0.02)
            exhaustion_factors.append(volume_divergence.astype(int))

        if exhaustion_factors:
            features['trend_exhaustion_score'] = sum(exhaustion_factors) / len(exhaustion_factors)

        self.feature_groups['composite'].extend([col for col in features.columns
                                                 if col not in sum(list(self.feature_groups.values()), [])])

        return features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill missing values (limit to 5 periods)
        features = features.fillna(method='ffill', limit=5)

        # Backward fill any remaining NaNs at the start
        features = features.fillna(method='bfill', limit=5)

        # Drop any remaining rows with NaNs
        features = features.dropna()

        # Ensure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')

        return features

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis"""
        return self.feature_groups


# =============================================================================
# MAIN INITIALIZATION
# =============================================================================

def initialize_trading_system(watchlist_file: str = None):
    """Initialize the complete trading system"""
    logger.info("Initializing ML Trading System...")

    # Create configuration
    config = TradingConfig()

    # Initialize components
    watchlist_manager = WatchlistManager(watchlist_file=watchlist_file)
    data_pipeline = DataPipeline(config)
    feature_engineer = EnhancedFeatureEngineer(config)

    logger.info(f"System initialized with {len(watchlist_manager.symbols)} symbols")
    logger.info(f"GPU Device: {config.DEVICE}")

    return {
        'config': config,
        'watchlist': watchlist_manager,
        'data_pipeline': data_pipeline,
        'feature_engineer': feature_engineer
    }


if __name__ == "__main__":
    # Initialize system
    system = initialize_trading_system()

    # Test data fetching
    logger.info("Testing data fetch for first 5 symbols...")
    test_symbols = system['watchlist'].symbols[:5]
    data = system['data_pipeline'].fetch_data(test_symbols)

    if not data.empty:
        logger.info(f"Successfully fetched data: {data.shape}")

        # Test feature engineering
        logger.info("Testing feature engineering...")
        features = system['feature_engineer'].engineer_features(data, test_symbols[0])

        if not features.empty:
            logger.info(f"Successfully engineered {len(features.columns)} features")
            logger.info(f"Feature groups: {list(system['feature_engineer'].get_feature_groups().keys())}")
        else:
            logger.error("Feature engineering failed")
    else:
        logger.error("Data fetch failed")