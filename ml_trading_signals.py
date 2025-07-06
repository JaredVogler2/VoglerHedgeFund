# ml_trading_signals.py

"""
ML Trading System - Signal Generation & Risk Management
Professional signal generation with Bayesian scoring and institutional risk controls
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    timestamp: datetime
    direction: str  # 'long' or 'short'
    win_probability: float
    expected_return: float
    predicted_volatility: float
    confidence_score: float
    bayesian_score: float
    risk_score: float
    position_size: float
    stop_loss: float
    take_profit: float
    features_snapshot: Dict
    model_predictions: Dict

class SignalGenerator:
    """Generates trading signals from ML predictions"""
    
    def __init__(self, config, feature_engineer, prediction_engine):
        self.config = config
        self.feature_engineer = feature_engineer
        self.prediction_engine = prediction_engine
        self.market_regime = MarketRegimeDetector(config)
        
    def generate_signals(self, market_data: pd.DataFrame, 
                        predictions: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals for all symbols"""
        signals = []
        
        # Detect current market regime
        regime = self.market_regime.detect_regime(market_data)
        logger.info(f"Current market regime: {regime}")
        
        # Generate signals for each symbol
        for symbol, symbol_predictions in predictions.items():
            if symbol_predictions.empty:
                continue
                
            # Get latest prediction
            latest_pred = symbol_predictions.iloc[-1]
            
            # Check basic criteria
            if not self._meets_basic_criteria(latest_pred):
                continue
            
            # Calculate Bayesian score
            bayesian_score = self._calculate_bayesian_score(
                win_probability=latest_pred['consensus_direction'],
                expected_return=latest_pred['consensus_return'],
                predicted_volatility=latest_pred.get('predicted_volatility', 0.02)
            )
            
            # Check regime filters
            if not self._passes_regime_filter(regime, bayesian_score):
                continue
            
            # Check liquidity
            if not self._check_liquidity(market_data, symbol):
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(
                symbol, latest_pred, market_data, regime
            )
            
            if position_size < 0.001:  # Minimum position size
                continue
            
            # Calculate stop loss and take profit
            stops = self._calculate_stops(
                symbol, latest_pred, market_data
            )
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                direction='long' if latest_pred['consensus_direction'] > 0.5 else 'short',
                win_probability=latest_pred['consensus_direction'],
                expected_return=latest_pred['consensus_return'],
                predicted_volatility=latest_pred.get('predicted_volatility', 0.02),
                confidence_score=latest_pred.get('confidence_score', 0.5),
                bayesian_score=bayesian_score,
                risk_score=self._calculate_risk_score(latest_pred, regime),
                position_size=position_size,
                stop_loss=stops['stop_loss'],
                take_profit=stops['take_profit'],
                features_snapshot=self._get_feature_snapshot(market_data, symbol),
                model_predictions=latest_pred.to_dict()
            )
            
            signals.append(signal)
        
        # Apply portfolio-level filters
        signals = self._apply_portfolio_filters(signals)
        
        # Rank and select top signals
        signals = self._rank_and_select_signals(signals)
        
        logger.info(f"Generated {len(signals)} trading signals")
        
        return signals
    
    def _meets_basic_criteria(self, prediction: pd.Series) -> bool:
        """Check if prediction meets basic criteria"""
        # Minimum win probability
        if prediction['consensus_direction'] < self.config.MIN_WIN_PROBABILITY:
            return False
        
        # Minimum expected return (annualized)
        expected_annual_return = prediction['consensus_return'] * 252
        if abs(expected_annual_return) < 0.10:  # 10% minimum
            return False
        
        # Model agreement
        if prediction.get('consensus_agreement', 1.0) > 0.3:  # Max 30% disagreement
            return False
        
        return True
    
    def _calculate_bayesian_score(self, win_probability: float, 
                                 expected_return: float, 
                                 predicted_volatility: float) -> float:
        """Calculate Bayesian score for signal quality"""
        # Adjust for risk
        risk_adjusted_return = expected_return / (predicted_volatility + 0.001)
        
        # Bayesian score combines probability and magnitude
        score = win_probability * abs(risk_adjusted_return)
        
        # Apply non-linear transformation to emphasize high-confidence signals
        score = np.sign(expected_return) * (score ** 1.5)
        
        return score
    
    def _passes_regime_filter(self, regime: Dict, bayesian_score: float) -> bool:
        """Check if signal passes regime-based filters"""
        # High volatility regime
        if regime['volatility_regime'] == 'extreme' and abs(bayesian_score) < 2.0:
            return False
        
        # Bear market filter
        if regime['trend_regime'] == 'strong_bear' and bayesian_score > 0:
            # Only allow strong short signals in bear markets
            return False
        
        # Low liquidity regime
        if regime['liquidity_regime'] == 'low':
            return abs(bayesian_score) > 1.5  # Higher threshold
        
        return True
    
    def _check_liquidity(self, market_data: pd.DataFrame, symbol: str) -> bool:
        """Check if symbol meets liquidity requirements"""
        try:
            symbol_data = market_data.xs(symbol, level=1, axis=1)
            
            # Average daily volume in dollars
            avg_volume = symbol_data['Volume'].tail(20).mean()
            avg_price = symbol_data['Close'].tail(20).mean()
            daily_dollar_volume = avg_volume * avg_price
            
            return daily_dollar_volume >= self.config.MIN_DAILY_VOLUME
        except:
            return False
    
    def _calculate_position_size(self, symbol: str, prediction: pd.Series,
                               market_data: pd.DataFrame, regime: Dict) -> float:
        """Calculate position size using Kelly Criterion and risk parity"""
        # Get symbol volatility
        symbol_data = market_data.xs(symbol, level=1, axis=1)
        returns = symbol_data['Close'].pct_change()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Kelly Criterion
        win_prob = prediction['consensus_direction']
        win_prob = max(0.01, min(0.99, win_prob))  # Clip to avoid extreme values
        
        expected_return = abs(prediction['consensus_return'])
        
        # Full Kelly
        if win_prob > 0.5:
            kelly_f = (win_prob * expected_return - (1 - win_prob) * expected_return) / expected_return
        else:
            kelly_f = 0
        
        # Apply fractional Kelly
        position_size = kelly_f * self.config.KELLY_FRACTION
        
        # Risk parity adjustment
        target_risk = 0.02  # 2% portfolio risk per position
        risk_parity_size = target_risk / volatility
        
        # Combine Kelly and risk parity
        position_size = min(position_size, risk_parity_size)
        
        # Regime adjustments
        if regime['volatility_regime'] == 'high':
            position_size *= 0.7
        elif regime['volatility_regime'] == 'extreme':
            position_size *= 0.5
        
        # Apply maximum position limit
        position_size = min(position_size, self.config.MAX_POSITION_SIZE)
        
        return max(0, position_size)
    
    def _calculate_stops(self, symbol: str, prediction: pd.Series,
                        market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic stop loss and take profit levels"""
        symbol_data = market_data.xs(symbol, level=1, axis=1)
        current_price = symbol_data['Close'].iloc[-1]
        
        # Get ATR for dynamic stops
        high = symbol_data['High']
        low = symbol_data['Low']
        close = symbol_data['Close']
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Use quantile predictions if available
        if 'lgb_q10' in prediction and 'lgb_q90' in prediction:
            # Asymmetric stops based on predicted distribution
            if prediction['consensus_direction'] > 0.5:  # Long signal
                stop_loss = current_price - 2 * atr
                # Use 90th percentile for take profit
                take_profit = current_price * (1 + prediction['lgb_q90'])
            else:  # Short signal
                stop_loss = current_price + 2 * atr
                # Use 10th percentile for take profit
                take_profit = current_price * (1 + prediction['lgb_q10'])
        else:
            # Fallback to ATR-based stops
            multiplier = 2.5 if prediction.get('confidence_score', 0.5) > 0.7 else 2.0
            
            if prediction['consensus_direction'] > 0.5:  # Long
                stop_loss = current_price - multiplier * atr
                take_profit = current_price + (3 * multiplier * atr)
            else:  # Short
                stop_loss = current_price + multiplier * atr
                take_profit = current_price - (3 * multiplier * atr)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _calculate_risk_score(self, prediction: pd.Series, regime: Dict) -> float:
        """Calculate risk score for the signal"""
        risk_factors = []
        
        # Prediction uncertainty
        if 'consensus_agreement' in prediction:
            risk_factors.append(prediction['consensus_agreement'])
        
        # Volatility risk
        if 'predicted_volatility' in prediction:
            vol_percentile = stats.percentileofscore(
                [0.1, 0.15, 0.2, 0.25, 0.3, 0.4], 
                prediction['predicted_volatility']
            ) / 100
            risk_factors.append(vol_percentile)
        
        # Regime risk
        regime_risk = {
            'calm': 0.2,
            'normal': 0.4,
            'high': 0.7,
            'extreme': 1.0
        }
        risk_factors.append(regime_risk.get(regime['volatility_regime'], 0.5))
        
        # Quantile spread risk
        if 'xgb_q75' in prediction and 'xgb_q25' in prediction:
            iqr = prediction['xgb_q75'] - prediction['xgb_q25']
            spread_risk = min(1.0, iqr / 0.05)  # Normalize by 5% spread
            risk_factors.append(spread_risk)
        
        return np.mean(risk_factors)
    
    def _get_feature_snapshot(self, market_data: pd.DataFrame, symbol: str) -> Dict:
        """Get snapshot of key features for the signal"""
        try:
            symbol_data = market_data.xs(symbol, level=1, axis=1)
            features = self.feature_engineer.engineer_features(market_data, symbol)
            
            if features.empty:
                return {}
            
            latest_features = features.iloc[-1]
            
            # Key technical indicators
            snapshot = {
                'price': symbol_data['Close'].iloc[-1],
                'volume_ratio': latest_features.get('volume_ratio_20', 1.0),
                'rsi_14': latest_features.get('rsi_14', 50),
                'atr_pct': latest_features.get('atr_pct_14', 0.02),
                'bb_position': latest_features.get('bb_position_20_20', 0.5),
                'trend_strength': latest_features.get('adx_14', 25),
                'momentum': latest_features.get('return_5d', 0),
                'support_distance': latest_features.get('price_position_20d', 0.5)
            }
            
            return snapshot
        except:
            return {}
    
    def _apply_portfolio_filters(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply portfolio-level risk filters"""
        filtered_signals = []
        
        # Group by sector if available
        sector_exposure = {}
        
        for signal in signals:
            # Check sector exposure
            sector = self._get_symbol_sector(signal.symbol)
            current_exposure = sector_exposure.get(sector, 0)
            
            if current_exposure + signal.position_size > self.config.MAX_SECTOR_EXPOSURE:
                # Reduce position size to fit within limit
                signal.position_size = max(0, self.config.MAX_SECTOR_EXPOSURE - current_exposure)
            
            if signal.position_size > 0:
                filtered_signals.append(signal)
                sector_exposure[sector] = current_exposure + signal.position_size
        
        return filtered_signals
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        # This would connect to the watchlist manager
        # Simplified for this example
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN']
        financial_symbols = ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in financial_symbols:
            return 'Financials'
        else:
            return 'Other'
    
    def _rank_and_select_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals and select the best ones"""
        if not signals:
            return []
        
        # Score each signal
        for signal in signals:
            # Composite score combining multiple factors
            signal.composite_score = (
                abs(signal.bayesian_score) * 0.4 +
                signal.confidence_score * 0.3 +
                (1 - signal.risk_score) * 0.3
            )
        
        # Sort by composite score
        signals.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply portfolio heat limit
        total_risk = 0
        selected_signals = []
        
        for signal in signals:
            position_risk = signal.position_size * signal.predicted_volatility
            
            if total_risk + position_risk <= self.config.MAX_PORTFOLIO_HEAT:
                selected_signals.append(signal)
                total_risk += position_risk
        
        return selected_signals

# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegimeDetector:
    """Detects current market regime for adaptive trading"""
    
    def __init__(self, config):
        self.config = config
        
    def detect_regime(self, market_data: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        regime = {}
        
        # Get market indices
        spy_data = self._get_index_data(market_data, 'SPY')
        vix_data = self._get_index_data(market_data, 'VIX')
        
        # Volatility regime
        if vix_data is not None:
            current_vix = vix_data['Close'].iloc[-1]
            regime['vix_level'] = current_vix
            
            if current_vix < 15:
                regime['volatility_regime'] = 'calm'
            elif current_vix < 20:
                regime['volatility_regime'] = 'normal'
            elif current_vix < 30:
                regime['volatility_regime'] = 'high'
            else:
                regime['volatility_regime'] = 'extreme'
        else:
            # Fallback to SPY volatility
            if spy_data is not None:
                spy_returns = spy_data['Close'].pct_change()
                realized_vol = spy_returns.tail(20).std() * np.sqrt(252)
                
                if realized_vol < 0.12:
                    regime['volatility_regime'] = 'calm'
                elif realized_vol < 0.18:
                    regime['volatility_regime'] = 'normal'
                elif realized_vol < 0.25:
                    regime['volatility_regime'] = 'high'
                else:
                    regime['volatility_regime'] = 'extreme'
            else:
                regime['volatility_regime'] = 'normal'
        
        # Trend regime
        if spy_data is not None:
            sma_50 = spy_data['Close'].rolling(50).mean().iloc[-1]
            sma_200 = spy_data['Close'].rolling(200).mean().iloc[-1]
            current_price = spy_data['Close'].iloc[-1]
            
            # Calculate trend strength
            trend_score = 0
            if current_price > sma_50:
                trend_score += 1
            if current_price > sma_200:
                trend_score += 1
            if sma_50 > sma_200:
                trend_score += 2
            
            if trend_score >= 3:
                regime['trend_regime'] = 'strong_bull'
            elif trend_score >= 2:
                regime['trend_regime'] = 'bull'
            elif trend_score >= 1:
                regime['trend_regime'] = 'neutral'
            else:
                regime['trend_regime'] = 'bear'
                
            # Check for strong bear
            if current_price < sma_200 * 0.9:
                regime['trend_regime'] = 'strong_bear'
        else:
            regime['trend_regime'] = 'neutral'
        
        # Liquidity regime
        if spy_data is not None:
            volume_ma = spy_data['Volume'].rolling(20).mean()
            recent_volume = spy_data['Volume'].tail(5).mean()
            volume_ratio = recent_volume / volume_ma.iloc[-1]
            
            if volume_ratio < 0.7:
                regime['liquidity_regime'] = 'low'
            elif volume_ratio > 1.3:
                regime['liquidity_regime'] = 'high'
            else:
                regime['liquidity_regime'] = 'normal'
        else:
            regime['liquidity_regime'] = 'normal'
        
        # Breadth regime (simplified)
        advances = 0
        declines = 0
        
        for symbol in market_data.columns.levels[1][:20]:  # Check first 20 symbols
            try:
                symbol_close = market_data.xs(symbol, level=1, axis=1)['Close']
                if symbol_close.iloc[-1] > symbol_close.iloc[-2]:
                    advances += 1
                else:
                    declines += 1
            except:
                continue
        
        advance_ratio = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        
        if advance_ratio > 0.7:
            regime['breadth_regime'] = 'strong'
        elif advance_ratio > 0.3:
            regime['breadth_regime'] = 'normal'
        else:
            regime['breadth_regime'] = 'weak'
        
        return regime
    
    def _get_index_data(self, market_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Get data for an index symbol"""
        try:
            return market_data.xs(symbol, level=1, axis=1)
        except:
            return None

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.risk_limits = RiskLimits(config)
        self.position_tracker = PositionTracker()
        
    def check_pre_trade_risk(self, signal: TradingSignal, 
                           current_positions: Dict) -> Tuple[bool, str]:
        """Check if a trade passes pre-trade risk checks"""
        # Check position limits
        if not self.risk_limits.check_position_limit(signal, current_positions):
            return False, "Position limit exceeded"
        
        # Check sector limits
        if not self.risk_limits.check_sector_limit(signal, current_positions):
            return False, "Sector exposure limit exceeded"
        
        # Check correlation limits
        if not self.risk_limits.check_correlation_limit(signal, current_positions):
            return False, "Correlation limit exceeded"
        
        # Check portfolio heat
        if not self.risk_limits.check_portfolio_heat(signal, current_positions):
            return False, "Portfolio heat limit exceeded"
        
        # Check daily loss limit
        if not self.risk_limits.check_daily_loss_limit(current_positions):
            return False, "Daily loss limit reached"
        
        return True, "Passed all risk checks"
    
    def calculate_position_risk_metrics(self, position: Dict) -> Dict:
        """Calculate risk metrics for a position"""
        metrics = {}
        
        # Value at Risk (VaR)
        position_value = position['quantity'] * position['current_price']
        daily_vol = position['volatility'] / np.sqrt(252)
        
        # 95% VaR
        metrics['var_95'] = position_value * daily_vol * 1.645
        
        # 99% VaR
        metrics['var_99'] = position_value * daily_vol * 2.326
        
        # Expected Shortfall (CVaR)
        metrics['cvar_95'] = position_value * daily_vol * 2.063
        
        # Maximum loss (to stop loss)
        if position['direction'] == 'long':
            max_loss = (position['entry_price'] - position['stop_loss']) * position['quantity']
        else:
            max_loss = (position['stop_loss'] - position['entry_price']) * position['quantity']
        
        metrics['max_loss'] = max_loss
        
        # Risk/Reward ratio
        if position['direction'] == 'long':
            potential_profit = (position['take_profit'] - position['entry_price']) * position['quantity']
        else:
            potential_profit = (position['entry_price'] - position['take_profit']) * position['quantity']
        
        metrics['risk_reward_ratio'] = potential_profit / max_loss if max_loss > 0 else 0
        
        return metrics
    
    def update_stop_losses(self, positions: Dict, market_data: pd.DataFrame) -> Dict:
        """Update stop losses based on market conditions"""
        updated_positions = {}
        
        for symbol, position in positions.items():
            try:
                symbol_data = market_data.xs(symbol, level=1, axis=1)
                current_price = symbol_data['Close'].iloc[-1]
                
                # Calculate trailing stop
                if position['direction'] == 'long':
                    # For profitable longs, trail the stop
                    if current_price > position['entry_price']:
                        # ATR-based trailing stop
                        atr = self._calculate_atr(symbol_data)
                        new_stop = current_price - 2 * atr
                        
                        # Only move stop up, never down
                        if new_stop > position['stop_loss']:
                            position['stop_loss'] = new_stop
                            position['stop_type'] = 'trailing'
                else:  # Short position
                    if current_price < position['entry_price']:
                        atr = self._calculate_atr(symbol_data)
                        new_stop = current_price + 2 * atr
                        
                        # Only move stop down for shorts
                        if new_stop < position['stop_loss']:
                            position['stop_loss'] = new_stop
                            position['stop_type'] = 'trailing'
                
                updated_positions[symbol] = position
                
            except Exception as e:
                logger.error(f"Error updating stop for {symbol}: {e}")
                updated_positions[symbol] = position
        
        return updated_positions
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for stop loss adjustment"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]

class RiskLimits:
    """Enforces risk limits"""
    
    def __init__(self, config):
        self.config = config
        
    def check_position_limit(self, signal: TradingSignal, 
                           current_positions: Dict) -> bool:
        """Check if position size is within limits"""
        # Single position limit
        if signal.position_size > self.config.MAX_POSITION_SIZE:
            return False
        
        # Total exposure to symbol
        if signal.symbol in current_positions:
            total_exposure = current_positions[signal.symbol]['position_size'] + signal.position_size
            if total_exposure > self.config.MAX_POSITION_SIZE * 1.5:
                return False
        
        return True
    
    def check_sector_limit(self, signal: TradingSignal, 
                          current_positions: Dict) -> bool:
        """Check sector exposure limits"""
        sector = self._get_symbol_sector(signal.symbol)
        
        current_sector_exposure = 0
        for symbol, position in current_positions.items():
            if self._get_symbol_sector(symbol) == sector:
                current_sector_exposure += position['position_size']
        
        if current_sector_exposure + signal.position_size > self.config.MAX_SECTOR_EXPOSURE:
            return False
        
        return True
    
    def check_correlation_limit(self, signal: TradingSignal, 
                              current_positions: Dict) -> bool:
        """Check correlation limits to avoid concentrated bets"""
        # This would calculate correlations between positions
        # Simplified for this example
        
        high_correlation_threshold = 0.7
        max_correlated_positions = 3
        
        # Count highly correlated positions
        correlated_count = 0
        for symbol in current_positions:
            # Would calculate actual correlation here
            if self._is_highly_correlated(signal.symbol, symbol):
                correlated_count += 1
        
        return correlated_count < max_correlated_positions
    
    def check_portfolio_heat(self, signal: TradingSignal, 
                           current_positions: Dict) -> bool:
        """Check total portfolio risk (heat)"""
        current_heat = 0
        
        for symbol, position in current_positions.items():
            # Risk = position size * volatility
            position_risk = position['position_size'] * position.get('volatility', 0.02)
            current_heat += position_risk
        
        # Add new position risk
        new_position_risk = signal.position_size * signal.predicted_volatility
        total_heat = current_heat + new_position_risk
        
        return total_heat <= self.config.MAX_PORTFOLIO_HEAT
    
    def check_daily_loss_limit(self, current_positions: Dict) -> bool:
        """Check if daily loss limit has been reached"""
        daily_pnl = 0
        
        for symbol, position in current_positions.items():
            if 'daily_pnl' in position:
                daily_pnl += position['daily_pnl']
        
        # 2% daily loss limit
        max_daily_loss = -0.02
        
        return daily_pnl > max_daily_loss
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        # Simplified - would use actual sector mapping
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
        if symbol in tech_symbols:
            return 'Technology'
        return 'Other'
    
    def _is_highly_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated"""
        # Simplified - would calculate actual correlation
        same_sector_pairs = [
            ('AAPL', 'MSFT'), ('GOOGL', 'META'), ('JPM', 'BAC')
        ]
        
        return (symbol1, symbol2) in same_sector_pairs or (symbol2, symbol1) in same_sector_pairs

class PositionTracker:
    """Tracks all positions and their metrics"""
    
    def __init__(self):
        self.positions = {}
        self.position_history = []
        
    def add_position(self, signal: TradingSignal, fill_price: float):
        """Add a new position"""
        position = {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'entry_price': fill_price,
            'entry_time': datetime.now(),
            'quantity': self._calculate_shares(signal.position_size, fill_price),
            'position_size': signal.position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'volatility': signal.predicted_volatility,
            'initial_signal': signal
        }
        
        self.positions[signal.symbol] = position
        self.position_history.append(position.copy())
        
    def update_position(self, symbol: str, current_price: float, 
                       current_data: Dict = None):
        """Update position with current market data"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Update current price
        position['current_price'] = current_price
        
        # Calculate P&L
        if position['direction'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            position['unrealized_pnl_pct'] = (current_price / position['entry_price'] - 1) * 100
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
            position['unrealized_pnl_pct'] = (position['entry_price'] / current_price - 1) * 100
        
        # Update other metrics if data provided
        if current_data:
            position['current_volatility'] = current_data.get('volatility', position['volatility'])
            position['days_held'] = (datetime.now() - position['entry_time']).days
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'signal'):
        """Close a position and record results"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        if position['direction'] == 'long':
            position['realized_pnl'] = (exit_price - position['entry_price']) * position['quantity']
            position['realized_pnl_pct'] = (exit_price / position['entry_price'] - 1) * 100
        else:
            position['realized_pnl'] = (position['entry_price'] - exit_price) * position['quantity']
            position['realized_pnl_pct'] = (position['entry_price'] / exit_price - 1) * 100
        
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = reason
        position['duration'] = position['exit_time'] - position['entry_time']
        
        # Add to history
        self.position_history.append(position)
        
        # Remove from active positions
        del self.positions[symbol]
        
        return position
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate current portfolio metrics"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_exposure': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0
            }
        
        metrics = {
            'total_positions': len(self.positions),
            'total_exposure': sum(p['position_size'] for p in self.positions.values()),
            'total_pnl': sum(p.get('unrealized_pnl', 0) for p in self.positions.values()),
            'long_positions': sum(1 for p in self.positions.values() if p['direction'] == 'long'),
            'short_positions': sum(1 for p in self.positions.values() if p['direction'] == 'short'),
            'avg_days_held': np.mean([p.get('days_held', 0) for p in self.positions.values()]),
            'winning_positions': sum(1 for p in self.positions.values() if p.get('unrealized_pnl', 0) > 0),
            'losing_positions': sum(1 for p in self.positions.values() if p.get('unrealized_pnl', 0) < 0)
        }
        
        # Calculate total P&L percentage
        total_value = sum(p['entry_price'] * p['quantity'] for p in self.positions.values())
        if total_value > 0:
            metrics['total_pnl_pct'] = (metrics['total_pnl'] / total_value) * 100
        else:
            metrics['total_pnl_pct'] = 0
        
        return metrics
    
    def _calculate_shares(self, position_size: float, price: float, 
                         portfolio_value: float = 1000000) -> int:
        """Calculate number of shares to buy"""
        position_value = portfolio_value * position_size
        shares = int(position_value / price)
        return max(1, shares)  # At least 1 share

# Example usage
if __name__ == "__main__":
    logger.info("Signal Generation & Risk Management module loaded successfully")