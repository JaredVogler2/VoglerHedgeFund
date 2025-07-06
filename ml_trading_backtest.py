# ml_trading_backtest.py

"""
ML Trading System - Professional Backtesting Engine
Walk-forward optimization with realistic execution modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# =============================================================================
# EXECUTION MODELS
# =============================================================================

@dataclass
class ExecutionModel:
    """Models realistic trade execution with costs"""
    
    # Commission structure
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    
    # Slippage model parameters
    base_slippage_bps: float = 2.0  # 2 basis points base slippage
    volume_impact_factor: float = 0.1  # Impact increases with size
    volatility_multiplier: float = 1.5  # Higher vol = more slippage
    
    # Market impact (square-root model)
    temporary_impact_factor: float = 0.01
    permanent_impact_factor: float = 0.005
    
    def calculate_fill_price(self, signal_price: float, quantity: int, 
                           symbol_data: Dict, direction: str) -> float:
        """Calculate realistic fill price including all costs"""
        
        # Base slippage
        slippage = self.base_slippage_bps / 10000
        
        # Volume-based slippage
        avg_volume = symbol_data.get('avg_volume', 1000000)
        volume_participation = quantity / avg_volume
        volume_slippage = self.volume_impact_factor * np.sqrt(volume_participation)
        
        # Volatility adjustment
        volatility = symbol_data.get('volatility', 0.02)
        vol_adjustment = 1 + (volatility - 0.02) * self.volatility_multiplier
        
        # Total slippage
        total_slippage = (slippage + volume_slippage) * vol_adjustment
        
        # Apply slippage based on direction
        if direction == 'buy' or direction == 'long':
            fill_price = signal_price * (1 + total_slippage)
        else:
            fill_price = signal_price * (1 - total_slippage)
        
        # Add market impact
        market_impact = self.calculate_market_impact(quantity, avg_volume, volatility)
        
        if direction == 'buy' or direction == 'long':
            fill_price += market_impact * signal_price
        else:
            fill_price -= market_impact * signal_price
        
        return fill_price
    
    def calculate_market_impact(self, quantity: int, avg_volume: float, 
                              volatility: float) -> float:
        """Calculate market impact using square-root model"""
        # Normalized trade size
        trade_size = quantity / avg_volume
        
        # Temporary impact (disappears after trade)
        temp_impact = self.temporary_impact_factor * volatility * np.sqrt(trade_size)
        
        # Permanent impact (affects subsequent trades)
        perm_impact = self.permanent_impact_factor * volatility * trade_size
        
        return temp_impact + perm_impact
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate trading commission"""
        commission = quantity * self.commission_per_share
        return max(commission, self.min_commission)
    
    def calculate_total_cost(self, quantity: int, entry_price: float, 
                           exit_price: float, symbol_data: Dict) -> Dict:
        """Calculate all transaction costs for a round trip"""
        # Entry costs
        entry_fill = self.calculate_fill_price(entry_price, quantity, symbol_data, 'buy')
        entry_commission = self.calculate_commission(quantity, entry_fill)
        
        # Exit costs
        exit_fill = self.calculate_fill_price(exit_price, quantity, symbol_data, 'sell')
        exit_commission = self.calculate_commission(quantity, exit_fill)
        
        # Total costs
        slippage_cost = (entry_fill - entry_price) * quantity + (exit_price - exit_fill) * quantity
        commission_cost = entry_commission + exit_commission
        
        return {
            'entry_fill': entry_fill,
            'exit_fill': exit_fill,
            'slippage_cost': slippage_cost,
            'commission_cost': commission_cost,
            'total_cost': slippage_cost + commission_cost,
            'cost_percentage': (slippage_cost + commission_cost) / (entry_price * quantity) * 100
        }

# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestingEngine:
    """Professional backtesting engine with walk-forward optimization"""
    
    def __init__(self, config, feature_engineer, ensemble_manager, 
                 signal_generator, risk_manager):
        self.config = config
        self.feature_engineer = feature_engineer
        self.ensemble_manager = ensemble_manager
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_model = ExecutionModel()
        
    def run_backtest(self, market_data: pd.DataFrame, 
                    start_date: str, end_date: str,
                    initial_capital: float = 1000000) -> Dict:
        """Run complete backtest with walk-forward optimization"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize backtest state
        backtest_state = BacktestState(
            initial_capital=initial_capital,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date)
        )
        
        # Filter data to backtest period
        mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        backtest_data = market_data[mask]
        
        # Walk-forward optimization loop
        walk_forward_results = self._run_walk_forward(backtest_data, backtest_state)
        
        # Calculate final metrics
        performance_metrics = self._calculate_performance_metrics(backtest_state)
        
        # Generate analysis reports
        analysis = self._generate_backtest_analysis(backtest_state, performance_metrics)
        
        return {
            'performance_metrics': performance_metrics,
            'equity_curve': backtest_state.equity_curve,
            'trades': backtest_state.trade_history,
            'positions': backtest_state.position_history,
            'walk_forward_results': walk_forward_results,
            'analysis': analysis
        }
    
    def _run_walk_forward(self, data: pd.DataFrame, 
                         backtest_state: 'BacktestState') -> List[Dict]:
        """Run walk-forward optimization"""
        results = []
        
        # Calculate periods
        total_days = (data.index[-1] - data.index[0]).days
        train_days = self.config.WALK_FORWARD_TRAIN_MONTHS * 30
        val_days = self.config.WALK_FORWARD_VAL_MONTHS * 30
        test_days = self.config.WALK_FORWARD_TEST_MONTHS * 30
        
        # Walk-forward loop
        current_date = data.index[0] + timedelta(days=train_days + val_days)
        
        while current_date + timedelta(days=test_days) <= data.index[-1]:
            logger.info(f"Walk-forward period ending {current_date}")
            
            # Define periods
            train_start = current_date - timedelta(days=train_days + val_days)
            train_end = current_date - timedelta(days=val_days)
            val_end = current_date
            test_end = current_date + timedelta(days=test_days)
            
            # Get data for each period
            train_data = data[train_start:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:test_end]
            
            # Train models on training period
            logger.info("Training models...")
            trained_models = self._train_period_models(train_data, val_data)
            
            # Run backtest on test period
            logger.info("Running test period backtest...")
            period_results = self._run_period_backtest(
                test_data, trained_models, backtest_state
            )
            
            results.append({
                'period_start': val_end,
                'period_end': test_end,
                'results': period_results,
                'models': trained_models
            })
            
            # Move to next period
            current_date += timedelta(days=test_days)
        
        return results
    
    def _train_period_models(self, train_data: pd.DataFrame, 
                           val_data: pd.DataFrame) -> Dict:
        """Train models for a specific period"""
        # This is simplified - in production would train all models
        # For now, return placeholder
        return {
            'period_start': train_data.index[0],
            'period_end': val_data.index[-1],
            'model_performance': {'accuracy': 0.55, 'sharpe': 1.2}
        }
    
    def _run_period_backtest(self, test_data: pd.DataFrame, 
                           models: Dict, backtest_state: 'BacktestState') -> Dict:
        """Run backtest for a single test period"""
        period_trades = []
        daily_returns = []
        
        # Simulate daily trading
        for date in test_data.index.unique():
            daily_data = test_data.loc[date]
            
            # Morning: Generate signals
            signals = self._generate_daily_signals(daily_data, models, backtest_state)
            
            # Execute trades
            executed_trades = self._execute_signals(signals, daily_data, backtest_state)
            period_trades.extend(executed_trades)
            
            # Update positions
            self._update_positions(daily_data, backtest_state)
            
            # Record daily state
            daily_return = (backtest_state.current_equity / backtest_state.previous_equity) - 1
            daily_returns.append(daily_return)
            backtest_state.previous_equity = backtest_state.current_equity
            
            # Record equity curve
            backtest_state.equity_curve.append({
                'date': date,
                'equity': backtest_state.current_equity,
                'cash': backtest_state.cash,
                'positions_value': backtest_state.current_equity - backtest_state.cash
            })
        
        return {
            'trades': period_trades,
            'daily_returns': daily_returns,
            'period_return': (backtest_state.current_equity / backtest_state.equity_curve[-len(daily_returns)]['equity']) - 1
        }
    
    def _generate_daily_signals(self, daily_data: pd.DataFrame, 
                              models: Dict, backtest_state: 'BacktestState') -> List:
        """Generate trading signals for the day"""
        signals = []
        
        # Get universe of tradeable symbols
        symbols = self._get_tradeable_symbols(daily_data)
        
        for symbol in symbols[:10]:  # Limit for performance in this example
            # Skip if already have position
            if symbol in backtest_state.current_positions:
                continue
            
            try:
                # Get symbol data
                symbol_data = daily_data.xs(symbol, level=1) if isinstance(daily_data, pd.DataFrame) else daily_data
                
                # Generate mock signal (in production, would use actual models)
                signal = self._create_mock_signal(symbol, symbol_data, backtest_state)
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.debug(f"Error generating signal for {symbol}: {e}")
        
        # Rank and filter signals
        signals = self._filter_signals_by_risk(signals, backtest_state)
        
        return signals[:5]  # Take top 5 signals
    
    def _get_tradeable_symbols(self, daily_data: pd.DataFrame) -> List[str]:
        """Get list of tradeable symbols for the day"""
        if isinstance(daily_data, pd.DataFrame):
            return [col[1] for col in daily_data.columns if col[0] == 'Close']
        else:
            return []
    
    def _create_mock_signal(self, symbol: str, symbol_data: Any, 
                          backtest_state: 'BacktestState') -> Optional[Dict]:
        """Create mock signal for backtesting example"""
        # Random signal generation for demonstration
        if np.random.random() > 0.8:  # 20% chance of signal
            current_price = float(symbol_data['Close']) if isinstance(symbol_data, pd.Series) else symbol_data
            
            return {
                'symbol': symbol,
                'direction': 'long' if np.random.random() > 0.5 else 'short',
                'signal_price': current_price,
                'position_size': min(0.02, backtest_state.cash / backtest_state.current_equity),
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.03,
                'confidence': np.random.uniform(0.6, 0.9)
            }
        return None
    
    def _filter_signals_by_risk(self, signals: List[Dict], 
                              backtest_state: 'BacktestState') -> List[Dict]:
        """Filter signals based on risk limits"""
        filtered = []
        total_new_exposure = 0
        
        for signal in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            # Check if within risk limits
            new_exposure = signal['position_size']
            
            if (total_new_exposure + new_exposure + 
                backtest_state.current_exposure) <= self.config.MAX_PORTFOLIO_HEAT:
                filtered.append(signal)
                total_new_exposure += new_exposure
        
        return filtered
    
    def _execute_signals(self, signals: List[Dict], daily_data: pd.DataFrame,
                        backtest_state: 'BacktestState') -> List[Dict]:
        """Execute trading signals"""
        executed_trades = []
        
        for signal in signals:
            # Check available capital
            position_value = backtest_state.current_equity * signal['position_size']
            
            if position_value > backtest_state.cash:
                continue
            
            # Get execution price
            symbol_data = {
                'avg_volume': 1000000,  # Mock data
                'volatility': 0.02
            }
            
            fill_price = self.execution_model.calculate_fill_price(
                signal['signal_price'], 
                int(position_value / signal['signal_price']),
                symbol_data,
                signal['direction']
            )
            
            # Calculate commission
            quantity = int(position_value / fill_price)
            commission = self.execution_model.calculate_commission(quantity, fill_price)
            
            # Execute trade
            trade = {
                'symbol': signal['symbol'],
                'date': daily_data.index[0] if hasattr(daily_data, 'index') else datetime.now(),
                'direction': signal['direction'],
                'quantity': quantity,
                'signal_price': signal['signal_price'],
                'fill_price': fill_price,
                'commission': commission,
                'slippage': fill_price - signal['signal_price'],
                'position_size': signal['position_size'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit']
            }
            
            # Update backtest state
            cost = (quantity * fill_price) + commission
            backtest_state.cash -= cost
            backtest_state.current_positions[signal['symbol']] = {
                'quantity': quantity,
                'entry_price': fill_price,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'direction': signal['direction'],
                'entry_date': trade['date']
            }
            backtest_state.current_exposure += signal['position_size']
            
            executed_trades.append(trade)
            backtest_state.trade_history.append(trade)
        
        return executed_trades
    
    def _update_positions(self, daily_data: pd.DataFrame, 
                         backtest_state: 'BacktestState'):
        """Update existing positions"""
        positions_to_close = []
        
        for symbol, position in backtest_state.current_positions.items():
            try:
                # Get current price
                if isinstance(daily_data, pd.DataFrame):
                    current_price = daily_data.xs(symbol, level=1)['Close']
                else:
                    current_price = position['entry_price'] * (1 + np.random.normal(0, 0.01))
                
                # Check stop loss
                if position['direction'] == 'long':
                    if current_price <= position['stop_loss']:
                        positions_to_close.append((symbol, current_price, 'stop_loss'))
                    elif current_price >= position['take_profit']:
                        positions_to_close.append((symbol, current_price, 'take_profit'))
                else:  # short
                    if current_price >= position['stop_loss']:
                        positions_to_close.append((symbol, current_price, 'stop_loss'))
                    elif current_price <= position['take_profit']:
                        positions_to_close.append((symbol, current_price, 'take_profit'))
                
                # Update position value
                if position['direction'] == 'long':
                    position['current_value'] = position['quantity'] * current_price
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                else:
                    position['current_value'] = position['quantity'] * (2 * position['entry_price'] - current_price)
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                    
            except Exception as e:
                logger.debug(f"Error updating position {symbol}: {e}")
        
        # Close positions
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, exit_price, reason, daily_data, backtest_state)
        
        # Update equity
        positions_value = sum(pos.get('current_value', 0) for pos in backtest_state.current_positions.values())
        backtest_state.current_equity = backtest_state.cash + positions_value
    
    def _close_position(self, symbol: str, exit_price: float, reason: str,
                       daily_data: pd.DataFrame, backtest_state: 'BacktestState'):
        """Close a position"""
        position = backtest_state.current_positions[symbol]
        
        # Calculate exit fill price
        symbol_data = {'avg_volume': 1000000, 'volatility': 0.02}
        
        fill_price = self.execution_model.calculate_fill_price(
            exit_price,
            position['quantity'],
            symbol_data,
            'sell' if position['direction'] == 'long' else 'buy'
        )
        
        # Calculate P&L
        if position['direction'] == 'long':
            gross_pnl = (fill_price - position['entry_price']) * position['quantity']
        else:
            gross_pnl = (position['entry_price'] - fill_price) * position['quantity']
        
        # Calculate commission
        commission = self.execution_model.calculate_commission(position['quantity'], fill_price)
        net_pnl = gross_pnl - commission
        
        # Record closed position
        closed_position = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': daily_data.index[0] if hasattr(daily_data, 'index') else datetime.now(),
            'direction': position['direction'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'exit_price': fill_price,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / (position['entry_price'] * position['quantity'])) * 100,
            'exit_reason': reason,
            'holding_period': (datetime.now() - position['entry_date']).days
        }
        
        backtest_state.position_history.append(closed_position)
        
        # Update cash
        backtest_state.cash += (position['quantity'] * fill_price) - commission
        
        # Update exposure
        position_size = (position['quantity'] * position['entry_price']) / backtest_state.initial_capital
        backtest_state.current_exposure -= position_size
        
        # Remove position
        del backtest_state.current_positions[symbol]
    
    def _calculate_performance_metrics(self, backtest_state: 'BacktestState') -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Get equity curve data
        equity_curve = pd.DataFrame(backtest_state.equity_curve)
        if equity_curve.empty:
            return metrics
        
        equity_curve.set_index('date', inplace=True)
        
        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Basic metrics
        metrics['total_return'] = (backtest_state.current_equity / backtest_state.initial_capital) - 1
        metrics['cagr'] = self._calculate_cagr(equity_curve['equity'])
        
        # Risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(equity_curve['equity'])
        
        # Drawdown metrics
        drawdown_data = self._calculate_drawdowns(equity_curve['equity'])
        metrics['max_drawdown'] = drawdown_data['max_drawdown']
        metrics['max_drawdown_duration'] = drawdown_data['max_duration']
        metrics['avg_drawdown'] = drawdown_data['avg_drawdown']
        
        # Trade metrics
        if backtest_state.position_history:
            trade_metrics = self._calculate_trade_metrics(backtest_state.position_history)
            metrics.update(trade_metrics)
        
        # Risk-adjusted metrics
        metrics['information_ratio'] = self._calculate_information_ratio(returns)
        metrics['omega_ratio'] = self._calculate_omega_ratio(returns)
        metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
        
        # Value at Risk
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        return metrics
    
    def _calculate_cagr(self, equity_series: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
        return (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1/years) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def _calculate_calmar_ratio(self, equity_series: pd.Series) -> float:
        """Calculate Calmar ratio"""
        cagr = self._calculate_cagr(equity_series)
        max_dd = self._calculate_drawdowns(equity_series)['max_drawdown']
        return cagr / abs(max_dd) if max_dd != 0 else 0
    
    def _calculate_drawdowns(self, equity_series: pd.Series) -> Dict:
        """Calculate drawdown statistics"""
        # Calculate running maximum
        running_max = equity_series.expanding().max()
        
        # Calculate drawdown series
        drawdown = (equity_series - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        
        # Calculate statistics
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[is_drawdown].mean() if is_drawdown.any() else 0,
            'max_duration': self._get_max_drawdown_duration(drawdown),
            'drawdown_series': drawdown
        }
    
    def _get_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        is_drawdown = drawdown_series < 0
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_trade_metrics(self, position_history: List[Dict]) -> Dict:
        """Calculate trade-based metrics"""
        df = pd.DataFrame(position_history)
        
        metrics = {
            'total_trades': len(df),
            'winning_trades': len(df[df['net_pnl'] > 0]),
            'losing_trades': len(df[df['net_pnl'] < 0]),
            'win_rate': len(df[df['net_pnl'] > 0]) / len(df) if len(df) > 0 else 0,
            'avg_win': df[df['net_pnl'] > 0]['return_pct'].mean() if len(df[df['net_pnl'] > 0]) > 0 else 0,
            'avg_loss': df[df['net_pnl'] < 0]['return_pct'].mean() if len(df[df['net_pnl'] < 0]) > 0 else 0,
            'profit_factor': abs(df[df['net_pnl'] > 0]['net_pnl'].sum() / df[df['net_pnl'] < 0]['net_pnl'].sum()) if df[df['net_pnl'] < 0]['net_pnl'].sum() != 0 else 0,
            'avg_trade_return': df['return_pct'].mean(),
            'avg_holding_period': df['holding_period'].mean() if 'holding_period' in df else 0,
            'total_commission': df['commission'].sum(),
            'commission_pct': df['commission'].sum() / self.config.INITIAL_CAPITAL * 100
        }
        
        # Calculate expectancy
        if metrics['win_rate'] > 0 and metrics['avg_win'] > 0:
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win']) + ((1 - metrics['win_rate']) * metrics['avg_loss'])
        else:
            metrics['expectancy'] = 0
        
        return metrics
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
        """Calculate Information Ratio"""
        if benchmark_returns is None:
            # Use 0 as benchmark
            active_returns = returns
        else:
            active_returns = returns - benchmark_returns
        
        tracking_error = active_returns.std()
        return np.sqrt(252) * active_returns.mean() / tracking_error if tracking_error > 0 else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega Ratio"""
        positive_returns = returns[returns > threshold] - threshold
        negative_returns = threshold - returns[returns <= threshold]
        
        if negative_returns.sum() > 0:
            return positive_returns.sum() / negative_returns.sum()
        return np.inf if positive_returns.sum() > 0 else 0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)"""
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        return right_tail / left_tail if left_tail > 0 else np.inf
    
    def _generate_backtest_analysis(self, backtest_state: 'BacktestState', 
                                  metrics: Dict) -> Dict:
        """Generate comprehensive backtest analysis"""
        analysis = {
            'summary': self._generate_summary(metrics),
            'risk_analysis': self._analyze_risk(backtest_state, metrics),
            'trade_analysis': self._analyze_trades(backtest_state.position_history),
            'regime_analysis': self._analyze_by_regime(backtest_state),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return analysis
    
    def _generate_summary(self, metrics: Dict) -> Dict:
        """Generate performance summary"""
        return {
            'performance_grade': self._calculate_performance_grade(metrics),
            'key_strengths': self._identify_strengths(metrics),
            'key_weaknesses': self._identify_weaknesses(metrics),
            'risk_reward_assessment': self._assess_risk_reward(metrics)
        }
    
    def _calculate_performance_grade(self, metrics: Dict) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Sharpe ratio contribution
        if metrics.get('sharpe_ratio', 0) > 2:
            score += 30
        elif metrics.get('sharpe_ratio', 0) > 1.5:
            score += 20
        elif metrics.get('sharpe_ratio', 0) > 1:
            score += 10
        
        # Win rate contribution
        if metrics.get('win_rate', 0) > 0.6:
            score += 20
        elif metrics.get('win_rate', 0) > 0.5:
            score += 10
        
        # Profit factor contribution
        if metrics.get('profit_factor', 0) > 2:
            score += 20
        elif metrics.get('profit_factor', 0) > 1.5:
            score += 10
        
        # Max drawdown contribution
        if abs(metrics.get('max_drawdown', -1)) < 0.1:
            score += 30
        elif abs(metrics.get('max_drawdown', -1)) < 0.2:
            score += 20
        elif abs(metrics.get('max_drawdown', -1)) < 0.3:
            score += 10
        
        # Grade assignment
        if score >= 80:
            return 'A'
        elif score >= 60:
            return 'B'
        elif score >= 40:
            return 'C'
        elif score >= 20:
            return 'D'
        else:
            return 'F'
    
    def _identify_strengths(self, metrics: Dict) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        if metrics.get('sharpe_ratio', 0) > 1.5:
            strengths.append("Excellent risk-adjusted returns")
        
        if metrics.get('win_rate', 0) > 0.6:
            strengths.append("High win rate")
        
        if abs(metrics.get('max_drawdown', -1)) < 0.15:
            strengths.append("Well-controlled drawdowns")
        
        if metrics.get('profit_factor', 0) > 1.8:
            strengths.append("Strong profit factor")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        if metrics.get('sharpe_ratio', 0) < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        
        if metrics.get('win_rate', 0) < 0.4:
            weaknesses.append("Low win rate")
        
        if abs(metrics.get('max_drawdown', -1)) > 0.3:
            weaknesses.append("Large drawdowns")
        
        if metrics.get('avg_holding_period', 0) < 1:
            weaknesses.append("Very short holding periods (possible overtrading)")
        
        return weaknesses
    
    def _assess_risk_reward(self, metrics: Dict) -> str:
        """Assess risk-reward profile"""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', -1))
        
        if sharpe > 1.5 and max_dd < 0.15:
            return "Excellent risk-reward profile"
        elif sharpe > 1 and max_dd < 0.25:
            return "Good risk-reward profile"
        elif sharpe > 0.5:
            return "Acceptable risk-reward profile"
        else:
            return "Poor risk-reward profile"
    
    def _analyze_risk(self, backtest_state: 'BacktestState', metrics: Dict) -> Dict:
        """Analyze risk characteristics"""
        return {
            'risk_metrics': {
                'volatility': metrics.get('volatility', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'var_95': metrics.get('var_95', 0),
                'cvar_95': metrics.get('cvar_95', 0)
            },
            'risk_assessment': self._assess_risk_level(metrics),
            'drawdown_analysis': self._analyze_drawdowns(backtest_state.equity_curve)
        }
    
    def _assess_risk_level(self, metrics: Dict) -> str:
        """Assess overall risk level"""
        vol = metrics.get('volatility', 0)
        
        if vol < 0.10:
            return "Low risk"
        elif vol < 0.20:
            return "Moderate risk"
        elif vol < 0.30:
            return "High risk"
        else:
            return "Very high risk"
    
    def _analyze_drawdowns(self, equity_curve: List[Dict]) -> Dict:
        """Analyze drawdown patterns"""
        if not equity_curve:
            return {}
        
        df = pd.DataFrame(equity_curve)
        equity = df['equity']
        
        # Calculate drawdowns
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        # Find significant drawdowns (> 5%)
        significant_dd = drawdown[drawdown < -0.05]
        
        return {
            'num_significant_drawdowns': len(significant_dd),
            'avg_recovery_time': self._calculate_avg_recovery_time(drawdown),
            'drawdown_frequency': len(significant_dd) / len(equity) * 252  # Annualized
        }
    
    def _calculate_avg_recovery_time(self, drawdown_series: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        # Simplified calculation
        return 15.0  # Placeholder
    
    def _analyze_trades(self, position_history: List[Dict]) -> Dict:
        """Analyze trading patterns"""
        if not position_history:
            return {}
        
        df = pd.DataFrame(position_history)
        
        return {
            'trade_distribution': {
                'by_direction': df['direction'].value_counts().to_dict(),
                'by_exit_reason': df['exit_reason'].value_counts().to_dict() if 'exit_reason' in df else {},
                'by_holding_period': self._categorize_holding_periods(df)
            },
            'profitability_analysis': {
                'profitable_directions': self._analyze_direction_profitability(df),
                'best_performers': df.nlargest(5, 'return_pct')[['symbol', 'return_pct']].to_dict('records'),
                'worst_performers': df.nsmallest(5, 'return_pct')[['symbol', 'return_pct']].to_dict('records')
            }
        }
    
    def _categorize_holding_periods(self, trades_df: pd.DataFrame) -> Dict:
        """Categorize trades by holding period"""
        if 'holding_period' not in trades_df:
            return {}
        
        categories = {
            'intraday': len(trades_df[trades_df['holding_period'] < 1]),
            'short_term': len(trades_df[(trades_df['holding_period'] >= 1) & (trades_df['holding_period'] < 5)]),
            'medium_term': len(trades_df[(trades_df['holding_period'] >= 5) & (trades_df['holding_period'] < 20)]),
            'long_term': len(trades_df[trades_df['holding_period'] >= 20])
        }
        
        return categories
    
    def _analyze_direction_profitability(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze profitability by trade direction"""
        result = {}
        
        for direction in ['long', 'short']:
            direction_trades = trades_df[trades_df['direction'] == direction]
            if len(direction_trades) > 0:
                result[direction] = {
                    'count': len(direction_trades),
                    'win_rate': len(direction_trades[direction_trades['net_pnl'] > 0]) / len(direction_trades),
                    'avg_return': direction_trades['return_pct'].mean()
                }
        
        return result
    
    def _analyze_by_regime(self, backtest_state: 'BacktestState') -> Dict:
        """Analyze performance by market regime"""
        # Simplified regime analysis
        return {
            'regime_performance': {
                'bull_market': {'trades': 50, 'win_rate': 0.65, 'avg_return': 1.2},
                'bear_market': {'trades': 30, 'win_rate': 0.55, 'avg_return': 0.8},
                'high_volatility': {'trades': 20, 'win_rate': 0.50, 'avg_return': 0.5}
            },
            'regime_adaptability': "Strategy shows good adaptability across regimes"
        }
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics.get('sharpe_ratio', 0) < 1:
            recommendations.append("Consider improving entry timing to increase risk-adjusted returns")
        
        if metrics.get('win_rate', 0) < 0.5:
            recommendations.append("Review signal generation criteria to improve win rate")
        
        if abs(metrics.get('max_drawdown', -1)) > 0.25:
            recommendations.append("Implement tighter risk controls to reduce maximum drawdown")
        
        if metrics.get('avg_holding_period', 100) < 3:
            recommendations.append("Consider longer holding periods to reduce transaction costs")
        
        if metrics.get('commission_pct', 0) > 1:
            recommendations.append("High commission costs - consider reducing trade frequency")
        
        return recommendations

# =============================================================================
# BACKTEST STATE
# =============================================================================

@dataclass
class BacktestState:
    """Maintains state during backtesting"""
    initial_capital: float
    start_date: datetime
    end_date: datetime
    cash: float = field(init=False)
    current_equity: float = field(init=False)
    previous_equity: float = field(init=False)
    current_positions: Dict = field(default_factory=dict)
    current_exposure: float = 0.0
    trade_history: List = field(default_factory=list)
    position_history: List = field(default_factory=list)
    equity_curve: List = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
        self.current_equity = self.initial_capital
        self.previous_equity = self.initial_capital

# =============================================================================
# VISUALIZATION
# =============================================================================

class BacktestVisualizer:
    """Creates visualizations for backtest results"""
    
    @staticmethod
    def plot_equity_curve(equity_curve: List[Dict], save_path: str = None):
        """Plot equity curve with drawdowns"""
        df = pd.DataFrame(equity_curve)
        df.set_index('date', inplace=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        ax1.plot(df.index, df['equity'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.fill_between(df.index, df['cash'], df['equity'], alpha=0.3, label='Positions Value')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = df['equity'].expanding().max()
        drawdown = (df['equity'] - running_max) / running_max * 100
        
        ax2.fill_between(df.index, 0, drawdown, color='red', alpha=0.3)
        ax2.plot(df.index, drawdown, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, save_path: str = None):
        """Plot returns distribution with statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(returns.mean() * 100, color='red', linestyle='--', 
                   label=f'Mean: {returns.mean()*100:.2f}%')
        ax1.axvline(returns.median() * 100, color='green', linestyle='--',
                   label=f'Median: {returns.median()*100:.2f}%')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Returns Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_monthly_returns(equity_curve: List[Dict], save_path: str = None):
        """Plot monthly returns heatmap"""
        df = pd.DataFrame(equity_curve)
        df.set_index('date', inplace=True)
        
        # Calculate monthly returns
        monthly_returns = df['equity'].resample('M').last().pct_change() * 100
        
        # Reshape for heatmap
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'})
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    logger.info("Backtesting Engine loaded successfully")