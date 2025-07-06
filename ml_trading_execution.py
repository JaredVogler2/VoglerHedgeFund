# ml_trading_execution.py

"""
ML Trading System - Live Trading Integration with Alpaca
Professional execution system with real-time monitoring and risk controls
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import websocket
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from queue import Queue, Empty
import requests

logger = logging.getLogger(__name__)

# =============================================================================
# ALPACA CLIENT
# =============================================================================

class AlpacaTradingClient:
    """Manages connection and operations with Alpaca API"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url='https://paper-api.alpaca.markets',  # Use paper trading
            api_version='v2'
        )
        
        # WebSocket for real-time data
        self.ws_client = None
        self.market_data_ws = None
        
        # Order management
        self.pending_orders = {}
        self.active_orders = {}
        
        # Position tracking
        self.positions = {}
        self.account_info = None
        
        # Initialize account
        self._initialize_account()
        
    def _initialize_account(self):
        """Initialize account information"""
        try:
            self.account_info = self.api.get_account()
            logger.info(f"Account initialized. Buying power: ${self.account_info.buying_power}")
            logger.info(f"Account status: {self.account_info.status}")
            
            # Check PDT rule
            if self.account_info.pattern_day_trader:
                logger.warning("Account is marked as Pattern Day Trader")
                
        except Exception as e:
            logger.error(f"Failed to initialize account: {e}")
            raise
    
    def get_account_status(self) -> Dict:
        """Get current account status"""
        try:
            account = self.api.get_account()
            
            return {
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'daytrade_count': account.daytrade_count,
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'maintenance_margin': float(account.maintenance_margin),
                'initial_margin': float(account.initial_margin)
            }
            
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all current positions"""
        try:
            positions = self.api.list_positions()
            
            position_dict = {}
            for position in positions:
                position_dict[position.symbol] = {
                    'symbol': position.symbol,
                    'quantity': int(position.qty),
                    'side': position.side,
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price) if hasattr(position, 'current_price') else 0,
                    'lastday_price': float(position.lastday_price) if hasattr(position, 'lastday_price') else 0,
                    'change_today': float(position.change_today) if hasattr(position, 'change_today') else 0
                }
                
            self.positions = position_dict
            return position_dict
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def place_order(self, symbol: str, quantity: int, side: str, 
                   order_type: str = 'market', limit_price: float = None,
                   stop_price: float = None, time_in_force: str = 'day',
                   extended_hours: bool = False) -> Optional[Dict]:
        """Place an order through Alpaca"""
        try:
            # Validate order
            if not self._validate_order(symbol, quantity, side):
                return None
            
            # Build order parameters
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force,
                'extended_hours': extended_hours
            }
            
            if order_type == 'limit' and limit_price:
                order_params['limit_price'] = limit_price
            elif order_type == 'stop' and stop_price:
                order_params['stop_price'] = stop_price
            elif order_type == 'stop_limit' and limit_price and stop_price:
                order_params['limit_price'] = limit_price
                order_params['stop_price'] = stop_price
            
            # Submit order
            order = self.api.submit_order(**order_params)
            
            logger.info(f"Order submitted: {side} {quantity} {symbol} @ {order_type}")
            
            # Track order
            order_dict = {
                'id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            self.pending_orders[order.id] = order_dict
            
            return order_dict
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _validate_order(self, symbol: str, quantity: int, side: str) -> bool:
        """Validate order before submission"""
        # Check account status
        account = self.get_account_status()
        
        if account.get('trading_blocked'):
            logger.error("Trading is blocked on this account")
            return False
        
        if account.get('account_blocked'):
            logger.error("Account is blocked")
            return False
        
        # Check buying power for buy orders
        if side == 'buy':
            # Get current price
            try:
                quote = self.api.get_latest_quote(symbol)
                price = quote.ap  # ask price
                required_capital = price * quantity
                
                if required_capital > float(account.get('buying_power', 0)):
                    logger.error(f"Insufficient buying power. Required: ${required_capital:.2f}, Available: ${account.get('buying_power')}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error checking price for {symbol}: {e}")
                return False
        
        # Check position for sell orders
        elif side == 'sell':
            positions = self.get_positions()
            if symbol not in positions:
                logger.error(f"No position in {symbol} to sell")
                return False
                
            if positions[symbol]['quantity'] < quantity:
                logger.error(f"Insufficient shares. Have: {positions[symbol]['quantity']}, Want to sell: {quantity}")
                return False
        
        # Check PDT rule
        if account.get('pattern_day_trader') and account.get('daytrade_count', 0) >= 3:
            if float(account.get('equity', 0)) < 25000:
                logger.warning("PDT rule: Cannot make more day trades with equity < $25,000")
                # Don't block the order, just warn
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        try:
            cancelled_orders = self.api.cancel_all_orders()
            logger.info(f"Cancelled {len(cancelled_orders)} orders")
            
            self.pending_orders.clear()
            
            return len(cancelled_orders)
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order"""
        try:
            order = self.api.get_order(order_id)
            
            return {
                'id': order.id,
                'status': order.status,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'expired_at': order.expired_at,
                'canceled_at': order.canceled_at,
                'failed_at': order.failed_at
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def get_recent_orders(self, limit: int = 100) -> List[Dict]:
        """Get recent orders"""
        try:
            orders = self.api.list_orders(status='all', limit=limit)
            
            order_list = []
            for order in orders:
                order_list.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'quantity': int(order.qty),
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'created_at': order.created_at,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
                })
                
            return order_list
            
        except Exception as e:
            logger.error(f"Error getting recent orders: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """Close entire position in a symbol"""
        try:
            self.api.close_position(symbol)
            logger.info(f"Closed position in {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position in {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            self.api.close_all_positions()
            logger.info("Closed all positions")
            return True
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

# =============================================================================
# ORDER MANAGER
# =============================================================================

class OrderManager:
    """Manages order placement, monitoring, and execution"""
    
    def __init__(self, alpaca_client: AlpacaTradingClient, risk_manager):
        self.alpaca = alpaca_client
        self.risk_manager = risk_manager
        self.active_orders = {}
        self.order_history = []
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_time': 0,
            'slippage_stats': []
        }
        
    def execute_signal(self, signal: 'TradingSignal') -> Optional[Dict]:
        """Execute a trading signal"""
        logger.info(f"Executing signal for {signal.symbol}")
        
        # Pre-trade risk check
        current_positions = self.alpaca.get_positions()
        risk_approved, risk_message = self.risk_manager.check_pre_trade_risk(
            signal, current_positions
        )
        
        if not risk_approved:
            logger.warning(f"Risk check failed for {signal.symbol}: {risk_message}")
            return None
        
        # Calculate order size
        order_details = self._calculate_order_details(signal)
        
        if order_details['quantity'] == 0:
            logger.warning(f"Order quantity is 0 for {signal.symbol}")
            return None
        
        # Determine order type and parameters
        order_params = self._determine_order_parameters(signal, order_details)
        
        # Place the order
        order = self.alpaca.place_order(
            symbol=signal.symbol,
            quantity=order_details['quantity'],
            side=order_params['side'],
            order_type=order_params['type'],
            limit_price=order_params.get('limit_price'),
            time_in_force=order_params['time_in_force']
        )
        
        if order:
            # Track the order
            self.active_orders[order['id']] = {
                'order': order,
                'signal': signal,
                'placed_at': datetime.now(),
                'expected_price': signal.signal_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            # Place stop loss and take profit orders
            self._place_exit_orders(signal, order)
            
            self.execution_stats['total_orders'] += 1
            
        return order
    
    def _calculate_order_details(self, signal: 'TradingSignal') -> Dict:
        """Calculate order size and other details"""
        account = self.alpaca.get_account_status()
        portfolio_value = float(account['portfolio_value'])
        
        # Calculate position value
        position_value = portfolio_value * signal.position_size
        
        # Get current price
        try:
            quote = self.alpaca.api.get_latest_quote(signal.symbol)
            current_price = quote.ap if signal.direction == 'long' else quote.bp
        except:
            current_price = signal.signal_price
        
        # Calculate shares
        quantity = int(position_value / current_price)
        
        # Apply minimum share requirement
        quantity = max(1, quantity)
        
        # Round to nearest lot for some symbols
        if current_price > 100:
            quantity = round(quantity / 10) * 10  # Round to nearest 10
        elif current_price > 50:
            quantity = round(quantity / 5) * 5    # Round to nearest 5
        
        return {
            'quantity': quantity,
            'position_value': quantity * current_price,
            'current_price': current_price
        }
    
    def _determine_order_parameters(self, signal: 'TradingSignal', 
                                   order_details: Dict) -> Dict:
        """Determine order type and parameters"""
        # Default to limit orders for better execution
        order_type = 'limit'
        
        # Set limit price with small buffer for execution
        if signal.direction == 'long':
            limit_price = order_details['current_price'] * 1.001  # 0.1% above ask
            side = 'buy'
        else:
            limit_price = order_details['current_price'] * 0.999  # 0.1% below bid
            side = 'sell'
        
        # Use IOC (Immediate or Cancel) for quick execution
        time_in_force = 'ioc' if signal.confidence_score > 0.8 else 'day'
        
        return {
            'side': side,
            'type': order_type,
            'limit_price': round(limit_price, 2),
            'time_in_force': time_in_force
        }
    
    def _place_exit_orders(self, signal: 'TradingSignal', entry_order: Dict):
        """Place stop loss and take profit orders"""
        # Wait a moment for the entry order to fill
        time.sleep(1)
        
        # Check if order filled
        order_status = self.alpaca.get_order_status(entry_order['id'])
        
        if order_status and order_status['status'] == 'filled':
            filled_qty = order_status['filled_qty']
            
            # Place stop loss order
            if signal.direction == 'long':
                stop_side = 'sell'
            else:
                stop_side = 'buy'
            
            stop_order = self.alpaca.place_order(
                symbol=signal.symbol,
                quantity=filled_qty,
                side=stop_side,
                order_type='stop',
                stop_price=signal.stop_loss,
                time_in_force='gtc'  # Good till cancelled
            )
            
            if stop_order:
                logger.info(f"Stop loss order placed for {signal.symbol} at {signal.stop_loss}")
            
            # Place take profit order
            tp_order = self.alpaca.place_order(
                symbol=signal.symbol,
                quantity=filled_qty,
                side=stop_side,
                order_type='limit',
                limit_price=signal.take_profit,
                time_in_force='gtc'
            )
            
            if tp_order:
                logger.info(f"Take profit order placed for {signal.symbol} at {signal.take_profit}")
    
    def monitor_orders(self):
        """Monitor active orders and update status"""
        orders_to_remove = []
        
        for order_id, order_info in self.active_orders.items():
            try:
                # Get current status
                status = self.alpaca.get_order_status(order_id)
                
                if not status:
                    continue
                
                # Update based on status
                if status['status'] == 'filled':
                    self._handle_filled_order(order_id, order_info, status)
                    orders_to_remove.append(order_id)
                    
                elif status['status'] == 'cancelled':
                    self.execution_stats['cancelled_orders'] += 1
                    orders_to_remove.append(order_id)
                    
                elif status['status'] == 'rejected':
                    self.execution_stats['rejected_orders'] += 1
                    logger.error(f"Order {order_id} was rejected")
                    orders_to_remove.append(order_id)
                    
                # Check if order is too old (not filled within 5 minutes)
                elif (datetime.now() - order_info['placed_at']).seconds > 300:
                    logger.warning(f"Order {order_id} not filled within 5 minutes, cancelling")
                    self.alpaca.cancel_order(order_id)
                    orders_to_remove.append(order_id)
                    
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
        
        # Remove processed orders
        for order_id in orders_to_remove:
            del self.active_orders[order_id]
    
    def _handle_filled_order(self, order_id: str, order_info: Dict, status: Dict):
        """Handle a filled order"""
        filled_price = status['filled_avg_price']
        expected_price = order_info['expected_price']
        
        # Calculate slippage
        if order_info['signal'].direction == 'long':
            slippage = (filled_price - expected_price) / expected_price
        else:
            slippage = (expected_price - filled_price) / expected_price
        
        self.execution_stats['slippage_stats'].append(slippage)
        self.execution_stats['filled_orders'] += 1
        
        # Calculate fill time
        fill_time = (datetime.now() - order_info['placed_at']).seconds
        current_avg = self.execution_stats['avg_fill_time']
        total_filled = self.execution_stats['filled_orders']
        self.execution_stats['avg_fill_time'] = (
            (current_avg * (total_filled - 1) + fill_time) / total_filled
        )
        
        # Log execution
        logger.info(f"Order {order_id} filled at {filled_price} (slippage: {slippage*100:.2f}%)")
        
        # Add to history
        self.order_history.append({
            'order_id': order_id,
            'symbol': order_info['signal'].symbol,
            'filled_at': datetime.now(),
            'filled_price': filled_price,
            'expected_price': expected_price,
            'slippage': slippage,
            'fill_time': fill_time
        })
    
    def get_execution_metrics(self) -> Dict:
        """Get execution quality metrics"""
        metrics = self.execution_stats.copy()
        
        if self.execution_stats['slippage_stats']:
            slippage_array = np.array(self.execution_stats['slippage_stats'])
            metrics['avg_slippage'] = np.mean(slippage_array)
            metrics['median_slippage'] = np.median(slippage_array)
            metrics['slippage_std'] = np.std(slippage_array)
            metrics['positive_slippage_pct'] = (slippage_array > 0).mean()
        
        metrics['fill_rate'] = (
            self.execution_stats['filled_orders'] / 
            self.execution_stats['total_orders']
            if self.execution_stats['total_orders'] > 0 else 0
        )
        
        return metrics

# =============================================================================
# PAPER TRADING SIMULATOR
# =============================================================================

class PaperTradingSimulator:
    """Simulates trading without real money"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.order_id_counter = 0
        self.orders = {}
        self.trade_history = []
        self.equity_curve = []
        
    def place_order(self, symbol: str, quantity: int, side: str,
                   order_type: str = 'market', limit_price: float = None,
                   current_price: float = None) -> Dict:
        """Simulate order placement"""
        self.order_id_counter += 1
        order_id = f"PAPER_{self.order_id_counter}"
        
        # For paper trading, assume immediate fill at current price
        if not current_price:
            current_price = limit_price if limit_price else 100  # Default
        
        fill_price = current_price
        
        # Calculate commission
        commission = max(1.0, quantity * 0.005)
        
        # Execute based on side
        if side == 'buy':
            cost = (quantity * fill_price) + commission
            
            if cost > self.cash:
                return {'error': 'Insufficient funds'}
            
            self.cash -= cost
            
            if symbol in self.positions:
                # Add to existing position
                old_qty = self.positions[symbol]['quantity']
                old_avg = self.positions[symbol]['avg_price']
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * fill_price)) / new_qty
                
                self.positions[symbol] = {
                    'quantity': new_qty,
                    'avg_price': new_avg
                }
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': fill_price
                }
                
        else:  # sell
            if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                return {'error': 'Insufficient shares'}
            
            # Calculate P&L
            avg_cost = self.positions[symbol]['avg_price']
            gross_pnl = (fill_price - avg_cost) * quantity
            net_pnl = gross_pnl - commission
            
            self.cash += (quantity * fill_price) - commission
            
            # Update position
            self.positions[symbol]['quantity'] -= quantity
            
            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': avg_cost,
                'exit_price': fill_price,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'commission': commission,
                'timestamp': datetime.now()
            })
        
        # Create order record
        order = {
            'id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'fill_price': fill_price,
            'commission': commission,
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
        self.orders[order_id] = order
        
        # Update equity curve
        self._update_equity()
        
        return order
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.positions.copy()
    
    def get_account_status(self) -> Dict:
        """Get account status"""
        # Calculate position values (assuming current prices)
        positions_value = sum(
            pos['quantity'] * pos['avg_price'] * 1.01  # Assume 1% gain
            for pos in self.positions.values()
        )
        
        total_equity = self.cash + positions_value
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'buying_power': self.cash,
            'num_positions': len(self.positions),
            'pnl': total_equity - self.initial_capital,
            'pnl_pct': ((total_equity / self.initial_capital) - 1) * 100
        }
    
    def _update_equity(self):
        """Update equity curve"""
        account = self.get_account_status()
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': account['total_equity'],
            'cash': account['cash'],
            'positions_value': account['positions_value']
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['net_pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['net_pnl'] < 0]),
            'win_rate': len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df),
            'total_pnl': trades_df['net_pnl'].sum(),
            'avg_win': trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean(),
            'avg_loss': trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean(),
            'total_commission': trades_df['commission'].sum()
        }
        
        # Calculate profit factor
        total_wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        total_losses = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else np.inf
        
        return metrics

# =============================================================================
# LIVE TRADING COORDINATOR
# =============================================================================

class LiveTradingCoordinator:
    """Coordinates all components for live trading"""
    
    def __init__(self, config, signal_generator, risk_manager, use_paper_trading=True):
        self.config = config
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.use_paper_trading = use_paper_trading
        
        # Initialize trading client
        if use_paper_trading:
            self.trading_client = PaperTradingSimulator()
            logger.info("Using paper trading simulator")
        else:
            self.trading_client = AlpacaTradingClient(config)
            logger.info("Using Alpaca live trading")
        
        # Initialize order manager
        self.order_manager = OrderManager(self.trading_client, risk_manager)
        
        # Trading state
        self.is_trading = False
        self.last_signal_time = None
        self.daily_trades = 0
        self.daily_pnl = 0
        
    def start_trading_session(self):
        """Start a trading session"""
        logger.info("Starting trading session...")
        
        self.is_trading = True
        self.daily_trades = 0
        self.daily_pnl = 0
        
        # Check market hours
        if not self._is_market_open():
            logger.warning("Market is closed")
            return
        
        # Main trading loop
        try:
            while self.is_trading and self._is_market_open():
                # Generate signals
                if self._should_generate_signals():
                    self._process_signals()
                
                # Monitor positions
                self._monitor_positions()
                
                # Monitor orders
                self.order_manager.monitor_orders()
                
                # Sleep for a bit
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Trading session interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        finally:
            self.stop_trading_session()
    
    def stop_trading_session(self):
        """Stop trading session"""
        logger.info("Stopping trading session...")
        
        self.is_trading = False
        
        # Cancel all pending orders
        if hasattr(self.trading_client, 'cancel_all_orders'):
            self.trading_client.cancel_all_orders()
        
        # Log session summary
        self._log_session_summary()
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        if self.use_paper_trading:
            # For paper trading, simulate market hours
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)
            
            return market_open <= now <= market_close and now.weekday() < 5
        else:
            # Check with Alpaca
            try:
                clock = self.trading_client.api.get_clock()
                return clock.is_open
            except:
                return False
    
    def _should_generate_signals(self) -> bool:
        """Check if we should generate new signals"""
        if not self.last_signal_time:
            return True
        
        # Generate signals every 5 minutes
        time_since_last = (datetime.now() - self.last_signal_time).seconds
        return time_since_last >= 300
    
    def _process_signals(self):
        """Process new trading signals"""
        logger.info("Generating trading signals...")
        
        # Get current market data (simplified for example)
        # In production, this would fetch real data
        mock_predictions = self._generate_mock_predictions()
        
        # Generate signals
        signals = self.signal_generator.generate_signals({}, mock_predictions)
        
        logger.info(f"Generated {len(signals)} signals")
        
        # Execute signals
        for signal in signals:
            # Check daily trade limit
            if self.daily_trades >= 10:
                logger.warning("Daily trade limit reached")
                break
            
            # Execute signal
            order = self.order_manager.execute_signal(signal)
            
            if order:
                self.daily_trades += 1
                logger.info(f"Executed trade #{self.daily_trades}: {signal.symbol}")
        
        self.last_signal_time = datetime.now()
    
    def _generate_mock_predictions(self) -> Dict:
        """Generate mock predictions for testing"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        predictions = {}
        
        for symbol in symbols:
            # Random predictions
            predictions[symbol] = pd.DataFrame({
                'consensus_direction': [np.random.uniform(0.4, 0.7)],
                'consensus_return': [np.random.uniform(-0.02, 0.02)],
                'predicted_volatility': [np.random.uniform(0.01, 0.03)],
                'confidence_score': [np.random.uniform(0.5, 0.9)],
                'consensus_agreement': [np.random.uniform(0.1, 0.3)]
            })
        
        return predictions
    
    def _monitor_positions(self):
        """Monitor and manage existing positions"""
        positions = self.trading_client.get_positions()
        
        for symbol, position in positions.items():
            # Update stop losses
            # Check for exit conditions
            # etc.
            pass
    
    def _log_session_summary(self):
        """Log trading session summary"""
        logger.info("=== Trading Session Summary ===")
        logger.info(f"Total trades: {self.daily_trades}")
        
        # Get performance metrics
        if hasattr(self.trading_client, 'get_performance_metrics'):
            metrics = self.trading_client.get_performance_metrics()
            logger.info(f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
            logger.info(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        
        # Get execution metrics
        exec_metrics = self.order_manager.get_execution_metrics()
        logger.info(f"Fill rate: {exec_metrics.get('fill_rate', 0)*100:.1f}%")
        logger.info(f"Avg slippage: {exec_metrics.get('avg_slippage', 0)*100:.2f}%")

# Example usage
if __name__ == "__main__":
    logger.info("Live Trading Integration loaded successfully")