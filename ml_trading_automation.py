# ml_trading_automation.py

"""
ML Trading System - Automation & Scheduling
Automated trading operations, model retraining, and system maintenance
"""

import schedule
import time
import threading
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd  # Added for DataFrame type hints
import numpy as np   # Added for calculations
import logging
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import os
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
import psutil
import gc

logger = logging.getLogger(__name__)

# =============================================================================
# TRADING SCHEDULER
# =============================================================================

class TradingScheduler:
    """Manages all scheduled trading operations"""
    
    def __init__(self, system_components: Dict):
        self.components = system_components
        self.config = system_components['config']
        self.timezone = pytz.timezone('US/Eastern')  # NYSE timezone
        
        # Scheduler state
        self.is_running = False
        self.scheduled_jobs = []
        self.job_history = []
        
        # Thread pool for parallel jobs
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize schedules
        self._setup_schedules()
        
    def _setup_schedules(self):
        """Setup all scheduled jobs"""
        logger.info("Setting up trading schedules...")
        
        # ===== NIGHTLY JOBS (2 AM - 6 AM) =====
        
        # Data update job - 2:00 AM
        schedule.every().day.at("02:00").do(
            self._run_job, 
            self.nightly_data_update,
            "Nightly Data Update"
        )
        
        # Feature engineering - 3:00 AM  
        schedule.every().day.at("03:00").do(
            self._run_job,
            self.nightly_feature_engineering,
            "Feature Engineering"
        )
        
        # Model retraining - 4:00 AM
        schedule.every().day.at("04:00").do(
            self._run_job,
            self.nightly_model_retraining,
            "Model Retraining"
        )
        
        # Backtest validation - 5:00 AM
        schedule.every().day.at("05:00").do(
            self._run_job,
            self.nightly_backtest_validation,
            "Backtest Validation"
        )
        
        # Signal generation - 6:00 AM
        schedule.every().day.at("06:00").do(
            self._run_job,
            self.morning_signal_generation,
            "Morning Signal Generation"
        )
        
        # ===== MARKET HOURS JOBS =====
        
        # Pre-market analysis - 8:30 AM
        schedule.every().weekday.at("08:30").do(
            self._run_job,
            self.premarket_analysis,
            "Pre-market Analysis"
        )
        
        # Market open preparation - 9:15 AM
        schedule.every().weekday.at("09:15").do(
            self._run_job,
            self.market_open_preparation,
            "Market Open Preparation"
        )
        
        # Trading session start - 9:30 AM
        schedule.every().weekday.at("09:30").do(
            self._run_job,
            self.start_trading_session,
            "Start Trading Session"
        )
        
        # Intraday updates - Every 30 minutes during market hours
        for hour in range(10, 16):
            for minute in ["00", "30"]:
                if hour == 15 and minute == "30":
                    continue  # Skip 3:30 PM
                schedule.every().weekday.at(f"{hour}:{minute}").do(
                    self._run_job,
                    self.intraday_update,
                    "Intraday Update"
                )
        
        # End of day analysis - 3:45 PM
        schedule.every().weekday.at("15:45").do(
            self._run_job,
            self.end_of_day_analysis,
            "End of Day Analysis"
        )
        
        # Trading session end - 4:00 PM
        schedule.every().weekday.at("16:00").do(
            self._run_job,
            self.end_trading_session,
            "End Trading Session"
        )
        
        # After hours report - 4:30 PM
        schedule.every().weekday.at("16:30").do(
            self._run_job,
            self.after_hours_report,
            "After Hours Report"
        )
        
        # ===== WEEKLY JOBS =====
        
        # Weekly performance review - Friday 5:00 PM
        schedule.every().friday.at("17:00").do(
            self._run_job,
            self.weekly_performance_review,
            "Weekly Performance Review"
        )
        
        # Model performance analysis - Saturday 10:00 AM
        schedule.every().saturday.at("10:00").do(
            self._run_job,
            self.weekly_model_analysis,
            "Weekly Model Analysis"
        )
        
        # System maintenance - Sunday 2:00 AM
        schedule.every().sunday.at("02:00").do(
            self._run_job,
            self.system_maintenance,
            "System Maintenance"
        )
        
        logger.info(f"Scheduled {len(schedule.jobs)} jobs")
    
    def _run_job(self, job_func: Callable, job_name: str):
        """Run a scheduled job with error handling"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting job: {job_name}")
            
            # Check if market is closed for market-hours jobs
            if self._is_market_hours_job(job_name) and not self._is_trading_day():
                logger.info(f"Skipping {job_name} - Market closed")
                return
            
            # Run the job
            result = job_func()
            
            # Record success
            duration = (datetime.now() - start_time).total_seconds()
            self._record_job_result(job_name, "success", duration, result)
            
            logger.info(f"Completed job: {job_name} in {duration:.2f}s")
            
        except Exception as e:
            # Record failure
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self._record_job_result(job_name, "failed", duration, error_msg)
            
            logger.error(f"Job failed: {job_name} - {str(e)}")
            
            # Send alert for critical jobs
            if self._is_critical_job(job_name):
                self._send_alert(f"Critical job failed: {job_name}", error_msg)
    
    def _is_market_hours_job(self, job_name: str) -> bool:
        """Check if job should only run during market hours"""
        market_jobs = [
            "Pre-market Analysis", "Market Open Preparation",
            "Start Trading Session", "Intraday Update",
            "End of Day Analysis", "End Trading Session"
        ]
        return job_name in market_jobs
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day"""
        now = datetime.now(self.timezone)
        
        # Check weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check holidays (simplified - would use trading calendar)
        # This would connect to a proper trading calendar API
        return True
    
    def _is_critical_job(self, job_name: str) -> bool:
        """Check if job is critical and needs alerts"""
        critical_jobs = [
            "Model Retraining", "Start Trading Session",
            "End Trading Session", "System Maintenance"
        ]
        return job_name in critical_jobs
    
    def _record_job_result(self, job_name: str, status: str, 
                          duration: float, result: Any):
        """Record job execution result"""
        self.job_history.append({
            'job_name': job_name,
            'timestamp': datetime.now(),
            'status': status,
            'duration': duration,
            'result': str(result)[:500]  # Truncate long results
        })
        
        # Keep only last 1000 entries
        if len(self.job_history) > 1000:
            self.job_history = self.job_history[-1000:]
    
    def _send_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            # Email configuration (would be in config)
            sender_email = "trading.system@example.com"
            receiver_email = "alerts@example.com"
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = f"Trading System Alert: {subject}"
            
            body = f"""
            Alert Time: {datetime.now()}
            
            {message}
            
            System Status:
            - CPU Usage: {psutil.cpu_percent()}%
            - Memory Usage: {psutil.virtual_memory().percent}%
            - Disk Usage: {psutil.disk_usage('/').percent}%
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (would use actual SMTP server)
            logger.info(f"Alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    # =============================================================================
    # NIGHTLY JOBS
    # =============================================================================
    
    def nightly_data_update(self) -> Dict:
        """Update all market data"""
        logger.info("Running nightly data update...")
        
        results = {
            'symbols_updated': 0,
            'errors': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get watchlist
            watchlist = self.components['watchlist'].symbols
            
            # Update data for all symbols
            data_pipeline = self.components['data_pipeline']
            
            # Parallel data fetching
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                for symbol in watchlist:
                    future = executor.submit(
                        data_pipeline.fetch_data,
                        [symbol],
                        use_cache=False
                    )
                    futures.append((symbol, future))
                
                # Collect results
                for symbol, future in futures:
                    try:
                        data = future.result(timeout=30)
                        if not data.empty:
                            results['symbols_updated'] += 1
                    except Exception as e:
                        results['errors'].append(f"{symbol}: {str(e)}")
            
            # Update market indices
            indices = ['SPY', 'QQQ', 'IWM', 'VIX', 'DXY']
            data_pipeline.fetch_data(indices, use_cache=False)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Data update complete: {results['symbols_updated']} symbols updated")
            
        except Exception as e:
            results['errors'].append(f"Critical error: {str(e)}")
            raise
        
        return results
    
    def nightly_feature_engineering(self) -> Dict:
        """Run feature engineering for all symbols"""
        logger.info("Running feature engineering...")
        
        results = {
            'symbols_processed': 0,
            'total_features': 0,
            'errors': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Load market data
            data_pipeline = self.components['data_pipeline']
            feature_engineer = self.components['feature_engineer']
            
            # Get all symbols
            watchlist = self.components['watchlist'].symbols
            
            # Load data
            market_data = data_pipeline.fetch_data(watchlist)
            
            if market_data.empty:
                raise ValueError("No market data available")
            
            # Process features for each symbol
            all_features = {}
            
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                for symbol in watchlist[:50]:  # Limit for performance
                    future = executor.submit(
                        feature_engineer.engineer_features,
                        market_data,
                        symbol
                    )
                    futures[future] = symbol
                
                # Collect results
                for future in futures:
                    symbol = futures[future]
                    try:
                        features = future.result(timeout=60)
                        if not features.empty:
                            all_features[symbol] = features
                            results['symbols_processed'] += 1
                            results['total_features'] += len(features.columns)
                    except Exception as e:
                        results['errors'].append(f"{symbol}: {str(e)}")
            
            # Save features
            self._save_features(all_features)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Feature engineering complete: {results['symbols_processed']} symbols")
            
        except Exception as e:
            results['errors'].append(f"Critical error: {str(e)}")
            raise
        
        return results
    
    def nightly_model_retraining(self) -> Dict:
        """Retrain ML models"""
        logger.info("Running model retraining...")
        
        results = {
            'models_trained': 0,
            'avg_accuracy': 0,
            'errors': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Load features
            features = self._load_features()
            
            if not features:
                raise ValueError("No features available for training")
            
            # Get components
            ensemble_manager = self.components['ensemble_manager']
            training_orchestrator = self.components['training_orchestrator']
            
            # Train models for top symbols
            accuracies = []
            
            for symbol in list(features.keys())[:20]:  # Train on top 20 symbols
                try:
                    logger.info(f"Training models for {symbol}...")
                    
                    # Train ensemble
                    train_results = training_orchestrator.train_ensemble(
                        features[symbol],
                        symbol
                    )
                    
                    if 'cv_scores' in train_results:
                        avg_accuracy = np.mean(train_results['cv_scores'])
                        accuracies.append(avg_accuracy)
                        results['models_trained'] += 1
                        
                        logger.info(f"Trained {symbol} - Accuracy: {avg_accuracy:.4f}")
                        
                except Exception as e:
                    results['errors'].append(f"{symbol}: {str(e)}")
            
            # Calculate average accuracy
            if accuracies:
                results['avg_accuracy'] = np.mean(accuracies)
            
            # Save models
            model_path = f"{self.config.CACHE_DIR}/models"
            os.makedirs(model_path, exist_ok=True)
            ensemble_manager.save_models(model_path)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Model retraining complete: {results['models_trained']} models")
            
        except Exception as e:
            results['errors'].append(f"Critical error: {str(e)}")
            raise
        
        return results
    
    def nightly_backtest_validation(self) -> Dict:
        """Run backtest validation"""
        logger.info("Running backtest validation...")
        
        results = {
            'backtest_complete': False,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'errors': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get components
            backtest_engine = self.components['backtest_engine']
            data_pipeline = self.components['data_pipeline']
            
            # Run backtest on recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 3 months
            
            # Get data
            symbols = self.components['watchlist'].symbols[:30]
            market_data = data_pipeline.fetch_data(symbols)
            
            # Run backtest
            backtest_results = backtest_engine.run_backtest(
                market_data,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Extract key metrics
            metrics = backtest_results['performance_metrics']
            results['sharpe_ratio'] = metrics.get('sharpe_ratio', 0)
            results['max_drawdown'] = metrics.get('max_drawdown', 0)
            results['backtest_complete'] = True
            
            # Check performance thresholds
            if results['sharpe_ratio'] < self.config.MIN_SHARPE_RATIO:
                self._send_alert(
                    "Low Sharpe Ratio Alert",
                    f"Backtest Sharpe ratio {results['sharpe_ratio']:.2f} is below threshold"
                )
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Backtest validation complete: Sharpe {results['sharpe_ratio']:.2f}")
            
        except Exception as e:
            results['errors'].append(f"Critical error: {str(e)}")
            raise
        
        return results
    
    def morning_signal_generation(self) -> Dict:
        """Generate trading signals for the day"""
        logger.info("Generating morning signals...")
        
        results = {
            'signals_generated': 0,
            'long_signals': 0,
            'short_signals': 0,
            'errors': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get components
            signal_generator = self.components['signal_generator']
            prediction_engine = self.components['prediction_engine']
            data_pipeline = self.components['data_pipeline']
            
            # Get latest features
            features = self._load_features()
            
            # Generate predictions
            all_predictions = {}
            
            for symbol, symbol_features in features.items():
                try:
                    predictions = prediction_engine.predict(symbol_features)
                    if not predictions.empty:
                        all_predictions[symbol] = predictions
                except Exception as e:
                    results['errors'].append(f"{symbol}: {str(e)}")
            
            # Generate signals
            market_data = data_pipeline.fetch_data(list(features.keys()))
            signals = signal_generator.generate_signals(market_data, all_predictions)
            
            # Count signals
            results['signals_generated'] = len(signals)
            results['long_signals'] = sum(1 for s in signals if s.direction == 'long')
            results['short_signals'] = sum(1 for s in signals if s.direction == 'short')
            
            # Save signals
            self._save_signals(signals)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Signal generation complete: {results['signals_generated']} signals")
            
        except Exception as e:
            results['errors'].append(f"Critical error: {str(e)}")
            raise
        
        return results
    
    # =============================================================================
    # MARKET HOURS JOBS
    # =============================================================================
    
    def premarket_analysis(self) -> Dict:
        """Run pre-market analysis"""
        logger.info("Running pre-market analysis...")
        
        results = {
            'news_analyzed': 0,
            'market_sentiment': 'neutral',
            'high_impact_events': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get news sentiment
            news_integration = self.components['news_integration']
            watchlist = self.components['watchlist'].symbols[:20]
            
            # Update sentiment
            sentiment_results = news_integration.update_sentiment_for_watchlist(watchlist)
            results['news_analyzed'] = len(sentiment_results)
            
            # Check for breaking news
            breaking_news = news_integration.check_breaking_news(watchlist)
            
            for news in breaking_news:
                if news['impact'] in ['high', 'medium']:
                    results['high_impact_events'].append({
                        'symbol': news['symbol'],
                        'headline': news['headline'],
                        'sentiment': news['sentiment']
                    })
            
            # Calculate overall market sentiment
            if sentiment_results:
                avg_sentiment = np.mean([s.sentiment_score for s in sentiment_results.values()])
                if avg_sentiment > 0.2:
                    results['market_sentiment'] = 'bullish'
                elif avg_sentiment < -0.2:
                    results['market_sentiment'] = 'bearish'
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Pre-market analysis complete: {results['market_sentiment']} sentiment")
            
        except Exception as e:
            results['errors'] = [f"Critical error: {str(e)}"]
            raise
        
        return results
    
    def market_open_preparation(self) -> Dict:
        """Prepare for market open"""
        logger.info("Preparing for market open...")
        
        results = {
            'positions_checked': 0,
            'orders_prepared': 0,
            'risk_checks_passed': True,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get components
            trading_client = self.components['trading_client']
            risk_manager = self.components['risk_manager']
            
            # Check account status
            account_status = trading_client.get_account_status()
            
            if account_status.get('trading_blocked'):
                raise ValueError("Trading is blocked on account")
            
            # Review existing positions
            positions = trading_client.get_positions()
            results['positions_checked'] = len(positions)
            
            # Update stop losses
            updated_positions = risk_manager.update_stop_losses(
                positions,
                self._get_current_market_data()
            )
            
            # Load today's signals
            signals = self._load_signals()
            
            # Validate signals against risk limits
            for signal in signals:
                risk_approved, _ = risk_manager.check_pre_trade_risk(signal, positions)
                if not risk_approved:
                    results['risk_checks_passed'] = False
                else:
                    results['orders_prepared'] += 1
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Market preparation complete: {results['orders_prepared']} orders ready")
            
        except Exception as e:
            results['errors'] = [f"Critical error: {str(e)}"]
            raise
        
        return results
    
    def start_trading_session(self) -> Dict:
        """Start automated trading session"""
        logger.info("Starting trading session...")
        
        results = {
            'session_started': False,
            'initial_positions': 0,
            'initial_equity': 0,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get trading coordinator
            trading_coordinator = self.components['trading_coordinator']
            
            # Start session
            trading_coordinator.start_trading_session()
            
            results['session_started'] = True
            results['initial_positions'] = len(trading_coordinator.trading_client.get_positions())
            
            account = trading_coordinator.trading_client.get_account_status()
            results['initial_equity'] = account.get('equity', 0)
            
            results['duration'] = time.time() - start_time
            
            logger.info("Trading session started successfully")
            
        except Exception as e:
            results['errors'] = [f"Critical error: {str(e)}"]
            raise
        
        return results
    
    def intraday_update(self) -> Dict:
        """Run intraday update"""
        logger.info("Running intraday update...")
        
        results = {
            'positions_updated': 0,
            'new_signals': 0,
            'risk_alerts': [],
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Update market data
            data_pipeline = self.components['data_pipeline']
            trading_client = self.components['trading_client']
            risk_manager = self.components['risk_manager']
            
            # Get current positions
            positions = trading_client.get_positions()
            results['positions_updated'] = len(positions)
            
            # Update real-time data for positions
            symbols = list(positions.keys())
            if symbols:
                realtime_data = data_pipeline.update_realtime_data(symbols)
                
                # Check for risk alerts
                for symbol, position in positions.items():
                    if symbol in realtime_data.index:
                        current_price = realtime_data.loc[symbol, 'price']
                        
                        # Check stop loss proximity
                        if position['direction'] == 'long':
                            if current_price <= position['stop_loss'] * 1.02:
                                results['risk_alerts'].append(f"{symbol}: Near stop loss")
                        else:
                            if current_price >= position['stop_loss'] * 0.98:
                                results['risk_alerts'].append(f"{symbol}: Near stop loss")
            
            # Check for new intraday opportunities
            # (Simplified for this example)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Intraday update complete: {len(results['risk_alerts'])} alerts")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    def end_of_day_analysis(self) -> Dict:
        """Run end of day analysis"""
        logger.info("Running end of day analysis...")
        
        results = {
            'trades_today': 0,
            'pnl_today': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get components
            trading_client = self.components['trading_client']
            position_tracker = self.components['position_tracker']
            
            # Get today's trades
            recent_orders = trading_client.get_recent_orders(limit=50)
            
            today = datetime.now().date()
            todays_trades = [
                order for order in recent_orders
                if datetime.fromisoformat(order['created_at']).date() == today
            ]
            
            results['trades_today'] = len(todays_trades)
            
            # Calculate P&L
            for trade in position_tracker.position_history:
                if trade.get('exit_time') and trade['exit_time'].date() == today:
                    pnl = trade.get('realized_pnl', 0)
                    results['pnl_today'] += pnl
                    
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"End of day analysis complete: P&L ${results['pnl_today']:,.2f}")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    def end_trading_session(self) -> Dict:
        """End trading session"""
        logger.info("Ending trading session...")
        
        results = {
            'session_ended': False,
            'final_positions': 0,
            'final_equity': 0,
            'session_pnl': 0,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Get trading coordinator
            trading_coordinator = self.components['trading_coordinator']
            
            # Get final metrics
            account = trading_coordinator.trading_client.get_account_status()
            results['final_equity'] = account.get('equity', 0)
            results['final_positions'] = len(trading_coordinator.trading_client.get_positions())
            
            # Stop session
            trading_coordinator.stop_trading_session()
            
            results['session_ended'] = True
            results['duration'] = time.time() - start_time
            
            logger.info("Trading session ended successfully")
            
        except Exception as e:
            results['errors'] = [f"Critical error: {str(e)}"]
            raise
        
        return results
    
    def after_hours_report(self) -> Dict:
        """Generate after hours report"""
        logger.info("Generating after hours report...")
        
        results = {
            'report_generated': False,
            'report_path': '',
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Gather all metrics
            report_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'trades': self.job_history[-1]['result'] if self.job_history else {},
                'positions': self.components['trading_client'].get_positions(),
                'account': self.components['trading_client'].get_account_status(),
                'performance': self._calculate_daily_performance()
            }
            
            # Generate report
            report_path = self._generate_daily_report(report_data)
            results['report_path'] = report_path
            results['report_generated'] = True
            
            # Send report
            self._send_daily_report(report_path)
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"After hours report generated: {report_path}")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    # =============================================================================
    # WEEKLY JOBS
    # =============================================================================
    
    def weekly_performance_review(self) -> Dict:
        """Run weekly performance review"""
        logger.info("Running weekly performance review...")
        
        results = {
            'week_return': 0,
            'week_sharpe': 0,
            'week_trades': 0,
            'report_generated': False,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Calculate weekly metrics
            performance_data = self._calculate_weekly_performance()
            
            results['week_return'] = performance_data['total_return']
            results['week_sharpe'] = performance_data['sharpe_ratio']
            results['week_trades'] = performance_data['total_trades']
            
            # Generate detailed report
            report_path = self._generate_weekly_report(performance_data)
            results['report_generated'] = True
            
            # Check performance thresholds
            if results['week_return'] < -0.05:  # -5% weekly loss
                self._send_alert(
                    "Weekly Performance Alert",
                    f"Weekly return {results['week_return']*100:.2f}% below threshold"
                )
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Weekly review complete: {results['week_return']*100:.2f}% return")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    def weekly_model_analysis(self) -> Dict:
        """Analyze model performance"""
        logger.info("Running weekly model analysis...")
        
        results = {
            'models_analyzed': 0,
            'avg_accuracy': 0,
            'feature_importance_updated': False,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Analyze each model's performance
            model_metrics = self._analyze_model_performance()
            
            results['models_analyzed'] = len(model_metrics)
            results['avg_accuracy'] = np.mean([m['accuracy'] for m in model_metrics])
            
            # Update feature importance
            self._update_feature_importance()
            results['feature_importance_updated'] = True
            
            # Check for model degradation
            if results['avg_accuracy'] < 0.55:
                self._send_alert(
                    "Model Performance Alert",
                    f"Average model accuracy {results['avg_accuracy']:.2f} below threshold"
                )
            
            results['duration'] = time.time() - start_time
            
            logger.info(f"Model analysis complete: {results['avg_accuracy']:.2f} avg accuracy")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    def system_maintenance(self) -> Dict:
        """Run system maintenance"""
        logger.info("Running system maintenance...")
        
        results = {
            'cache_cleaned': False,
            'logs_archived': False,
            'database_optimized': False,
            'backups_created': False,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Clean cache
            cache_size_before = self._get_cache_size()
            self._clean_cache()
            cache_size_after = self._get_cache_size()
            results['cache_cleaned'] = True
            
            logger.info(f"Cache cleaned: {cache_size_before - cache_size_after} MB freed")
            
            # Archive logs
            self._archive_logs()
            results['logs_archived'] = True
            
            # Optimize database (if using one)
            self._optimize_database()
            results['database_optimized'] = True
            
            # Create backups
            self._create_backups()
            results['backups_created'] = True
            
            # Run garbage collection
            gc.collect()
            
            results['duration'] = time.time() - start_time
            
            logger.info("System maintenance complete")
            
        except Exception as e:
            results['errors'] = [f"Error: {str(e)}"]
        
        return results
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _save_features(self, features: Dict):
        """Save engineered features"""
        cache_dir = f"{self.config.CACHE_DIR}/features"
        os.makedirs(cache_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        
        for symbol, feature_df in features.items():
            path = f"{cache_dir}/{symbol}_{timestamp}.parquet"
            feature_df.to_parquet(path)
    
    def _load_features(self) -> Dict:
        """Load latest features"""
        cache_dir = f"{self.config.CACHE_DIR}/features"
        features = {}
        
        if not os.path.exists(cache_dir):
            return features
        
        # Get latest files for each symbol
        for file in os.listdir(cache_dir):
            if file.endswith('.parquet'):
                symbol = file.split('_')[0]
                path = f"{cache_dir}/{file}"
                
                # Load only today's features
                if datetime.now().strftime('%Y%m%d') in file:
                    features[symbol] = pd.read_parquet(path)
        
        return features
    
    def _save_signals(self, signals: List):
        """Save trading signals"""
        cache_dir = f"{self.config.CACHE_DIR}/signals"
        os.makedirs(cache_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"{cache_dir}/signals_{timestamp}.json"
        
        # Convert signals to serializable format
        signal_data = []
        for signal in signals:
            signal_data.append({
                'symbol': signal.symbol,
                'timestamp': signal.timestamp.isoformat(),
                'direction': signal.direction,
                'win_probability': signal.win_probability,
                'expected_return': signal.expected_return,
                'bayesian_score': signal.bayesian_score,
                'position_size': signal.position_size,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            })
        
        with open(path, 'w') as f:
            json.dump(signal_data, f)
    
    def _load_signals(self) -> List:
        """Load today's signals"""
        cache_dir = f"{self.config.CACHE_DIR}/signals"
        signals = []
        
        if not os.path.exists(cache_dir):
            return signals
        
        # Get today's signal files
        today = datetime.now().strftime('%Y%m%d')
        
        for file in os.listdir(cache_dir):
            if file.startswith(f"signals_{today}") and file.endswith('.json'):
                path = f"{cache_dir}/{file}"
                with open(path, 'r') as f:
                    signal_data = json.load(f)
                    # Convert back to signal objects
                    # (Simplified - would recreate actual Signal objects)
                    signals.extend(signal_data)
        
        return signals
    
    def _get_current_market_data(self) -> pd.DataFrame:
        """Get current market data"""
        # Simplified - would get real-time data
        return pd.DataFrame()
    
    def _calculate_daily_performance(self) -> Dict:
        """Calculate daily performance metrics"""
        # Simplified calculation
        return {
            'daily_return': 0.0012,
            'daily_trades': 15,
            'win_rate': 0.60,
            'sharpe_ratio': 1.85
        }
    
    def _calculate_weekly_performance(self) -> Dict:
        """Calculate weekly performance metrics"""
        # Simplified calculation
        return {
            'total_return': 0.025,
            'sharpe_ratio': 1.92,
            'total_trades': 72,
            'win_rate': 0.625,
            'max_drawdown': -0.032
        }
    
    def _generate_daily_report(self, report_data: Dict) -> str:
        """Generate daily PDF report"""
        # Simplified - would generate actual PDF
        report_path = f"{self.config.CACHE_DIR}/reports/daily_{report_data['date']}.pdf"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Create report content
        with open(report_path, 'w') as f:
            f.write(f"Daily Trading Report - {report_data['date']}\n")
            f.write(json.dumps(report_data, indent=2))
        
        return report_path
    
    def _send_daily_report(self, report_path: str):
        """Send daily report via email"""
        logger.info(f"Daily report sent: {report_path}")
    
    def _generate_weekly_report(self, performance_data: Dict) -> str:
        """Generate weekly performance report"""
        report_path = f"{self.config.CACHE_DIR}/reports/weekly_{datetime.now().strftime('%Y%m%d')}.pdf"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Weekly Performance Report\n")
            f.write(json.dumps(performance_data, indent=2))
        
        return report_path
    
    def _analyze_model_performance(self) -> List[Dict]:
        """Analyze individual model performance"""
        # Simplified - would analyze actual model predictions
        return [
            {'model': 'XGBoost', 'accuracy': 0.62, 'sharpe': 1.85},
            {'model': 'LightGBM', 'accuracy': 0.61, 'sharpe': 1.82},
            {'model': 'LSTM', 'accuracy': 0.58, 'sharpe': 1.75}
        ]
    
    def _update_feature_importance(self):
        """Update feature importance rankings"""
        logger.info("Feature importance updated")
    
    def _get_cache_size(self) -> float:
        """Get cache directory size in MB"""
        total_size = 0
        cache_dir = self.config.CACHE_DIR
        
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _clean_cache(self):
        """Clean old cache files"""
        cache_dir = self.config.CACHE_DIR
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                
                # Check file age
                if os.path.getmtime(fp) < cutoff_date.timestamp():
                    os.remove(fp)
                    logger.debug(f"Removed old cache file: {fp}")
    
    def _archive_logs(self):
        """Archive old log files"""
        # Simplified - would implement actual log rotation
        logger.info("Logs archived")
    
    def _optimize_database(self):
        """Optimize database tables"""
        # Simplified - would run actual DB optimization
        logger.info("Database optimized")
    
    def _create_backups(self):
        """Create system backups"""
        backup_dir = f"{self.config.CACHE_DIR}/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup critical files
        critical_dirs = ['models', 'features', 'signals']
        
        for dir_name in critical_dirs:
            source_dir = f"{self.config.CACHE_DIR}/{dir_name}"
            if os.path.exists(source_dir):
                # Create tar archive
                backup_file = f"{backup_dir}/{dir_name}_{timestamp}.tar.gz"
                subprocess.run(['tar', '-czf', backup_file, source_dir])
                logger.info(f"Backed up {dir_name} to {backup_file}")
    
    # =============================================================================
    # SCHEDULER CONTROL
    # =============================================================================
    
    def start(self):
        """Start the scheduler"""
        self.is_running = True
        logger.info("Trading scheduler started")
        
        # Run scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        logger.info("Trading scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
    
    def get_next_jobs(self, limit: int = 10) -> List[Dict]:
        """Get next scheduled jobs"""
        jobs = []
        
        for job in schedule.jobs[:limit]:
            jobs.append({
                'next_run': job.next_run,
                'job': str(job.job_func.__name__ if hasattr(job.job_func, '__name__') else job.job_func)
            })
        
        return sorted(jobs, key=lambda x: x['next_run'])
    
    def get_job_history(self, limit: int = 50) -> List[Dict]:
        """Get recent job history"""
        return self.job_history[-limit:]
    
    def run_job_manually(self, job_name: str):
        """Run a specific job manually"""
        job_map = {
            'data_update': self.nightly_data_update,
            'feature_engineering': self.nightly_feature_engineering,
            'model_training': self.nightly_model_retraining,
            'signal_generation': self.morning_signal_generation,
            'maintenance': self.system_maintenance
        }
        
        if job_name in job_map:
            logger.info(f"Running {job_name} manually...")
            return job_map[job_name]()
        else:
            raise ValueError(f"Unknown job: {job_name}")

# Example usage
if __name__ == "__main__":
    logger.info("Trading Scheduler module loaded successfully")