# ml_trading_system_main.py

"""
ML Trading System - Complete System Integration
Professional hedge fund-quality trading system with all components integrated
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import signal
import psutil
import traceback

# Import all system components
from ml_trading_core import (
    TradingConfig, WatchlistManager, DataPipeline, 
    EnhancedFeatureEngineer, initialize_trading_system
)
from ml_trading_models import (
    EnsembleModelManager, TrainingOrchestrator, 
    PredictionEngine
)
from ml_trading_signals import (
    SignalGenerator, MarketRegimeDetector, 
    RiskManager, PositionTracker
)
from ml_trading_backtest import (
    BacktestingEngine, BacktestVisualizer, ExecutionModel
)
from ml_trading_execution import (
    AlpacaTradingClient, OrderManager, 
    PaperTradingSimulator, LiveTradingCoordinator
)
from ml_trading_sentiment import (
    NewsSentimentIntegration, RealTimeNewsMonitor
)
from ml_trading_dashboard import TradingDashboard
from ml_trading_automation import TradingScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN TRADING SYSTEM
# =============================================================================

class MLTradingSystem:
    """Complete ML Trading System Integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete trading system"""
        logger.info("=" * 80)
        logger.info("Initializing ML Trading System")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # System state
        self.is_running = False
        self.components = {}
        self.mode = 'paper'  # paper or live
        
        # Initialize all components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("ML Trading System initialized successfully")
    
    def _load_configuration(self, config_path: str) -> TradingConfig:
        """Load system configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Override default config with loaded values
            config = TradingConfig()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return TradingConfig()
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        try:
            # Core components
            logger.info("Initializing core components...")
            self.components['config'] = self.config
            self.components['watchlist'] = WatchlistManager()
            self.components['data_pipeline'] = DataPipeline(self.config)
            self.components['feature_engineer'] = EnhancedFeatureEngineer(self.config)
            
            # ML components
            logger.info("Initializing ML components...")
            self.components['ensemble_manager'] = EnsembleModelManager(
                self.config, 
                device=self.config.DEVICE
            )
            self.components['training_orchestrator'] = TrainingOrchestrator(
                self.config,
                self.components['feature_engineer'],
                self.components['ensemble_manager']
            )
            self.components['prediction_engine'] = PredictionEngine(
                self.components['ensemble_manager']
            )
            
            # Trading components
            logger.info("Initializing trading components...")
            self.components['market_regime'] = MarketRegimeDetector(self.config)
            self.components['signal_generator'] = SignalGenerator(
                self.config,
                self.components['feature_engineer'],
                self.components['prediction_engine']
            )
            self.components['risk_manager'] = RiskManager(self.config)
            self.components['position_tracker'] = PositionTracker()
            
            # Execution components
            logger.info("Initializing execution components...")
            if self.mode == 'paper':
                self.components['trading_client'] = PaperTradingSimulator(
                    initial_capital=1000000
                )
            else:
                self.components['trading_client'] = AlpacaTradingClient(self.config)
            
            self.components['order_manager'] = OrderManager(
                self.components['trading_client'],
                self.components['risk_manager']
            )
            
            self.components['trading_coordinator'] = LiveTradingCoordinator(
                self.config,
                self.components['signal_generator'],
                self.components['risk_manager'],
                use_paper_trading=(self.mode == 'paper')
            )
            
            # Analysis components
            logger.info("Initializing analysis components...")
            self.components['backtest_engine'] = BacktestingEngine(
                self.config,
                self.components['feature_engineer'],
                self.components['ensemble_manager'],
                self.components['signal_generator'],
                self.components['risk_manager']
            )
            
            # News sentiment
            logger.info("Initializing news sentiment...")
            if self.config.OPENAI_API_KEY:
                self.components['news_integration'] = NewsSentimentIntegration(self.config)
                self.components['news_monitor'] = RealTimeNewsMonitor(
                    self.components['news_integration'],
                    callback_func=self._handle_breaking_news
                )
            else:
                logger.warning("OpenAI API key not found - news sentiment disabled")
            
            # Automation
            logger.info("Initializing automation...")
            self.components['scheduler'] = TradingScheduler(self.components)
            
            # Dashboard
            logger.info("Initializing dashboard...")
            self.components['dashboard'] = TradingDashboard(self.components)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # =============================================================================
    # SYSTEM CONTROL
    # =============================================================================
    
    def start(self, mode: str = 'paper'):
        """Start the trading system"""
        logger.info(f"Starting ML Trading System in {mode} mode...")
        
        self.mode = mode
        self.is_running = True
        
        try:
            # Start scheduler
            self.components['scheduler'].start()
            logger.info("Scheduler started")
            
            # Start news monitoring
            if 'news_monitor' in self.components:
                asyncio.create_task(
                    self.components['news_monitor'].start_monitoring(
                        self.components['watchlist'].symbols[:20]
                    )
                )
                logger.info("News monitoring started")
            
            # System health check
            if self._system_health_check():
                logger.info("System health check passed")
            else:
                raise RuntimeError("System health check failed")
            
            # Log system status
            self._log_system_status()
            
            logger.info("ML Trading System started successfully")
            
            # If running interactively, start dashboard
            if self._is_interactive():
                self.run_dashboard()
            else:
                # Run in background
                self._run_background()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down ML Trading System...")
        
        self.is_running = False
        
        try:
            # Stop trading
            if 'trading_coordinator' in self.components:
                self.components['trading_coordinator'].stop_trading_session()
            
            # Stop scheduler
            if 'scheduler' in self.components:
                self.components['scheduler'].stop()
            
            # Stop news monitoring
            if 'news_monitor' in self.components:
                self.components['news_monitor'].stop_monitoring()
            
            # Save state
            self._save_system_state()
            
            logger.info("ML Trading System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _system_health_check(self) -> bool:
        """Perform system health check"""
        checks = {
            'memory': self._check_memory(),
            'disk': self._check_disk_space(),
            'gpu': self._check_gpu() if self.config.DEVICE == 'cuda' else True,
            'network': self._check_network(),
            'data': self._check_data_availability()
        }
        
        failed_checks = [name for name, status in checks.items() if not status]
        
        if failed_checks:
            logger.error(f"Health check failed: {', '.join(failed_checks)}")
            return False
        
        return True
    
    def _check_memory(self) -> bool:
        """Check available memory"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 4:  # Minimum 4GB
            logger.warning(f"Low memory: {available_gb:.1f}GB available")
            return False
        
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        available_gb = disk.free / (1024**3)
        
        if available_gb < 10:  # Minimum 10GB
            logger.warning(f"Low disk space: {available_gb:.1f}GB available")
            return False
        
        return True
    
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("GPU not available")
                return False
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 4:  # Minimum 4GB VRAM
                logger.warning(f"Low GPU memory: {gpu_memory:.1f}GB")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _check_network(self) -> bool:
        """Check network connectivity"""
        try:
            import requests
            response = requests.get('https://api.alpaca.markets/v2/clock', timeout=5)
            return response.status_code == 200 or response.status_code == 401
        except:
            logger.warning("Network connectivity check failed")
            return False
    
    def _check_data_availability(self) -> bool:
        """Check data availability"""
        try:
            # Test data fetch for a single symbol
            test_data = self.components['data_pipeline'].fetch_data(['SPY'])
            return not test_data.empty
        except:
            logger.warning("Data availability check failed")
            return False
    
    def _log_system_status(self):
        """Log current system status"""
        status = {
            'mode': self.mode,
            'components': list(self.components.keys()),
            'watchlist_size': len(self.components['watchlist'].symbols),
            'gpu_device': self.config.DEVICE,
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'cpu_usage': f"{psutil.cpu_percent()}%"
        }
        
        logger.info(f"System Status: {json.dumps(status, indent=2)}")
    
    def _is_interactive(self) -> bool:
        """Check if running in interactive mode"""
        return sys.stdin.isatty()
    
    def _save_system_state(self):
        """Save system state for recovery"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'positions': self._get_current_positions(),
            'performance': self._get_performance_summary()
        }
        
        state_file = f"{self.config.CACHE_DIR}/system_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"System state saved to {state_file}")
    
    def _get_current_positions(self) -> Dict:
        """Get current positions"""
        if 'trading_client' in self.components:
            return self.components['trading_client'].get_positions()
        return {}
    
    def _get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if 'trading_client' in self.components:
            if hasattr(self.components['trading_client'], 'get_performance_metrics'):
                return self.components['trading_client'].get_performance_metrics()
        return {}
    
    # =============================================================================
    # OPERATIONAL METHODS
    # =============================================================================
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        logger.info("Starting dashboard...")
        
        # Save dashboard script
        dashboard_script = """
import sys
sys.path.append('.')
from ml_trading_dashboard import main
main()
"""
        
        with open('run_dashboard.py', 'w') as f:
            f.write(dashboard_script)
        
        # Run Streamlit
        os.system('streamlit run run_dashboard.py')
    
    def _run_background(self):
        """Run system in background mode"""
        logger.info("Running in background mode...")
        
        while self.is_running:
            try:
                # Sleep and let scheduler handle everything
                asyncio.run(asyncio.sleep(60))
                
                # Periodic status log
                if datetime.now().minute == 0:  # Every hour
                    self._log_system_status()
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Background loop error: {e}")
    
    def _handle_breaking_news(self, news_item: Dict):
        """Handle breaking news callback"""
        logger.info(f"Breaking news for {news_item['symbol']}: {news_item['headline']}")
        
        # Check if we have a position
        positions = self.components['trading_client'].get_positions()
        
        if news_item['symbol'] in positions:
            # High impact negative news - consider closing position
            if news_item['sentiment'] < -0.5 and news_item['impact'] == 'high':
                logger.warning(f"High impact negative news for {news_item['symbol']}")
                # Could trigger position close or tighten stops
    
    # =============================================================================
    # MANUAL OPERATIONS
    # =============================================================================
    
    def train_models(self, symbols: List[str] = None):
        """Manually train models"""
        logger.info("Manual model training initiated...")
        
        if symbols is None:
            symbols = self.components['watchlist'].symbols[:20]
        
        results = {}
        
        for symbol in symbols:
            try:
                # Get features
                market_data = self.components['data_pipeline'].fetch_data([symbol])
                features = self.components['feature_engineer'].engineer_features(
                    market_data, symbol
                )
                
                if not features.empty:
                    # Train ensemble
                    train_results = self.components['training_orchestrator'].train_ensemble(
                        features, symbol
                    )
                    results[symbol] = train_results
                    logger.info(f"Trained models for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to train {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def run_backtest(self, start_date: str, end_date: str, 
                     symbols: List[str] = None) -> Dict:
        """Run manual backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        if symbols is None:
            symbols = self.components['watchlist'].symbols[:30]
        
        # Get data
        market_data = self.components['data_pipeline'].fetch_data(symbols)
        
        # Run backtest
        results = self.components['backtest_engine'].run_backtest(
            market_data, start_date, end_date
        )
        
        # Generate visualization
        BacktestVisualizer.plot_equity_curve(
            results['equity_curve'],
            save_path=f"{self.config.CACHE_DIR}/backtest_equity.png"
        )
        
        return results
    
    def generate_signals(self, symbols: List[str] = None) -> List:
        """Manually generate trading signals"""
        logger.info("Generating trading signals...")
        
        if symbols is None:
            symbols = self.components['watchlist'].symbols[:50]
        
        # Get market data
        market_data = self.components['data_pipeline'].fetch_data(symbols)
        
        # Generate predictions
        predictions = {}
        for symbol in symbols:
            try:
                features = self.components['feature_engineer'].engineer_features(
                    market_data, symbol
                )
                if not features.empty:
                    pred = self.components['prediction_engine'].predict(features)
                    predictions[symbol] = pred
            except Exception as e:
                logger.error(f"Failed to predict {symbol}: {e}")
        
        # Generate signals
        signals = self.components['signal_generator'].generate_signals(
            market_data, predictions
        )
        
        logger.info(f"Generated {len(signals)} signals")
        
        return signals
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        metrics = {
            'system': {
                'uptime': self._get_uptime(),
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent
            },
            'trading': self._get_performance_summary(),
            'models': self._get_model_metrics(),
            'scheduler': self._get_scheduler_metrics()
        }
        
        return metrics
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        # Simplified - would track actual start time
        return "0d 0h 0m"
    
    def _get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        # Simplified - would get actual metrics
        return {
            'avg_accuracy': 0.62,
            'avg_sharpe': 1.85,
            'models_active': 6
        }
    
    def _get_scheduler_metrics(self) -> Dict:
        """Get scheduler metrics"""
        scheduler = self.components['scheduler']
        
        return {
            'jobs_scheduled': len(scheduler.scheduled_jobs),
            'jobs_completed': len([j for j in scheduler.job_history if j['status'] == 'success']),
            'jobs_failed': len([j for j in scheduler.job_history if j['status'] == 'failed']),
            'next_jobs': scheduler.get_next_jobs(5)
        }

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML Trading System')
    
    parser.add_argument('command', choices=['start', 'train', 'backtest', 'signals', 'dashboard'],
                       help='Command to execute')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--symbols', nargs='+',
                       help='Symbols to process')
    parser.add_argument('--start-date', type=str,
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create system instance
    system = MLTradingSystem(config_path=args.config)
    
    try:
        if args.command == 'start':
            system.start(mode=args.mode)
            
        elif args.command == 'train':
            results = system.train_models(symbols=args.symbols)
            print(json.dumps(results, indent=2))
            
        elif args.command == 'backtest':
            if not args.start_date or not args.end_date:
                print("Error: --start-date and --end-date required for backtest")
                sys.exit(1)
                
            results = system.run_backtest(
                args.start_date, 
                args.end_date,
                symbols=args.symbols
            )
            
            print(f"Backtest Results:")
            print(f"Total Return: {results['performance_metrics']['total_return']*100:.2f}%")
            print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']*100:.2f}%")
            
        elif args.command == 'signals':
            signals = system.generate_signals(symbols=args.symbols)
            
            print(f"Generated {len(signals)} signals:")
            for signal in signals[:10]:  # Show first 10
                print(f"  {signal.symbol}: {signal.direction} "
                     f"(confidence: {signal.confidence_score:.2f}, "
                     f"score: {signal.bayesian_score:.2f})")
                     
        elif args.command == 'dashboard':
            system.run_dashboard()
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()