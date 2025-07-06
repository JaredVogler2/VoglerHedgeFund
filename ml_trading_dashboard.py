# ml_trading_dashboard.py

"""
ML Trading System - Professional Dashboard Interface
Real-time monitoring, analytics, and control center
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging
import json

# Configure Streamlit
st.set_page_config(
    page_title="ML Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #464646;
    }
    .positive {
        color: #00ff41;
    }
    .negative {
        color: #ff3131;
    }
    .neutral {
        color: #ffa500;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

class TradingDashboard:
    """Main dashboard class"""
    
    def __init__(self, system_components: Dict):
        self.data_pipeline = system_components.get('data_pipeline')
        self.signal_generator = system_components.get('signal_generator')
        self.risk_manager = system_components.get('risk_manager')
        self.trading_client = system_components.get('trading_client')
        self.backtest_engine = system_components.get('backtest_engine')
        self.news_integration = system_components.get('news_integration')
        
        # Dashboard state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = []
        
    def run(self):
        """Run the main dashboard"""
        # Header
        self.render_header()
        
        # Sidebar
        with st.sidebar:
            self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('page', 'Overview')
        
        if page == 'Overview':
            self.render_overview_page()
        elif page == 'Portfolio':
            self.render_portfolio_page()
        elif page == 'Predictions':
            self.render_predictions_page()
        elif page == 'ML Analytics':
            self.render_ml_analytics_page()
        elif page == 'Trade History':
            self.render_trade_history_page()
        elif page == 'Market Analysis':
            self.render_market_analysis_page()
        elif page == 'Risk Management':
            self.render_risk_management_page()
        elif page == 'Backtesting':
            self.render_backtesting_page()
        elif page == 'Settings':
            self.render_settings_page()
        
        # Auto-refresh
        if st.session_state.get('auto_refresh', False):
            time.sleep(10)
            st.rerun()
    
    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.title("🤖 ML Trading System")
            
        with col2:
            # Market status
            is_open = self.is_market_open()
            status_color = "🟢" if is_open else "🔴"
            st.metric("Market Status", f"{status_color} {'Open' if is_open else 'Closed'}")
            
        with col3:
            # Last update time
            time_diff = (datetime.now() - st.session_state.last_update).seconds
            st.metric("Last Update", f"{time_diff}s ago")
            
        with col4:
            # Refresh button
            if st.button("🔄 Refresh"):
                st.session_state.last_update = datetime.now()
                st.rerun()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.header("Navigation")
        
        # Page selection
        pages = [
            "Overview",
            "Portfolio",
            "Predictions", 
            "ML Analytics",
            "Trade History",
            "Market Analysis",
            "Risk Management",
            "Backtesting",
            "Settings"
        ]
        
        st.session_state.page = st.selectbox("Select Page", pages)
        
        st.divider()
        
        # Quick actions
        st.header("Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Trading"):
                self.start_trading_session()
        with col2:
            if st.button("⏸️ Stop Trading"):
                self.stop_trading_session()
        
        # System status
        st.divider()
        st.header("System Status")
        
        # Check component status
        components = {
            "Data Pipeline": self.data_pipeline is not None,
            "Signal Generator": self.signal_generator is not None,
            "Risk Manager": self.risk_manager is not None,
            "Trading Client": self.trading_client is not None,
            "News Analysis": self.news_integration is not None
        }
        
        for name, status in components.items():
            color = "🟢" if status else "🔴"
            st.write(f"{color} {name}")
        
        # Auto-refresh toggle
        st.divider()
        st.session_state.auto_refresh = st.checkbox("Auto-refresh (10s)", 
                                                   value=st.session_state.get('auto_refresh', False))
    
    # =============================================================================
    # OVERVIEW PAGE
    # =============================================================================
    
    def render_overview_page(self):
        """Render main overview page"""
        # Key metrics row
        st.header("Portfolio Performance")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Get portfolio metrics
        metrics = self.get_portfolio_metrics()
        
        with col1:
            self.render_metric_card(
                "Total Value",
                f"${metrics['total_value']:,.2f}",
                f"{metrics['total_return_pct']:.2f}%"
            )
            
        with col2:
            self.render_metric_card(
                "Day P&L",
                f"${metrics['day_pnl']:,.2f}",
                f"{metrics['day_pnl_pct']:.2f}%"
            )
            
        with col3:
            self.render_metric_card(
                "Open Positions",
                f"{metrics['open_positions']}",
                f"{metrics['win_rate']:.1f}% win rate"
            )
            
        with col4:
            self.render_metric_card(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                "Last 30 days"
            )
            
        with col5:
            self.render_metric_card(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1f}%",
                f"{metrics['current_drawdown']:.1f}% current"
            )
            
        with col6:
            self.render_metric_card(
                "Daily Volume",
                f"{metrics['daily_trades']}",
                "trades today"
            )
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Equity Curve")
            self.render_equity_chart()
            
        with col2:
            st.subheader("Position Distribution")
            self.render_position_distribution()
        
        # Active positions
        st.header("Active Positions")
        self.render_positions_table()
        
        # Recent signals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Signals")
            self.render_recent_signals()
            
        with col2:
            st.subheader("Market Conditions")
            self.render_market_conditions()
    
    def render_metric_card(self, title: str, value: str, subtitle: str):
        """Render a metric card"""
        # Determine color based on value
        if '+' in subtitle or (subtitle.replace('.', '').replace('-', '').replace('%', '').isdigit() and float(subtitle.replace('%', '')) > 0):
            color_class = "positive"
        elif '-' in subtitle:
            color_class = "negative"
        else:
            color_class = "neutral"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #888;">{title}</div>
            <div style="font-size: 24px; font-weight: bold;">{value}</div>
            <div style="font-size: 12px;" class="{color_class}">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_equity_chart(self):
        """Render equity curve chart"""
        # Get equity data
        equity_data = self.get_equity_curve_data()
        
        if equity_data.empty:
            st.info("No equity data available")
            return
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_data.index,
                y=equity_data['equity'],
                name='Portfolio Value',
                line=dict(color='#00ff41', width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark (S&P 500)
        if 'benchmark' in equity_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data['benchmark'],
                    name='S&P 500',
                    line=dict(color='#888', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=equity_data.index,
                y=equity_data['drawdown'],
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#ff3131', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            showlegend=True,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_position_distribution(self):
        """Render position distribution pie chart"""
        positions = self.get_current_positions()
        
        if not positions:
            st.info("No open positions")
            return
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[p['symbol'] for p in positions],
            values=[p['market_value'] for p in positions],
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set3
            )
        )])
        
        fig.update_layout(
            template='plotly_dark',
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_positions_table(self):
        """Render positions table"""
        positions = self.get_current_positions()
        
        if not positions:
            st.info("No open positions")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Format columns
        df['P&L'] = df.apply(lambda x: f"${x['unrealized_pnl']:,.2f} ({x['unrealized_pnl_pct']:.2f}%)", axis=1)
        df['Entry'] = df['entry_price'].apply(lambda x: f"${x:.2f}")
        df['Current'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        df['Value'] = df['market_value'].apply(lambda x: f"${x:,.2f}")
        
        # Select columns to display
        display_cols = ['symbol', 'quantity', 'Entry', 'Current', 'Value', 'P&L', 'stop_loss', 'take_profit']
        
        # Apply styling
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )
    
    # =============================================================================
    # PORTFOLIO PAGE
    # =============================================================================
    
    def render_portfolio_page(self):
        """Render detailed portfolio analysis"""
        st.header("Portfolio Analysis")
        
        # Portfolio summary
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Portfolio Allocation")
            self.render_allocation_chart()
            
        with col2:
            st.subheader("Risk Metrics")
            self.render_risk_metrics()
        
        # Performance analysis
        st.header("Performance Analysis")
        
        # Time period selector
        period = st.selectbox("Time Period", ["1D", "1W", "1M", "3M", "YTD", "1Y", "All"])
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Returns Distribution")
            self.render_returns_distribution(period)
            
        with col2:
            st.subheader("Rolling Performance")
            self.render_rolling_performance(period)
        
        # Correlation matrix
        st.subheader("Position Correlations")
        self.render_correlation_matrix()
        
        # Sector exposure
        st.subheader("Sector Exposure")
        self.render_sector_exposure()
    
    def render_allocation_chart(self):
        """Render portfolio allocation chart"""
        allocations = self.get_portfolio_allocations()
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=allocations['labels'],
            parents=allocations['parents'],
            values=allocations['values'],
            branchvalues="total",
            marker=dict(
                colors=allocations['colors'],
                line=dict(color='#000', width=2)
            )
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_metrics(self):
        """Render risk metrics table"""
        metrics = self.calculate_risk_metrics()
        
        # Create metrics grid
        for i in range(0, len(metrics), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(metrics):
                    metric = metrics[i + j]
                    col.metric(
                        metric['name'],
                        metric['value'],
                        metric['change']
                    )
    
    # =============================================================================
    # PREDICTIONS PAGE
    # =============================================================================
    
    def render_predictions_page(self):
        """Render ML predictions page"""
        st.header("ML Predictions & Signals")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_filter = st.selectbox("Signal Type", ["All", "Long", "Short"])
            
        with col2:
            confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.6)
            
        with col3:
            sort_by = st.selectbox("Sort By", ["Bayesian Score", "Confidence", "Expected Return"])
        
        # Get predictions
        predictions = self.get_latest_predictions(
            signal_filter=signal_filter,
            confidence_threshold=confidence_threshold,
            sort_by=sort_by
        )
        
        # Display predictions
        if predictions:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Signals", len(predictions))
            with col2:
                long_count = sum(1 for p in predictions if p['direction'] == 'long')
                st.metric("Long Signals", long_count)
            with col3:
                short_count = sum(1 for p in predictions if p['direction'] == 'short')
                st.metric("Short Signals", short_count)
            with col4:
                avg_confidence = np.mean([p['confidence'] for p in predictions])
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Predictions table
            st.subheader("Trading Signals")
            
            # Convert to DataFrame for display
            df = pd.DataFrame(predictions)
            
            # Add action buttons
            for idx, row in df.iterrows():
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
                
                with col1:
                    st.write(row['symbol'])
                with col2:
                    color = "🟢" if row['direction'] == 'long' else "🔴"
                    st.write(f"{color} {row['direction'].upper()}")
                with col3:
                    st.write(f"{row['confidence']:.2f}")
                with col4:
                    st.write(f"{row['expected_return']*100:.2f}%")
                with col5:
                    st.write(f"{row['bayesian_score']:.2f}")
                with col6:
                    st.write(f"${row['current_price']:.2f}")
                with col7:
                    if st.button("Details", key=f"details_{idx}"):
                        self.show_prediction_details(row)
                with col8:
                    if st.button("Trade", key=f"trade_{idx}"):
                        self.execute_prediction(row)
        else:
            st.info("No predictions match the selected criteria")
        
        # Model confidence chart
        st.subheader("Model Agreement Analysis")
        self.render_model_agreement_chart()
    
    # =============================================================================
    # ML ANALYTICS PAGE
    # =============================================================================
    
    def render_ml_analytics_page(self):
        """Render ML model analytics"""
        st.header("Machine Learning Analytics")
        
        # Model selection
        model_type = st.selectbox("Select Model", 
                                ["Ensemble", "XGBoost", "LightGBM", "CatBoost", 
                                 "Attention LSTM", "CNN-LSTM", "Transformer"])
        
        # Model performance metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Performance")
            self.render_model_performance(model_type)
            
        with col2:
            st.subheader("Feature Importance")
            self.render_feature_importance(model_type)
        
        # Prediction accuracy over time
        st.subheader("Prediction Accuracy Trends")
        self.render_accuracy_trends()
        
        # Model diagnostics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calibration Plot")
            self.render_calibration_plot(model_type)
            
        with col2:
            st.subheader("Confusion Matrix")
            self.render_confusion_matrix(model_type)
        
        # Feature analysis
        st.header("Feature Analysis")
        
        # Feature correlation
        st.subheader("Feature Correlations")
        self.render_feature_correlations()
        
        # Feature evolution
        st.subheader("Feature Importance Evolution")
        self.render_feature_evolution()
    
    # =============================================================================
    # MARKET ANALYSIS PAGE
    # =============================================================================
    
    def render_market_analysis_page(self):
        """Render market analysis page"""
        st.header("Market Analysis")
        
        # Market overview
        col1, col2, col3, col4 = st.columns(4)
        
        market_data = self.get_market_overview()
        
        with col1:
            self.render_market_metric("S&P 500", market_data['spy'])
        with col2:
            self.render_market_metric("NASDAQ", market_data['qqq'])
        with col3:
            self.render_market_metric("VIX", market_data['vix'])
        with col4:
            self.render_market_metric("Dollar Index", market_data['dxy'])
        
        # Market breadth
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Breadth")
            self.render_market_breadth()
            
        with col2:
            st.subheader("Sector Performance")
            self.render_sector_performance()
        
        # Market regime
        st.subheader("Market Regime Analysis")
        self.render_market_regime()
        
        # News sentiment
        st.header("News Sentiment Analysis")
        self.render_news_sentiment()
        
        # Correlation analysis
        st.subheader("Asset Correlations")
        self.render_asset_correlations()
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        # This would connect to actual trading system
        # Returning mock data for demonstration
        return {
            'total_value': 1025000,
            'total_return_pct': 2.5,
            'day_pnl': 5200,
            'day_pnl_pct': 0.51,
            'open_positions': 8,
            'win_rate': 62.5,
            'sharpe_ratio': 1.85,
            'max_drawdown': -8.3,
            'current_drawdown': -2.1,
            'daily_trades': 12
        }
    
    def get_equity_curve_data(self) -> pd.DataFrame:
        """Get equity curve data"""
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Simulate equity curve
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        equity = 1000000 * (1 + returns).cumprod()
        
        # Calculate drawdown
        running_max = pd.Series(equity).expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        # Benchmark
        benchmark_returns = np.random.normal(0.0003, 0.015, len(dates))
        benchmark = 1000000 * (1 + benchmark_returns).cumprod()
        
        return pd.DataFrame({
            'equity': equity,
            'drawdown': drawdown,
            'benchmark': benchmark
        }, index=dates)
    
    def get_current_positions(self) -> List[Dict]:
        """Get current positions"""
        # Mock data
        positions = [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 175.50,
                'current_price': 178.25,
                'market_value': 17825,
                'unrealized_pnl': 275,
                'unrealized_pnl_pct': 1.57,
                'stop_loss': 171.50,
                'take_profit': 182.00
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'entry_price': 140.25,
                'current_price': 139.50,
                'market_value': 6975,
                'unrealized_pnl': -37.50,
                'unrealized_pnl_pct': -0.53,
                'stop_loss': 137.00,
                'take_profit': 145.00
            },
            # Add more positions...
        ]
        
        return positions
    
    def get_latest_predictions(self, signal_filter: str = "All",
                             confidence_threshold: float = 0.6,
                             sort_by: str = "Bayesian Score") -> List[Dict]:
        """Get latest ML predictions"""
        # Mock predictions
        predictions = [
            {
                'symbol': 'NVDA',
                'direction': 'long',
                'confidence': 0.82,
                'expected_return': 0.025,
                'bayesian_score': 2.1,
                'current_price': 880.50,
                'predicted_volatility': 0.028,
                'model_agreement': 0.85
            },
            {
                'symbol': 'TSLA',
                'direction': 'short',
                'confidence': 0.75,
                'expected_return': -0.018,
                'bayesian_score': -1.8,
                'current_price': 245.20,
                'predicted_volatility': 0.035,
                'model_agreement': 0.72
            },
            # Add more predictions...
        ]
        
        # Filter
        if signal_filter != "All":
            predictions = [p for p in predictions if p['direction'] == signal_filter.lower()]
        
        predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
        
        # Sort
        sort_key = {
            "Bayesian Score": lambda x: abs(x['bayesian_score']),
            "Confidence": lambda x: x['confidence'],
            "Expected Return": lambda x: abs(x['expected_return'])
        }
        
        predictions.sort(key=sort_key[sort_by], reverse=True)
        
        return predictions
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        # Simple check for US market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        return market_open <= now <= market_close and now.weekday() < 5
    
    def start_trading_session(self):
        """Start automated trading session"""
        st.success("Trading session started!")
        # Would trigger actual trading system
        
    def stop_trading_session(self):
        """Stop automated trading session"""
        st.warning("Trading session stopped!")
        # Would stop actual trading system

# =============================================================================
# ADDITIONAL DASHBOARD PAGES
# =============================================================================

def render_trade_history_page():
    """Render trade history page"""
    st.header("Trade History")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol_filter = st.text_input("Symbol Filter")
    with col2:
        side_filter = st.selectbox("Side", ["All", "Long", "Short"])
    with col3:
        result_filter = st.selectbox("Result", ["All", "Winners", "Losers"])
    
    # Get trade history
    trades = get_trade_history(start_date, end_date, symbol_filter, side_filter, result_filter)
    
    if trades:
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trades)
        winners = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winners) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) or 0
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Total P&L", f"${total_pnl:,.2f}")
        with col4:
            profit_factor = abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in trades if t['pnl'] < 0)) if avg_loss != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # Trade details table
        st.subheader("Trade Details")
        df = pd.DataFrame(trades)
        
        # Format columns
        df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        df['exit_time'] = pd.to_datetime(df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        df['pnl_formatted'] = df['pnl'].apply(lambda x: f"${x:,.2f}")
        df['return_pct'] = df['return_pct'].apply(lambda x: f"{x:.2f}%")
        
        # Display table
        st.dataframe(
            df[['symbol', 'side', 'entry_time', 'exit_time', 'entry_price', 
                'exit_price', 'quantity', 'pnl_formatted', 'return_pct']],
            use_container_width=True,
            hide_index=True
        )
        
        # Trade analysis charts
        st.subheader("Trade Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L distribution
            fig = px.histogram(df, x='pnl', nbins=30, title='P&L Distribution')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Cumulative P&L
            df['cumulative_pnl'] = df['pnl'].cumsum()
            fig = px.line(df, x='exit_time', y='cumulative_pnl', title='Cumulative P&L')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades found for the selected criteria")

def get_trade_history(start_date, end_date, symbol_filter, side_filter, result_filter):
    """Get trade history (mock data)"""
    # Generate sample trades
    trades = []
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN']
    
    for i in range(50):
        entry_time = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        exit_time = entry_time + timedelta(hours=np.random.randint(1, 48))
        
        symbol = np.random.choice(symbols)
        side = np.random.choice(['long', 'short'])
        quantity = np.random.randint(10, 200)
        
        entry_price = np.random.uniform(100, 500)
        
        # Simulate realistic P&L
        if np.random.random() > 0.45:  # 55% win rate
            return_pct = np.random.uniform(0.5, 3.0)
        else:
            return_pct = np.random.uniform(-2.0, -0.5)
        
        if side == 'long':
            exit_price = entry_price * (1 + return_pct / 100)
            pnl = (exit_price - entry_price) * quantity
        else:
            exit_price = entry_price * (1 - return_pct / 100)
            pnl = (entry_price - exit_price) * quantity
        
        trades.append({
            'symbol': symbol,
            'side': side,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': return_pct
        })
    
    # Apply filters
    if symbol_filter:
        trades = [t for t in trades if symbol_filter.upper() in t['symbol']]
    
    if side_filter != "All":
        trades = [t for t in trades if t['side'] == side_filter.lower()]
    
    if result_filter == "Winners":
        trades = [t for t in trades if t['pnl'] > 0]
    elif result_filter == "Losers":
        trades = [t for t in trades if t['pnl'] < 0]
    
    return trades

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point"""
    # Initialize system components (mock for demo)
    system_components = {
        'data_pipeline': True,  # Would be actual component instances
        'signal_generator': True,
        'risk_manager': True,
        'trading_client': True,
        'backtest_engine': True,
        'news_integration': True
    }
    
    # Create and run dashboard
    dashboard = TradingDashboard(system_components)
    dashboard.run()

if __name__ == "__main__":
    main()