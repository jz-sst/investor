
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import Dict, List, Any

# Import existing modules
from data_retrieval import DataRetrieval
from technical_analysis import TechnicalAnalysis
from fundamental_analysis import FundamentalAnalysis
from recommendation import RecommendationEngine
from ml_engine import MLEngine
from database import Database

# Set page config to match Grok's design
st.set_page_config(
    page_title="InvestTrack Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InvestTrackPro:
    def __init__(self):
        self.init_session_state()
        self.init_components()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        if 'user_portfolio' not in st.session_state:
            st.session_state.user_portfolio = self.get_sample_portfolio()
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['AAPL', 'NVDA', 'MSFT']
        if 'alerts' not in st.session_state:
            st.session_state.alerts = self.get_sample_alerts()
    
    def init_components(self):
        """Initialize analysis components"""
        self.db = Database()
        self.data_retrieval = DataRetrieval(self.db)
        self.technical_analysis = TechnicalAnalysis()
        self.fundamental_analysis = FundamentalAnalysis()
        self.ml_engine = MLEngine()
        self.recommendation_engine = RecommendationEngine(
            self.technical_analysis, 
            self.fundamental_analysis
        )
    
    def get_sample_portfolio(self):
        """Sample portfolio data"""
        return {
            'AAPL': {'qty': 220, 'avg_cost': 165.00, 'current_price': 198.44},
            'NVDA': {'qty': 60, 'avg_cost': 720.00, 'current_price': 890.20},
            'MSFT': {'qty': 140, 'avg_cost': 350.00, 'current_price': 421.10},
            'XLE': {'qty': 180, 'avg_cost': 86.00, 'current_price': 85.210}
        }
    
    def get_sample_alerts(self):
        """Sample alerts data"""
        return [
            {
                'alert': 'ROI below 5%',
                'project': 'Logistics Hub',
                'triggered': '2h ago',
                'severity': 'High',
                'status': 'Open'
            },
            {
                'alert': 'IRR under 10%',
                'project': 'EV Charging Lot 4',
                'triggered': '1d ago',
                'severity': 'Medium',
                'status': 'Acknowledged'
            },
            {
                'alert': 'Cashflow variance > 15%',
                'project': 'Solar Farm A',
                'triggered': '3d ago',
                'severity': 'Critical',
                'status': 'Resolved'
            }
        ]
    
    def get_grok_theme_css(self):
        """Grok's InvestTrack Pro theme CSS"""
        return """
        <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* CSS Variables matching Grok's theme */
        :root {
            --bg-primary: #0a192f;
            --bg-secondary: #1e293b;
            --accent-teal: #14b8a6;
            --accent-cyan: #06b6d4;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-radius: 0.5rem;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --font-family: 'Inter', sans-serif;
        }
        
        /* Global Streamlit overrides */
        .stApp {
            background: var(--bg-primary) !important;
            font-family: var(--font-family) !important;
            color: var(--text-primary) !important;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Sidebar styling */
        .css-1d391kg {
            background: var(--bg-primary) !important;
            border-right: 1px solid var(--bg-secondary) !important;
        }
        
        .css-1d391kg .css-17eq0hr {
            color: var(--text-primary) !important;
        }
        
        /* Main content styling */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
            background: var(--bg-primary) !important;
        }
        
        /* Card styling matching Grok's design */
        .investtrack-card {
            background: var(--bg-secondary) !important;
            padding: 1.5rem !important;
            border-radius: var(--border-radius) !important;
            box-shadow: var(--shadow) !important;
            margin-bottom: 1rem !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        .investtrack-card h1,
        .investtrack-card h2,
        .investtrack-card h3 {
            color: var(--text-primary) !important;
            font-family: var(--font-family) !important;
            margin-bottom: 1rem !important;
        }
        
        /* Header styling */
        .investtrack-header {
            background: var(--bg-secondary) !important;
            padding: 1rem 1.5rem !important;
            border-radius: var(--border-radius) !important;
            margin-bottom: 1.5rem !important;
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        .investtrack-header h1 {
            color: var(--text-primary) !important;
            margin: 0 !important;
            font-size: 1.75rem !important;
            font-weight: 600 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: var(--accent-teal) !important;
            color: var(--text-primary) !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: var(--border-radius) !important;
            font-family: var(--font-family) !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background: #0f9c8a !important;
            transform: translateY(-1px) !important;
        }
        
        /* Cyan button variant */
        .button-cyan button {
            background: var(--accent-cyan) !important;
        }
        
        .button-cyan button:hover {
            background: #0891b2 !important;
        }
        
        /* Metrics styling */
        .metric-container {
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
            gap: 1rem !important;
            margin-bottom: 1.5rem !important;
        }
        
        .metric-card {
            background: var(--bg-secondary) !important;
            padding: 1.5rem !important;
            border-radius: var(--border-radius) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            text-align: center !important;
        }
        
        .metric-label {
            font-size: 0.875rem !important;
            color: var(--text-secondary) !important;
            margin-bottom: 0.5rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        .metric-value {
            font-size: 1.875rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            margin-bottom: 0.25rem !important;
        }
        
        .metric-change {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }
        
        .metric-positive { color: var(--accent-teal) !important; }
        .metric-negative { color: #ef4444 !important; }
        
        /* Table styling */
        .stDataFrame > div {
            background: var(--bg-secondary) !important;
            border-radius: var(--border-radius) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        .stDataFrame table {
            color: var(--text-primary) !important;
        }
        
        .stDataFrame thead tr th {
            background: var(--bg-primary) !important;
            color: var(--text-secondary) !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        /* Sidebar navigation styling */
        .sidebar-nav {
            padding: 1rem 0 !important;
        }
        
        .nav-item {
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-secondary) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }
        
        .nav-item:hover {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        .nav-item.active {
            background: var(--accent-teal) !important;
            color: var(--bg-primary) !important;
        }
        
        /* User profile styling */
        .user-profile {
            position: fixed !important;
            bottom: 1rem !important;
            left: 1rem !important;
            right: 1rem !important;
            background: var(--bg-secondary) !important;
            padding: 1rem !important;
            border-radius: var(--border-radius) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.75rem !important;
        }
        
        .profile-avatar {
            width: 40px !important;
            height: 40px !important;
            background: var(--accent-teal) !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: var(--bg-primary) !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        .profile-info {
            flex: 1 !important;
        }
        
        .profile-name {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            margin: 0 !important;
        }
        
        .profile-tier {
            color: var(--text-secondary) !important;
            font-size: 0.75rem !important;
            margin: 0 !important;
        }
        
        /* Chart styling */
        .plotly-graph-div {
            background: transparent !important;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: var(--border-radius) !important;
        }
        
        .stSelectbox > div > div > select {
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: var(--border-radius) !important;
        }
        
        /* Status badges */
        .status-badge {
            padding: 0.25rem 0.5rem !important;
            border-radius: 0.25rem !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
        }
        
        .status-open { 
            background: rgba(239, 68, 68, 0.1) !important; 
            color: #ef4444 !important; 
        }
        .status-acknowledged { 
            background: rgba(245, 158, 11, 0.1) !important; 
            color: #f59e0b !important; 
        }
        .status-resolved { 
            background: rgba(20, 184, 166, 0.1) !important; 
            color: var(--accent-teal) !important; 
        }
        
        /* Grid layout */
        .investtrack-grid {
            display: grid !important;
            gap: 1rem !important;
        }
        
        .grid-2 { grid-template-columns: repeat(2, 1fr) !important; }
        .grid-3 { grid-template-columns: repeat(3, 1fr) !important; }
        .grid-4 { grid-template-columns: repeat(4, 1fr) !important; }
        
        @media (max-width: 768px) {
            .grid-2, .grid-3, .grid-4 { 
                grid-template-columns: 1fr !important; 
            }
        }
        </style>
        """
    
    def render_sidebar(self):
        """Render the sidebar navigation matching Grok's design"""
        with st.sidebar:
            # App title
            st.markdown("""
            <div style="padding: 1.5rem 0; border-bottom: 1px solid var(--bg-secondary); margin-bottom: 1rem;">
                <h2 style="color: var(--text-primary); margin: 0; font-weight: 700; font-size: 1.5rem;">InvestTrack Pro</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu
            pages = [
                ("Dashboard", "🏠"),
                ("Portfolio", "💼"),
                ("Stock Analysis", "📊"),
                ("Market Scanner", "🔍"),
                ("Opportunities", "💡"),
                ("Projects", "📁"),
                ("Alerts", "🚨"),
                ("Settings", "⚙️")
            ]
            
            st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
            
            for page, icon in pages:
                active_class = "active" if st.session_state.current_page == page else ""
                
                if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # User profile
            st.markdown("""
            <div class="user-profile">
                <div class="profile-avatar">AM</div>
                <div class="profile-info">
                    <div class="profile-name">Alex Morgan</div>
                    <div class="profile-tier">Premium</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_header(self, title, buttons=None):
        """Render page header matching Grok's design"""
        if buttons is None:
            buttons = []
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="investtrack-header">
                <h1>{title}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if buttons:
                for button in buttons:
                    st.button(**button)
    
    def render_metric_cards(self, metrics):
        """Render metric cards in grid layout"""
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        cols = st.columns(len(metrics))
        
        for i, (col, metric) in enumerate(zip(cols, metrics)):
            with col:
                # Handle change value properly - check if it's a string starting with + or -
                change_value = metric.get('change', '')
                if isinstance(change_value, str):
                    change_class = "metric-positive" if change_value.startswith('+') else "metric-negative"
                else:
                    change_class = "metric-positive" if change_value >= 0 else "metric-negative"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['label']}</div>
                    <div class="metric-value">{metric['value']}</div>
                    {f'<div class="metric-change {change_class}">{metric["change"]}</div>' if 'change' in metric else ''}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render the main dashboard"""
        self.render_header("Dashboard", [
            {"label": "Alerts", "key": "dash_alerts"},
            {"label": "+ Add", "key": "dash_add"}
        ])
        
        # Portfolio metrics
        metrics = [
            {"label": "Total Portfolio Value", "value": "$248,920", "change": "+1.0% Today"},
            {"label": "Daily P/L", "value": "+$4,320", "change": "Last 24h"},
            {"label": "YTD Return", "value": "+12.4%"},
            {"label": "Risk Score", "value": "Moderate"}
        ]
        
        self.render_metric_cards(metrics)
        
        # Main content
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Performance chart
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Performance Overview</h2>', unsafe_allow_html=True)
            
            # Sample performance data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            performance = np.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=performance * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='#14b8a6', width=2)
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent news
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Recent News</h2>', unsafe_allow_html=True)
            
            news_items = [
                {"title": "S&P 500 rises as tech leads gains", "source": "Reuters", "time": "5m ago"},
                {"title": "NASDAQ posts third straight day of gains", "source": "Bloomberg", "time": "12m ago"},
                {"title": "Energy stocks slip as oil retreats", "source": "WSJ", "time": "23m ago"}
            ]
            
            for item in news_items:
                st.markdown(f"""
                <div style="padding: 0.75rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <div style="color: var(--text-primary); font-weight: 500; margin-bottom: 0.25rem;">{item['title']}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem;">{item['source']} • {item['time']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            # Key indices
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Key Indices</h2>', unsafe_allow_html=True)
            
            indices = [
                {"name": "S&P 500", "value": "5,235.12", "change": "+0.8%"},
                {"name": "NASDAQ", "value": "16,120.45", "change": "+1.2%"},
                {"name": "Dow Jones", "value": "39,210.33", "change": "+0.6%"}
            ]
            
            for idx in indices:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: var(--text-primary);">{idx['name']}</span>
                    <div style="text-align: right;">
                        <span style="color: var(--text-primary);">{idx['value']}</span>
                        <span style="color: var(--accent-teal); margin-left: 0.5rem;">{idx['change']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alerts
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Alerts</h2>', unsafe_allow_html=True)
            
            alert_items = [
                {"text": "AAPL price alert", "condition": "> $200", "status": "Armed"},
                {"text": "Portfolio drop", "condition": "< -5% daily", "status": "Armed"},
                {"text": "NVDA RSI", "condition": "RSI < 30", "status": "Paused"}
            ]
            
            for alert in alert_items:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                    <div>
                        <div style="color: var(--text-primary); font-size: 0.875rem;">{alert['text']}</div>
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">{alert['condition']}</div>
                    </div>
                    <span class="status-badge status-open">{alert['status']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Watchlist
            self.render_watchlist_widget()
    
    def render_watchlist_widget(self):
        """Render watchlist widget"""
        st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
        st.markdown('<h2>Watchlist</h2>', unsafe_allow_html=True)
        
        watchlist_data = [
            {"ticker": "AAPL", "price": "$198.44", "change": "+2.1%", "signal": "Buy"},
            {"ticker": "NVDA", "price": "$1,012.33", "change": "+0.8%", "signal": "Hold"},
            {"ticker": "MSFT", "price": "$414.22", "change": "+1.5%", "signal": "Buy"}
        ]
        
        for item in watchlist_data:
            change_color = "var(--accent-teal)" if "+" in item['change'] else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                <div>
                    <span style="color: var(--text-primary); font-weight: 500;">{item['ticker']}</span>
                    <span style="color: var(--text-secondary); margin-left: 0.75rem;">{item['price']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {change_color}; font-size: 0.75rem;">{item['change']}</div>
                    <div style="color: var(--accent-teal); font-size: 0.75rem;">{item['signal']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_portfolio(self):
        """Render portfolio page"""
        self.render_header("Portfolio", [
            {"label": "Import", "key": "port_import"},
            {"label": "Add Holding", "key": "port_add"}
        ])
        
        # Portfolio summary metrics
        total_value = sum(holding['qty'] * holding['current_price'] for holding in st.session_state.user_portfolio.values())
        total_cost = sum(holding['qty'] * holding['avg_cost'] for holding in st.session_state.user_portfolio.values())
        unrealized_pnl = total_value - total_cost
        cash = 12500
        
        metrics = [
            {"label": "Total Value", "value": f"${total_value:,.0f}"},
            {"label": "Unrealized P/L", "value": f"+${unrealized_pnl:,.0f}"},
            {"label": "Cash", "value": f"${cash:,.0f}"},
            {"label": "Risk Score", "value": "Moderate"}
        ]
        
        self.render_metric_cards(metrics)
        
        # Holdings table
        st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
        st.markdown('<h2>Holdings</h2>', unsafe_allow_html=True)
        
        # Create holdings DataFrame
        holdings_data = []
        for ticker, data in st.session_state.user_portfolio.items():
            market_value = data['qty'] * data['current_price']
            pnl = market_value - (data['qty'] * data['avg_cost'])
            pnl_pct = (pnl / (data['qty'] * data['avg_cost'])) * 100
            
            holdings_data.append({
                'Asset': ticker,
                'Qty': data['qty'],
                'Avg Cost': f"${data['avg_cost']:.2f}",
                'Market Value': f"${market_value:,.0f}",
                'P/L': f"${pnl:,.0f}",
                'P/L %': f"{pnl_pct:+.1f}%"
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Export")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_alerts(self):
        """Render alerts page"""
        self.render_header("Alerts Center", [
            {"label": "Filters", "key": "alerts_filters"},
            {"label": "New Rule", "key": "alerts_new"}
        ])
        
        # Alert summary
        metrics = [
            {"label": "Open Alerts", "value": "18"},
            {"label": "Acknowledged", "value": "12"},
            {"label": "Resolved", "value": "34"},
            {"label": "Avg Time to Close", "value": "1.8d"}
        ]
        
        self.render_metric_cards(metrics)
        
        # Recent alerts
        st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
        st.markdown('<h2>Recent Alerts</h2>', unsafe_allow_html=True)
        
        for alert in st.session_state.alerts:
            status_class = f"status-{alert['status'].lower()}"
            severity_color = {
                'High': '#ef4444',
                'Medium': '#f59e0b', 
                'Critical': '#ef4444'
            }.get(alert['severity'], 'var(--text-secondary)')
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="flex: 1;">
                    <div style="color: var(--text-primary); font-weight: 500;">{alert['alert']}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem;">{alert['project']} • {alert['triggered']}</div>
                </div>
                <div style="display: flex; gap: 0.75rem; align-items: center;">
                    <span style="color: {severity_color}; font-size: 0.75rem;">{alert['severity']}</span>
                    <span class="status-badge {status_class}">{alert['status']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_stock_analysis(self):
        """Render stock analysis page"""
        self.render_header("Stock Analysis", [
            {"label": "Watchlist", "key": "analysis_watchlist"},
            {"label": "Add to Portfolio", "key": "analysis_add"}
        ])
        
        # Stock input
        ticker_input = st.text_input("Search ticker or company...", placeholder="e.g. AAPL, Apple Inc.")
        
        if ticker_input:
            try:
                # Get stock data
                stock = yf.Ticker(ticker_input.upper())
                data = stock.history(period="1mo")
                info = stock.info
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = ((current_price - prev_close) / prev_close) * 100
                    
                    # Stock header
                    st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
                    st.markdown(f'<h2>{ticker_input.upper()} ${current_price:.2f} NASDAQ</h2>', unsafe_allow_html=True)
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#14b8a6', width=2)
                    ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9',
                        height=400,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(gridcolor='#1e293b'),
                        yaxis=dict(gridcolor='#1e293b')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time range buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.button("1D")
                    with col2:
                        st.button("1W")
                    with col3:
                        st.button("1M")
                    with col4:
                        st.button("1Y")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Fundamentals
                    if info:
                        st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
                        st.markdown('<h2>Fundamentals</h2>', unsafe_allow_html=True)
                        
                        fund_data = {
                            'P/E': info.get('trailingPE', 'N/A'),
                            'EPS': info.get('trailingEps', 'N/A'),
                            'Market Cap': info.get('marketCap', 'N/A'),
                            'Revenue': info.get('totalRevenue', 'N/A')
                        }
                        
                        cols = st.columns(len(fund_data))
                        for col, (key, value) in zip(cols, fund_data.items()):
                            with col:
                                if isinstance(value, (int, float)) and value > 1000000:
                                    if value > 1000000000:
                                        value_str = f"${value/1000000000:.1f}B"
                                    else:
                                        value_str = f"${value/1000000:.1f}M"
                                elif isinstance(value, (int, float)):
                                    value_str = f"{value:.2f}"
                                else:
                                    value_str = str(value)
                                
                                st.metric(key, value_str)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error fetching data for {ticker_input}: {str(e)}")
    
    def render_market_scanner(self):
        """Render market scanner page"""
        self.render_header("Market Scanner", [
            {"label": "Save Scan", "key": "scanner_save"},
            {"label": "Run", "key": "scanner_run"}
        ])
        
        col_filters, col_results = st.columns([1, 2])
        
        with col_filters:
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Filters</h2>', unsafe_allow_html=True)
            
            # Filter controls
            exchange = st.selectbox("Exchange", ["All", "NYSE", "NASDAQ"])
            sector = st.selectbox("Sector", ["Any", "Technology", "Healthcare", "Finance"])
            market_cap = st.selectbox("Market Cap", ["> $10B", "$1B - $10B", "< $1B"])
            
            st.markdown("**Technical**")
            rsi_filter = st.checkbox("RSI < 30")
            price_filter = st.checkbox("Price vs 200MA: Above")
            macd_filter = st.checkbox("MACD: Bullish")
            
            st.markdown("**Fundamentals**")
            pe_filter = st.checkbox("P/E: 5 - 40")
            revenue_filter = st.checkbox("Revenue Growth: > 5%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("Clear")
            with col2:
                if st.button("Apply Filters"):
                    st.success("Scanning 5,000+ stocks...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_results:
            st.markdown('<div class="investtrack-card">', unsafe_allow_html=True)
            st.markdown('<h2>Results</h2>', unsafe_allow_html=True)
            
            # Sample scanner results
            scanner_results = [
                {"Ticker": "AAPL", "Company": "Apple Inc.", "Price": "$198.42", "% Chg": "+1.2%", "RSI": 62, "Trend": "↗"},
                {"Ticker": "MSFT", "Company": "Microsoft", "Price": "$421.10", "% Chg": "+0.8%", "RSI": 58, "Trend": "↗"},
                {"Ticker": "NVDA", "Company": "NVIDIA", "Price": "$890.20", "% Chg": "+2.1%", "RSI": 65, "Trend": "↗"}
            ]
            
            results_df = pd.DataFrame(scanner_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("Export CSV")
            with col2:
                st.button("Add to Watchlist")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        # Apply Grok's theme CSS
        st.markdown(self.get_grok_theme_css(), unsafe_allow_html=True)
        
        # Render sidebar
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.current_page == "Dashboard":
            self.render_dashboard()
        elif st.session_state.current_page == "Portfolio":
            self.render_portfolio()
        elif st.session_state.current_page == "Alerts":
            self.render_alerts()
        elif st.session_state.current_page == "Stock Analysis":
            self.render_stock_analysis()
        elif st.session_state.current_page == "Market Scanner":
            self.render_market_scanner()
        elif st.session_state.current_page == "Opportunities":
            st.markdown("## 💡 Investment Opportunities")
            st.info("Investment Opportunities page - AI-powered opportunity discovery coming soon!")
        elif st.session_state.current_page == "Projects":
            st.markdown("## 📁 Capital Projects")
            st.info("Capital Projects page - Project management and ROI tracking coming soon!")
        elif st.session_state.current_page == "Settings":
            st.markdown("## ⚙️ Settings")
            st.info("Settings page - Profile and preferences management coming soon!")

if __name__ == "__main__":
    app = InvestTrackPro()
    app.run()
