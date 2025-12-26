"""
FinAI Pro - Enterprise Financial Intelligence Platform

A comprehensive AI-powered financial analysis platform integrating
stock prediction, sentiment analysis, portfolio optimization, and market analysis.
Built with industry-standard practices and modern commercial aesthetics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
import warnings
import traceback

warnings.filterwarnings('ignore')

# Import custom modules - Use Real Data Fetcher (only real market data)
try:
    from data_fetchers.real_data_fetcher import RealDataFetcher
    USE_REAL_FETCHER = True
except ImportError:
    try:
        from data_fetchers.smart_data_fetcher import SmartDataFetcher
        USE_REAL_FETCHER = False
        USE_SMART_FETCHER = True
    except ImportError:
        try:
            from data_fetchers.multi_source_fetcher import MultiSourceDataFetcher
            USE_REAL_FETCHER = False
            USE_SMART_FETCHER = False
            USE_MULTI_SOURCE = True
        except ImportError:
            from data_fetchers.data_fetcher import StockDataFetcher
            USE_REAL_FETCHER = False
            USE_SMART_FETCHER = False
            USE_MULTI_SOURCE = False

from models.stock_predictor import StockPredictor, EnsembleStockPredictor
from models.robust_stock_predictor import RobustStockPredictor
from core.sentiment_analyzer import SentimentAnalyzer
from core.portfolio_optimizer import PortfolioOptimizer
from core.utils import format_currency, format_percentage

# Cross-Sectional Alpha Engine (NEW - Industry-grade module)
from alpha import CrossSectionalAlphaEngine, CrossSectionalEvaluator
from risk import RiskAdjustmentLayer
from portfolio import PortfolioAllocator

# ============================================================================
# BRAND DESIGN SYSTEM & PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinAI Pro | Enterprise Financial Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "FinAI Pro - Enterprise Financial Intelligence Platform"
    }
)

# Force sidebar to be always visible
st.markdown("""
<style>
    /* Force sidebar to always be visible and expanded */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        display: block !important;
        visibility: visible !important;
    }
    
    /* Ensure sidebar content is visible */
    [data-testid="stSidebar"] > div {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Make sidebar toggle button more visible */
    [data-testid="collapsedControl"] {
        background-color: #0A2463 !important;
        color: white !important;
        z-index: 999 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ENTERPRISE-GRADE CSS DESIGN SYSTEM
# ============================================================================

st.markdown("""
<style>
    /* ===== BRAND COLOR PALETTE ===== */
    :root {
        --primary: #0A2463;
        --primary-dark: #051A3A;
        --primary-light: #1E3A8A;
        --accent: #00D4FF;
        --accent-dark: #0099CC;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --neutral: #6B7280;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F9FAFB;
        --bg-tertiary: #F3F4F6;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --border: #E5E7EB;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%);
    }
    
    /* ===== BRAND HEADER ===== */
    .brand-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0;
        box-shadow: var(--shadow-lg);
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .brand-tagline {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ===== CARDS & CONTAINERS ===== */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md);
        margin-bottom: 2rem;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    h2 {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }

    h3 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* ===== BUTTONS ===== */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }

    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ===== INPUTS ===== */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid var(--border);
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(10, 36, 99, 0.1);
    }

    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid var(--border);
    }

    /* ===== CHARTS ===== */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }

    /* ===== SECTION DIVIDERS ===== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }

    /* ===== BADGES ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
    }

    .badge-info {
        background: rgba(0, 212, 255, 0.1);
        color: var(--accent-dark);
    }

    /* ===== FOOTER ===== */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border);
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .brand-logo {
            font-size: 1.5rem;
        }
        
        h1 {
            font-size: 2rem;
        }
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'data_fetcher' not in st.session_state:
    if USE_REAL_FETCHER:
        print("\n" + "="*60)
        print("Initializing Real Data Fetcher (Real Market Data Only)...")
        print("="*60)
        st.session_state.data_fetcher = RealDataFetcher()
        print("="*60 + "\n")
    elif USE_SMART_FETCHER:
        print("\n" + "="*60)
        print("Initializing Smart Data Fetcher...")
        print("="*60)
        st.session_state.data_fetcher = SmartDataFetcher(use_demo_on_error=False)
        print("="*60 + "\n")
    elif USE_MULTI_SOURCE:
        st.session_state.data_fetcher = MultiSourceDataFetcher(
            preferred_source="yfinance",
            use_demo_on_error=False
        )
    else:
        st.session_state.data_fetcher = StockDataFetcher(use_demo_on_error=False)

if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()
if 'portfolio_optimizer' not in st.session_state:
    st.session_state.portfolio_optimizer = PortfolioOptimizer()

# ============================================================================
# BRAND HEADER
# ============================================================================

st.markdown("""
<div class="brand-header">
    <div style="max-width: 1400px; margin: 0 auto; padding: 0 2rem;">
        <div class="brand-logo">
            <span>FinAI Pro</span>
        </div>
        <div class="brand-tagline">
            Enterprise Financial Intelligence Platform | Powered by AI & Advanced Analytics
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

st.sidebar.markdown("""
<div style="padding: 1rem 0; border-bottom: 2px solid #E5E7EB; margin-bottom: 1.5rem;">
    <h2 style="margin: 0; color: #0A2463; font-size: 1.25rem;">Navigation</h2>
</div>
""", unsafe_allow_html=True)

# Get current page index
page_options = ["Dashboard", "LSTM Price Forecast", "Gradient Boosting Direction", "Sentiment Analysis", "Portfolio Optimization", "Market Analysis", "Cross-Sectional Alpha Engine"]
current_page = st.session_state.get('page', 'Dashboard')
try:
    default_index = page_options.index(current_page) if current_page in page_options else 0
except ValueError:
    default_index = 0

page = st.sidebar.radio(
    "Select Module",
    page_options,
    index=default_index,
    label_visibility="collapsed"
)

# Update session state when radio changes
if page != st.session_state.page:
    st.session_state.page = page

# Data source info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background: #F9FAFB; border-radius: 8px; border-left: 4px solid #0A2463;">
    <h4 style="margin: 0 0 0.5rem 0; color: #0A2463; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">Data Source</h4>
    <p style="margin: 0; color: #6B7280; font-size: 0.875rem; font-weight: 600;">Alpha Vantage API</p>
    <p style="margin: 0.25rem 0 0 0; color: #9CA3AF; font-size: 0.75rem;">Real-time market data</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONTENT
# ============================================================================

# Use session state page if set, otherwise use radio selection
current_page = st.session_state.get('page', page)
if current_page != page:
    page = current_page

# Dashboard / Home page
if page == "Dashboard":
    st.markdown("""
    <div class="content-card">
        <h1 style="color: #0A2463; margin-bottom: 1rem;">Welcome to FinAI Pro</h1>
        <p style="font-size: 1.125rem; color: #6B7280; line-height: 1.75; margin-bottom: 2rem;">
            Your comprehensive AI-powered financial intelligence platform. Leverage advanced machine learning, 
            natural language processing, and quantitative analysis to make informed investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards - Clickable navigation
    col1, col2 = st.columns(2)
    
    with col1:
        # LSTM Price Forecast Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">LSTM Price Forecast</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                Advanced LSTM deep learning models predict stock price trends with high accuracy. 
                Analyze historical patterns and forecast future movements.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>Deep learning-based predictions</li>
                <li>Multi-timeframe analysis</li>
                <li>Accuracy metrics & confidence intervals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open LSTM Forecast", key="nav_stock_prediction", use_container_width=True, type="primary"):
            st.session_state.page = "LSTM Price Forecast"
            page = "LSTM Price Forecast"
            st.rerun()

        # Portfolio Optimization Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">Portfolio Optimization</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                Modern Portfolio Theory (MPT) implementation for optimal asset allocation. 
                Maximize returns while minimizing risk.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>Risk-return optimization</li>
                <li>Efficient frontier analysis</li>
                <li>Sharpe ratio maximization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Portfolio Optimization", key="nav_portfolio", use_container_width=True, type="primary"):
            st.session_state.page = "Portfolio Optimization"
            page = "Portfolio Optimization"
            st.rerun()

    with col2:
        # Sentiment Analysis Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">Sentiment Analysis</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                Real-time financial news analysis using NLP techniques. 
                Understand market sentiment and its potential impact.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>VADER & TextBlob sentiment scoring</li>
                <li>Multi-source news aggregation</li>
                <li>Market impact assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Sentiment Analysis", key="nav_sentiment", use_container_width=True, type="primary"):
            st.session_state.page = "Sentiment Analysis"
            page = "Sentiment Analysis"
            st.rerun()

        # Market Analysis Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">Market Analysis</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                Comprehensive technical analysis with professional-grade indicators. 
                Interactive charts and real-time data visualization.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>RSI, MACD, Bollinger Bands</li>
                <li>Moving averages & volume analysis</li>
                <li>Interactive candlestick charts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Market Analysis", key="nav_market", use_container_width=True, type="primary"):
            st.session_state.page = "Market Analysis"
            page = "Market Analysis"
            st.rerun()
    
    # Add Gradient Boosting Direction card in a new row
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        # Gradient Boosting Direction Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">Gradient Boosting Direction</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                XGBoost/LightGBM-based direction classifier optimized for small-sample data (~100 days). 
                Conservative approach with strong regularization to prevent overfitting.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>Classification-based (up/down direction)</li>
                <li>Walk-forward validation</li>
                <li>Interpretable features & metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open GBM Direction", key="nav_robust_prediction", use_container_width=True, type="primary"):
            st.session_state.page = "Gradient Boosting Direction"
            page = "Gradient Boosting Direction"
            st.rerun()
    
    with col4:
        # Cross-Sectional Alpha Engine Card - Clickable
        st.markdown("""
        <div class="content-card" style="cursor: pointer; transition: all 0.3s; min-height: 200px;">
            <h2 style="margin: 0 0 1rem 0; color: #0A2463; font-size: 1.5rem; font-weight: 700;">Cross-Sectional Alpha Engine</h2>
            <p style="color: #6B7280; line-height: 1.75; margin-bottom: 1rem;">
                Industry-grade relative value ranking + risk allocation pipeline. 
                ML models as noisy alpha signal generators, not predictors.
            </p>
            <ul style="color: #6B7280; line-height: 2;">
                <li>Cross-sectional alpha signals</li>
                <li>Risk adjustment layer</li>
                <li>Portfolio allocation weights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Alpha Engine", key="nav_cross_sectional", use_container_width=True, type="primary"):
            st.session_state.page = "Cross-Sectional Alpha Engine"
            page = "Cross-Sectional Alpha Engine"
            st.rerun()

    # Technology stack
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #0A2463; margin-bottom: 1rem;">Technology Stack</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="padding: 1rem; background: #F9FAFB; border-radius: 8px;">
                <strong style="color: #0A2463;">Frontend</strong>
                <p style="margin: 0.5rem 0 0 0; color: #6B7280;">Streamlit</p>
            </div>
            <div style="padding: 1rem; background: #F9FAFB; border-radius: 8px;">
                <strong style="color: #0A2463;">Machine Learning</strong>
                <p style="margin: 0.5rem 0 0 0; color: #6B7280;">PyTorch, scikit-learn</p>
            </div>
            <div style="padding: 1rem; background: #F9FAFB; border-radius: 8px;">
                <strong style="color: #0A2463;">NLP</strong>
                <p style="margin: 0.5rem 0 0 0; color: #6B7280;">VADER, TextBlob, NLTK</p>
            </div>
            <div style="padding: 1rem; background: #F9FAFB; border-radius: 8px;">
                <strong style="color: #0A2463;">Visualization</strong>
                <p style="margin: 0.5rem 0 0 0; color: #6B7280;">Plotly</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# LSTM Price Forecast page
elif page == "LSTM Price Forecast":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">LSTM Price Forecast</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">AI-powered forecasting using advanced LSTM neural networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT", key="pred_symbol")
        with col2:
            period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=1, key="pred_period", help="Longer periods provide more training data for better predictions")
        with col3:
            predict_days = st.slider("Prediction Days", 7, 60, 30, key="pred_days")
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_ensemble = st.checkbox("Use Ensemble Model", value=False, help="Train multiple models and combine predictions for better accuracy")
            if use_ensemble:
                n_models = st.slider("Number of Models", 3, 7, 3, help="More models = better accuracy but slower training")
            else:
                n_models = 1
        with col2:
            show_feature_importance = st.checkbox("Analyze Feature Importance", value=False, help="Show which features are most important for predictions")
            run_cross_validation = st.checkbox("Run Cross-Validation", value=False, help="Perform walk-forward time series cross-validation")
        
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=False, help="Automatically find best hyperparameters (takes longer)")
        if optimize_hyperparams:
            n_trials = st.slider("Optimization Trials", 10, 50, 20, help="More trials = better results but slower")
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Fetching data and training model..."):
            try:
                data = st.session_state.data_fetcher.get_stock_data(symbol, period=period)
                
                if data.empty:
                    st.error(f"Unable to fetch data for {symbol}. Please check the symbol.")
                else:
                    info = st.session_state.data_fetcher.get_stock_info(symbol)
                    
                    # Key metrics in styled cards
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Market Overview</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("52W High", f"${info.get('52w_high', 0):.2f}")
                    with col3:
                        st.metric("52W Low", f"${info.get('52w_low', 0):.2f}")
                    with col4:
                        if len(data) >= 30:
                            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
                            st.metric("30D Change", f"{change:.2f}%")
                        else:
                            st.metric("30D Change", "N/A")
                    
                    # Hyperparameter optimization
                    best_params = None
                    if optimize_hyperparams:
                        with st.spinner(f"Optimizing hyperparameters ({n_trials} trials)..."):
                            try:
                                opt_result = StockPredictor.optimize_hyperparameters(
                                    data, target_col='Close', n_trials=n_trials
                                )
                                best_params = opt_result['best_params']
                                st.success(f"**Best RMSE**: ${opt_result['best_score']:.2f}")
                                st.json(best_params)
                                
                                # Update predictor with best parameters
                                st.session_state.predictor = StockPredictor(
                                    sequence_length=best_params['sequence_length'],
                                    hidden_size=best_params['hidden_size'],
                                    num_layers=best_params['num_layers']
                                )
                            except Exception as e:
                                st.warning(f"Hyperparameter optimization failed: {e}. Using default parameters.")
                    
                    # Auto-adjust sequence length if data is limited
                    min_required = st.session_state.predictor.sequence_length + 50
                    original_seq_len = st.session_state.predictor.sequence_length
                    if len(data) < min_required:
                        # Reduce sequence length to fit available data
                        new_seq_len = max(30, len(data) - 50)  # Keep at least 30, but leave 50 for training
                        if new_seq_len < original_seq_len:
                            st.session_state.predictor.sequence_length = new_seq_len
                            st.info(
                                f"**Auto-adjustment**: Reduced sequence length from {original_seq_len} to {new_seq_len} "
                                f"to fit available data ({len(data)} points). For better results, use a longer period (2y or 5y)."
                            )
                    
                    # Choose between single model and ensemble
                    if use_ensemble:
                        # Use ensemble model
                        ensemble_predictor = EnsembleStockPredictor(
                            n_models=n_models,
                            sequence_length=st.session_state.predictor.sequence_length,
                            hidden_size=st.session_state.predictor.hidden_size,
                            num_layers=st.session_state.predictor.num_layers
                        )
                        
                        # Train ensemble
                        with st.expander("Ensemble Model Training Progress", expanded=False):
                            try:
                                all_losses = ensemble_predictor.train(data, epochs=150)
                                # Show average loss across models
                                avg_losses = np.mean(all_losses, axis=0)
                                st.line_chart(avg_losses)
                                st.info(f"Trained {n_models} models for ensemble prediction")
                            except ValueError as ve:
                                if "Insufficient data" in str(ve):
                                    st.error(
                                        f"**Insufficient Data**: {str(ve)}\n\n"
                                        f"**Solution**: Please select a longer data period (2y or 5y) "
                                        f"or try a different stock symbol."
                                    )
                                    st.stop()
                                else:
                                    raise
                        
                        # Evaluate ensemble
                        metrics = ensemble_predictor.evaluate(data)
                        predictor_for_prediction = ensemble_predictor
                    else:
                        # Use single model
                        with st.expander("Model Training Progress", expanded=False):
                            try:
                                train_losses = st.session_state.predictor.train(data, epochs=150)
                                st.line_chart(train_losses)
                            except ValueError as ve:
                                if "Insufficient data" in str(ve):
                                    st.error(
                                        f"**Insufficient Data**: {str(ve)}\n\n"
                                        f"**Solution**: Please select a longer data period (2y or 5y) "
                                        f"or try a different stock symbol."
                                    )
                                    st.stop()
                                else:
                                    raise
                    
                    # Evaluate model
                        metrics = st.session_state.predictor.evaluate(data)
                        predictor_for_prediction = st.session_state.predictor
                    
                    # Cross-validation
                    if run_cross_validation:
                        with st.spinner("Running walk-forward cross-validation..."):
                            try:
                                cv_metrics = st.session_state.predictor.walk_forward_validation(data, n_splits=5)
                                st.success(f"**Cross-Validation Results** ({cv_metrics['n_splits']} folds):")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("CV RMSE", f"${cv_metrics['RMSE']:.2f}")
                                with col2:
                                    st.metric("CV MAPE", f"{cv_metrics['MAPE']:.2f}%")
                                with col3:
                                    st.metric("CV Direction Accuracy", f"{cv_metrics['Direction_Accuracy']:.2f}%")
                            except Exception as e:
                                st.warning(f"Cross-validation failed: {e}")
                    
                    # Feature importance analysis
                    if show_feature_importance:
                        with st.spinner("Analyzing feature importance..."):
                            try:
                                feature_importance = st.session_state.predictor.analyze_feature_importance(data)
                                
                                st.markdown("""
                                <div style="margin: 2rem 0;">
                                    <h3 style="color: #0A2463; margin-bottom: 1rem;">Feature Importance</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Sort by importance
                                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                                
                                # Create bar chart
                                feature_names = [f[0] for f in sorted_features]
                                importance_values = [f[1] for f in sorted_features]
                                
                                fig_importance = px.bar(
                                    x=importance_values,
                                    y=feature_names,
                                    orientation='h',
                                    labels={'x': 'Importance Score', 'y': 'Feature'},
                                    title='Feature Importance Analysis',
                                    color=importance_values,
                                    color_continuous_scale='Blues'
                                )
                                fig_importance.update_layout(height=400)
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Show table
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importance_values
                                })
                                st.dataframe(importance_df, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Feature importance analysis failed: {e}")
                    
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Model Performance</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
                    with col2:
                        st.metric("MAE", f"${metrics['MAE']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    with col4:
                        accuracy = max(0, 100 - metrics['MAPE'])
                        st.metric("Price Accuracy", f"{accuracy:.2f}%")
                    with col5:
                        st.metric("Direction Accuracy", f"{metrics.get('Direction_Accuracy', 0):.2f}%")
                    
                    # Predict future prices
                    predictions = predictor_for_prediction.predict(data, days=predict_days)
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=predict_days, freq='D')
                    
                    # Professional chart styling
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index[-60:],
                        y=data['Close'].iloc[-60:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#0A2463', width=3),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Prediction data
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines',
                        name='Predicted Price',
                        line=dict(color='#00D4FF', width=3, dash='dash'),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Current price point
                    fig.add_trace(go.Scatter(
                        x=[last_date],
                        y=[data['Close'].iloc[-1]],
                        mode='markers',
                        name='Current Price',
                        marker=dict(size=12, color='#10B981', symbol='circle'),
                        hovertemplate='<b>Current</b><br>Price: $%{y:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=dict(
                            text=f"{symbol} Stock Price Prediction",
                            font=dict(size=24, color='#0A2463', family='Arial, sans-serif')
                        ),
                        xaxis=dict(
                            title=dict(text="Date", font=dict(size=14, color='#6B7280')),
                            tickfont=dict(size=12, color='#6B7280'),
                            gridcolor='#E5E7EB',
                            showgrid=True
                        ),
                        yaxis=dict(
                            title=dict(text="Price ($)", font=dict(size=14, color='#6B7280')),
                            tickfont=dict(size=12, color='#6B7280'),
                            gridcolor='#E5E7EB',
                            showgrid=True
                        ),
                        hovermode='x unified',
                        height=600,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=12, color='#6B7280')
                        ),
                        margin=dict(t=80, b=60, l=60, r=60)
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Prediction statistics
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Prediction Statistics</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted High", f"${np.max(predictions):.2f}")
                    with col2:
                        st.metric("Predicted Low", f"${np.min(predictions):.2f}")
                    with col3:
                        predicted_change = ((predictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
                        st.metric("Predicted Change", f"{predicted_change:.2f}%")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    st.error("**Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("**Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Financial News Sentiment Analysis</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">Real-time news analysis using advanced NLP techniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT", key="sent_symbol")
    with col2:
        max_articles = st.slider("Number of Articles", 5, 20, 10, key="sent_articles")
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        with st.spinner("Fetching and analyzing news..."):
            try:
                result = st.session_state.sentiment_analyzer.analyze_stock_news(symbol, max_articles)
                
                if not result['articles']:
                    st.warning("No related news found")
                else:
                    summary = result['summary']
                    
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Overall Sentiment Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", summary['total_articles'])
                    with col2:
                        st.metric("Positive", summary['positive'], delta=f"{summary['positive']/summary['total_articles']*100:.1f}%")
                    with col3:
                        st.metric("Neutral", summary['neutral'], delta=f"{summary['neutral']/summary['total_articles']*100:.1f}%")
                    with col4:
                        st.metric("Negative", summary['negative'], delta=f"{summary['negative']/summary['total_articles']*100:.1f}%")
                    
                    st.metric("Average Sentiment Score", f"{summary['avg_sentiment']:.3f}", 
                             delta=summary['overall_sentiment'])
                    
                    impact = st.session_state.sentiment_analyzer.get_sentiment_impact(summary['avg_sentiment'])
                    st.info(f"💡 **Market Impact Assessment**: {impact}")
                    
                    # Sentiment distribution chart
                    sentiment_counts = {
                        'Positive': summary['positive'],
                        'Neutral': summary['neutral'],
                        'Negative': summary['negative']
                    }
                    fig = px.pie(
                        values=list(sentiment_counts.values()),
                        names=list(sentiment_counts.keys()),
                        title="Sentiment Distribution",
                        color_discrete_map={'Positive': '#10B981', 'Neutral': '#6B7280', 'Negative': '#EF4444'}
                    )
                    fig.update_layout(
                        title_font=dict(size=20, color='#0A2463'),
                        font=dict(size=12, color='#6B7280'),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Detailed article list
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Detailed Analysis Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    articles_df = pd.DataFrame(result['articles'])
                    articles_df = articles_df[['title', 'publisher', 'sentiment', 'combined_score', 'link']]
                    articles_df.columns = ['Title', 'Publisher', 'Sentiment', 'Score', 'Link']
                    articles_df = articles_df.sort_values('Score', ascending=False)
                    
                    for idx, row in articles_df.iterrows():
                        sentiment_color = {'Positive': '#10B981', 'Neutral': '#6B7280', 'Negative': '#EF4444'}.get(row['Sentiment'], '#6B7280')
                        with st.expander(f"{row['Title']} - **{row['Sentiment']}** ({row['Score']:.3f})"):
                            st.write(f"**Publisher**: {row['Publisher']}")
                            st.write(f"**Sentiment Score**: {row['Score']:.3f}")
                            st.write(f"**Link**: [{row['Link']}]({row['Link']})")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    st.error("**Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("**Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Portfolio Optimization page
elif page == "Portfolio Optimization":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Portfolio Optimization</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">Modern Portfolio Theory implementation for optimal asset allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Enter Stock Symbols (comma-separated)")
    symbols_input = st.text_input(
        "Stock Symbols", 
        value="AAPL,MSFT,GOOGL,AMZN,TSLA",
        placeholder="e.g., AAPL,MSFT,GOOGL",
        key="port_symbols"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1, key="port_period")
    with col2:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["max_sharpe", "min_volatility", "max_return"],
            format_func=lambda x: {
                "max_sharpe": "Maximize Sharpe Ratio",
                "min_volatility": "Minimize Volatility",
                "max_return": "Maximize Return"
            }[x],
            key="port_method"
        )
    
    if st.button("⚡ Optimize Portfolio", type="primary", use_container_width=True):
        with st.spinner("Analyzing and optimizing portfolio..."):
            try:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
                stocks_data = st.session_state.data_fetcher.get_multiple_stocks(symbols, period)
                
                if not stocks_data:
                    st.error("Unable to fetch stock data")
                else:
                    prices_df = pd.DataFrame({symbol: data['Close'] for symbol, data in stocks_data.items()})
                    returns_df = st.session_state.portfolio_optimizer.calculate_returns(prices_df)
                    result = st.session_state.portfolio_optimizer.optimize_portfolio(
                        returns_df, 
                        method=optimization_method
                    )
                    
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Optimization Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Annual Return", f"{result['expected_return']*100:.2f}%")
                    with col2:
                        st.metric("Annual Volatility", f"{result['volatility']*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                    
                    # Weight allocation
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Optimal Weight Allocation</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    weights_df = pd.DataFrame(
                        list(result['weights'].items()),
                        columns=['Stock', 'Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'] * 100
                    weights_df = weights_df.sort_values('Weight', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(weights_df.style.format({'Weight': '{:.2f}%'}))
                    
                    with col2:
                        fig = px.pie(
                            weights_df,
                            values='Weight',
                            names='Stock',
                            title="Portfolio Weight Allocation"
                        )
                        fig.update_layout(
                            title_font=dict(size=20, color='#0A2463'),
                            font=dict(size=12, color='#6B7280'),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    # Efficient frontier
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Efficient Frontier Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    frontier = st.session_state.portfolio_optimizer.efficient_frontier(returns_df, num_portfolios=100)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=frontier['volatility'] * 100,
                        y=frontier['return'] * 100,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=frontier['sharpe_ratio'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Portfolios'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[result['volatility'] * 100],
                        y=[result['expected_return'] * 100],
                        mode='markers',
                        marker=dict(size=15, color='#EF4444', symbol='star'),
                        name='Optimal Portfolio'
                    ))
                    
                    fig.update_layout(
                        title=dict(text="Efficient Frontier", font=dict(size=20, color='#0A2463')),
                        xaxis=dict(title=dict(text="Volatility (%)", font=dict(size=14, color='#6B7280'))),
                        yaxis=dict(title=dict(text="Expected Return (%)", font=dict(size=14, color='#6B7280'))),
                        height=600,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12, color='#6B7280')
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Compare different methods
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Method Comparison</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    comparison = st.session_state.portfolio_optimizer.compare_portfolios(
                        returns_df,
                        methods=['max_sharpe', 'min_volatility', 'equal_weight']
                    )
                    
                    comparison_df = pd.DataFrame({
                        'Method': ['Maximize Sharpe Ratio', 'Minimize Volatility', 'Equal Weight'],
                        'Return (%)': [c['expected_return']*100 for c in comparison.values()],
                        'Volatility (%)': [c['volatility']*100 for c in comparison.values()],
                        'Sharpe Ratio': [c['sharpe_ratio'] for c in comparison.values()]
                    })
                    
                    st.dataframe(comparison_df.style.format({
                        'Return (%)': '{:.2f}%',
                        'Volatility (%)': '{:.2f}%',
                        'Sharpe Ratio': '{:.3f}'
                    }))
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    st.error("**Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("**Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Market Analysis page
elif page == "Market Analysis":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Market Data Analysis</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">Comprehensive technical analysis with professional-grade indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT", key="market_symbol")
    with col2:
        period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"], index=2, key="market_period")
    
    if st.button("Load Market Data", type="primary", use_container_width=True):
        with st.spinner("Fetching market data..."):
            try:
                data = st.session_state.data_fetcher.get_stock_data(symbol, period=period)
                
                if data.empty:
                    st.error(f"Unable to fetch data for {symbol}")
                else:
                    info = st.session_state.data_fetcher.get_stock_info(symbol)
                    st.markdown(f"""
                    <div style="margin: 2rem 0;">
                        <h2 style="color: #0A2463; margin-bottom: 0.5rem;">{info.get('name', symbol)} Market Analysis</h2>
                        <p style="color: #6B7280;">{info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Market Cap", format_currency(info.get('market_cap', 0)))
                    with col3:
                        pe_ratio = info.get('pe_ratio', 0)
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio > 0 else "N/A")
                    with col4:
                        div_yield = info.get('dividend_yield', 0)
                        st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield > 0 else "N/A")
                    with col5:
                        change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                        st.metric("Period Change", f"{change:.2f}%")
                    
                    # Price chart
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Price Trend</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price',
                        increasing_line_color='#10B981',
                        decreasing_line_color='#EF4444'
                    ))
                    
                    # Moving averages
                    if 'MA20' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MA20'],
                            mode='lines',
                            name='MA20',
                            line=dict(color='#F59E0B', width=2)
                        ))
                    if 'MA50' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MA50'],
                            mode='lines',
                            name='MA50',
                            line=dict(color='#0A2463', width=2)
                        ))
                    
                    fig.update_layout(
                        title=dict(text=f"{symbol} Price Chart", font=dict(size=20, color='#0A2463')),
                        xaxis=dict(title=dict(text="Date", font=dict(size=14, color='#6B7280'))),
                        yaxis=dict(title=dict(text="Price ($)", font=dict(size=14, color='#6B7280'))),
                        height=600,
                        xaxis_rangeslider_visible=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12, color='#6B7280')
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Technical indicators
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Technical Indicators</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'RSI' in data.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(
                                x=data.index,
                                y=data['RSI'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='#8B5CF6', width=2)
                            ))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold")
                            fig_rsi.update_layout(
                                title=dict(text="RSI Indicator", font=dict(size=16, color='#0A2463')),
                                height=350,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(size=12, color='#6B7280')
                            )
                            st.plotly_chart(fig_rsi, width='stretch')
                    
                    with col2:
                        if 'MACD' in data.columns:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(
                                x=data.index,
                                y=data['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='#0A2463', width=2)
                            ))
                            fig_macd.add_trace(go.Scatter(
                                x=data.index,
                                y=data['MACD_signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='#EF4444', width=2)
                            ))
                            fig_macd.add_trace(go.Bar(
                                x=data.index,
                                y=data['MACD_hist'],
                                name='Histogram',
                                marker_color='#6B7280'
                            ))
                            fig_macd.update_layout(
                                title=dict(text="MACD Indicator", font=dict(size=16, color='#0A2463')),
                                height=350,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(size=12, color='#6B7280')
                            )
                            st.plotly_chart(fig_macd, width='stretch')
                    
                    # Volume
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Volume Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='#00D4FF'
                    ))
                    if 'Volume_MA' in data.columns:
                        fig_vol.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Volume_MA'],
                            mode='lines',
                            name='Volume MA',
                            line=dict(color='#EF4444', width=2)
                        ))
                    fig_vol.update_layout(
                        title=dict(text="Volume", font=dict(size=16, color='#0A2463')),
                        height=350,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12, color='#6B7280')
                    )
                    st.plotly_chart(fig_vol, width='stretch')
                    
                    # Data table
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Recent Data</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if 'RSI' in data.columns:
                        display_cols.append('RSI')
                    if 'MACD' in data.columns:
                        display_cols.append('MACD')
                    
                    display_data = data[display_cols].tail(20)
                    format_dict = {
                        'Open': '${:.2f}',
                        'High': '${:.2f}',
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,.0f}',
                    }
                    if 'RSI' in display_data.columns:
                        format_dict['RSI'] = '{:.2f}'
                    if 'MACD' in display_data.columns:
                        format_dict['MACD'] = '{:.4f}'
                    
                    st.dataframe(display_data.style.format(format_dict))
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    st.error("**Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("**Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# Gradient Boosting Direction page (XGBoost/LightGBM for small-sample data)
elif page == "Gradient Boosting Direction":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Gradient Boosting Direction</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">XGBoost/LightGBM-based classifier optimized for small-sample data (~100 trading days)</p>
        <p style="color: #9CA3AF; font-size: 0.875rem; margin-top: 0.5rem;">
            <strong>Note:</strong> This is a methodology demonstration for small-sample financial time-series, 
            NOT a live trading system. Conservative approach with strong regularization to prevent overfitting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT", key="robust_symbol")
        with col2:
            period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=0, key="robust_period", 
                                 help="Longer periods provide more training data")
        with col3:
            model_type = st.selectbox("Model Type", ["xgboost", "lightgbm"], index=0, key="robust_model_type")
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            forward_days = st.slider("Forward Days", 3, 10, 5, help="Number of days ahead to predict")
            transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05, 
                                         help="Estimated transaction cost threshold")
        with col2:
            run_cross_validation = st.checkbox("Run Walk-Forward Validation", value=True, 
                                              help="Time-series aware cross-validation")
            if run_cross_validation:
                n_splits = st.slider("Validation Folds", 3, 10, 5)
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Fetching data and training model..."):
            try:
                data = st.session_state.data_fetcher.get_stock_data(symbol, period=period)
                
                if data.empty:
                    st.error(f"Unable to fetch data for {symbol}. Please check the symbol.")
                else:
                    info = st.session_state.data_fetcher.get_stock_info(symbol)
                    
                    # Key metrics
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Market Overview</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("52W High", f"${info.get('52w_high', 0):.2f}")
                    with col3:
                        st.metric("52W Low", f"${info.get('52w_low', 0):.2f}")
                    with col4:
                        if len(data) >= 30:
                            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
                            st.metric("30D Change", f"{change:.2f}%")
                        else:
                            st.metric("30D Change", "N/A")
                    
                    # Initialize predictor
                    try:
                        predictor = RobustStockPredictor(
                            model_type=model_type,
                            forward_days=forward_days,
                            transaction_cost=transaction_cost / 100.0
                        )
                    except ImportError as e:
                        st.error(f"**Import Error**: {str(e)}")
                        st.info("Please install the required package:\n"
                               f"- For XGBoost: `pip install xgboost`\n"
                               f"- For LightGBM: `pip install lightgbm`")
                        st.stop()
                    
                    # Train model
                    with st.expander("Model Training", expanded=False):
                        try:
                            results = predictor.train(data, validation_size=0.2)
                            
                            st.success("Model trained successfully!")
                            
                            # Display metrics
                            st.markdown("""
                            <div style="margin: 2rem 0;">
                                <h3 style="color: #0A2463; margin-bottom: 1rem;">Model Performance</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                            with col2:
                                st.metric("Precision", f"{results['precision']*100:.2f}%")
                            with col3:
                                st.metric("Recall", f"{results['recall']*100:.2f}%")
                            with col4:
                                st.metric("F1 Score", f"{results['f1_score']*100:.2f}%")
                            with col5:
                                st.metric("Baseline", f"{results['baseline_accuracy']*100:.2f}%")
                            
                            # Confusion Matrix
                            st.markdown("""
                            <div style="margin: 2rem 0;">
                                <h4 style="color: #0A2463; margin-bottom: 1rem;">Confusion Matrix</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            cm = results['confusion_matrix']
                            cm_df = pd.DataFrame(
                                cm,
                                index=['Actual Down', 'Actual Up'],
                                columns=['Predicted Down', 'Predicted Up']
                            )
                            st.dataframe(cm_df, use_container_width=True)
                            
                            # Feature Importance
                            if results.get('feature_importance'):
                                st.markdown("""
                                <div style="margin: 2rem 0;">
                                    <h4 style="color: #0A2463; margin-bottom: 1rem;">Feature Importance</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                importance_df = pd.DataFrame({
                                    'Feature': list(results['feature_importance'].keys()),
                                    'Importance': list(results['feature_importance'].values())
                                }).sort_values('Importance', ascending=False)
                                
                                st.dataframe(importance_df, use_container_width=True)
                                
                                # Feature importance chart
                                fig_importance = px.bar(
                                    importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
                                    title='Feature Importance Analysis',
                                    color='Importance',
                                    color_continuous_scale='Blues'
                                )
                                fig_importance.update_layout(height=400)
                                st.plotly_chart(fig_importance, use_container_width=True)
                            
                        except ValueError as ve:
                            st.error(f"**Insufficient Data**: {str(ve)}")
                            st.info("**Solution**: Please select a longer data period (2y or 5y) or try a different stock symbol.")
                            st.stop()
                        except Exception as e:
                            st.error(f"**Training Error**: {str(e)}")
                            with st.expander("Show detailed error"):
                                st.code(traceback.format_exc())
                            st.stop()
                    
                    # Walk-forward validation
                    if run_cross_validation:
                        with st.expander("Walk-Forward Cross-Validation", expanded=True):
                            try:
                                cv_results = predictor.walk_forward_validation(data, n_splits=n_splits)
                                
                                st.success(f"Cross-validation completed ({cv_results['n_splits']} folds)")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CV Accuracy", f"{cv_results['accuracy']*100:.2f}%", 
                                             delta=f"±{cv_results['std_accuracy']*100:.2f}%")
                                with col2:
                                    st.metric("CV Precision", f"{cv_results['precision']*100:.2f}%")
                                with col3:
                                    st.metric("CV Recall", f"{cv_results['recall']*100:.2f}%")
                                with col4:
                                    st.metric("CV F1 Score", f"{cv_results['f1_score']*100:.2f}%")
                                
                                # Confusion matrix for CV
                                cv_cm = confusion_matrix(cv_results['actual'], cv_results['predictions'])
                                cv_cm_df = pd.DataFrame(
                                    cv_cm,
                                    index=['Actual Down', 'Actual Up'],
                                    columns=['Predicted Down', 'Predicted Up']
                                )
                                st.markdown("**Cross-Validation Confusion Matrix:**")
                                st.dataframe(cv_cm_df, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Cross-validation failed: {e}")
                    
                    # Make predictions for recent data
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Recent Predictions</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        predictions = predictor.predict(data)
                        probabilities = predictor.predict_proba(data)
                        
                        # Get recent predictions
                        recent_data = data.tail(20)
                        recent_predictions = predictions[-20:]
                        recent_proba = probabilities[-20:]
                        
                        # Create prediction dataframe
                        pred_df = pd.DataFrame({
                            'Date': recent_data.index,
                            'Close Price': recent_data['Close'].values,
                            'Prediction': ['Up' if p == 1 else 'Down' for p in recent_predictions],
                            'Probability': recent_proba
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Buy/Sell Recommendation
                        latest_prediction = recent_predictions[-1]  # Latest prediction (1 = Up, 0 = Down)
                        latest_probability = recent_proba[-1]  # Latest probability of "Up"
                        latest_date = recent_data.index[-1]
                        latest_price = recent_data['Close'].iloc[-1]
                        
                        st.markdown("""
                        <div style="margin: 2rem 0;">
                            <h3 style="color: #0A2463; margin-bottom: 1rem;">Trading Recommendation</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Determine recommendation based on prediction and probability
                        if latest_prediction == 1 and latest_probability > 0.7:
                            st.success(f"""
                            **✅ STRONG BUY RECOMMENDATION**
                            
                            - **Date**: {latest_date.strftime('%Y-%m-%d')}
                            - **Current Price**: ${latest_price:.2f}
                            - **Prediction**: Up (Future {forward_days}-day return expected to exceed transaction cost)
                            - **Confidence**: {latest_probability*100:.1f}%
                            - **Recommendation**: Model predicts a strong upward movement. Consider buying with appropriate risk management.
                            """)
                        elif latest_prediction == 1 and latest_probability > 0.6:
                            st.info(f"""
                            **📈 MODERATE BUY RECOMMENDATION**
                            
                            - **Date**: {latest_date.strftime('%Y-%m-%d')}
                            - **Current Price**: ${latest_price:.2f}
                            - **Prediction**: Up (Future {forward_days}-day return expected to exceed transaction cost)
                            - **Confidence**: {latest_probability*100:.1f}%
                            - **Recommendation**: Model predicts upward movement, but confidence is moderate. Consider buying with caution and set stop-loss.
                            """)
                        elif latest_prediction == 1 and latest_probability <= 0.6:
                            st.warning(f"""
                            **⚠️ WEAK BUY / HOLD RECOMMENDATION**
                            
                            - **Date**: {latest_date.strftime('%Y-%m-%d')}
                            - **Current Price**: ${latest_price:.2f}
                            - **Prediction**: Up (Future {forward_days}-day return expected to exceed transaction cost)
                            - **Confidence**: {latest_probability*100:.1f}%
                            - **Recommendation**: Model predicts upward movement, but confidence is low. Consider holding existing positions or waiting for stronger signals.
                            """)
                        else:
                            st.error(f"""
                            **❌ SELL / AVOID BUYING RECOMMENDATION**
                            
                            - **Date**: {latest_date.strftime('%Y-%m-%d')}
                            - **Current Price**: ${latest_price:.2f}
                            - **Prediction**: Down (Future {forward_days}-day return not expected to exceed transaction cost)
                            - **Confidence**: {(1-latest_probability)*100:.1f}% (probability of downward movement)
                            - **Recommendation**: Model predicts downward movement or insufficient return. Avoid buying new positions. Consider selling if holding.
                            """)
                        
                        # Prediction chart
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            x=recent_data.index,
                            y=recent_data['Close'].values,
                            mode='lines+markers',
                            name='Close Price',
                            line=dict(color='#0A2463', width=2)
                        ))
                        
                        # Add prediction markers
                        up_dates = [recent_data.index[i] for i, p in enumerate(recent_predictions) if p == 1]
                        down_dates = [recent_data.index[i] for i, p in enumerate(recent_predictions) if p == 0]
                        up_prices = [recent_data['Close'].iloc[i] for i, p in enumerate(recent_predictions) if p == 1]
                        down_prices = [recent_data['Close'].iloc[i] for i, p in enumerate(recent_predictions) if p == 0]
                        
                        if up_dates:
                            fig_pred.add_trace(go.Scatter(
                                x=up_dates,
                                y=up_prices,
                                mode='markers',
                                name='Predicted Up',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        if down_dates:
                            fig_pred.add_trace(go.Scatter(
                                x=down_dates,
                                y=down_prices,
                                mode='markers',
                                name='Predicted Down',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                        
                        fig_pred.update_layout(
                            title=dict(text=f"{symbol} Price with Predictions", font=dict(size=20, color='#0A2463')),
                            xaxis=dict(title="Date"),
                            yaxis=dict(title="Price ($)"),
                            height=500,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        with st.expander("Show detailed error"):
                            st.code(traceback.format_exc())
                    
                    # Disclaimer
                    st.markdown("""
                    <div style="margin: 2rem 0; padding: 1rem; background: #FEF3C7; border-radius: 8px; border-left: 4px solid #F59E0B;">
                        <p style="margin: 0; color: #92400E; font-size: 0.875rem;">
                            <strong>Disclaimer:</strong> This is a methodology demonstration for small-sample financial time-series analysis. 
                            Results are NOT guaranteed to be profitable and should NOT be used for live trading without extensive validation. 
                            Past performance does not guarantee future results.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"**Error**: {str(e)}")
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

# Cross-Sectional Alpha Engine page (NEW - Industry-grade module)
elif page == "Cross-Sectional Alpha Engine":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Cross-Sectional Alpha Engine (Experimental)</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">Industry-grade relative value ranking + risk allocation pipeline</p>
        <p style="color: #9CA3AF; font-size: 0.875rem; margin-top: 0.5rem;">
            <strong>Research / Allocation Signal</strong> - Relative ranking for risk allocation, NOT price prediction.
            This module provides risk-aware signals for portfolio construction, not investment advice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for cross-sectional engine
    if 'cross_sectional_engine' not in st.session_state:
        st.session_state.cross_sectional_engine = None
    if 'risk_adjustment_layer' not in st.session_state:
        st.session_state.risk_adjustment_layer = RiskAdjustmentLayer()
    if 'portfolio_allocator' not in st.session_state:
        st.session_state.portfolio_allocator = PortfolioAllocator()
    if 'cross_sectional_evaluator' not in st.session_state:
        st.session_state.cross_sectional_evaluator = CrossSectionalEvaluator()
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            symbols_input = st.text_input(
                "Stock Symbols (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,V,JNJ",
                placeholder="e.g., AAPL,MSFT,GOOGL",
                key="cs_symbols"
            )
        with col2:
            period = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1, key="cs_period")
        
        col3, col4 = st.columns(2)
        with col3:
            model_type = st.selectbox("Model Type", ["xgboost", "lightgbm"], index=0, key="cs_model_type")
        with col4:
            forward_days = st.slider("Forward Days (for labels)", 3, 10, 5, key="cs_forward_days")
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_weight = st.slider("Max Weight per Stock", 0.05, 0.30, 0.15, step=0.05, key="cs_max_weight")
            allocation_mode = st.selectbox(
                "Allocation Mode",
                ["long_only", "market_neutral"],
                index=0,
                key="cs_allocation_mode"
            )
        with col2:
            run_validation = st.checkbox("Run Walk-Forward Validation", value=True, key="cs_validation")
            if run_validation:
                n_splits = st.slider("Validation Folds", 3, 10, 5, key="cs_n_splits")
    
    if st.button("Generate Cross-Sectional Alpha", type="primary", use_container_width=True):
        with st.spinner("Fetching data and training cross-sectional model..."):
            try:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
                
                if len(symbols) < 3:
                    st.error("Please provide at least 3 stock symbols for cross-sectional analysis.")
                    st.stop()
                
                # Fetch data for all stocks
                stocks_data = {}
                progress_bar = st.progress(0)
                for i, symbol in enumerate(symbols):
                    try:
                        data = st.session_state.data_fetcher.get_stock_data(symbol, period=period)
                        if not data.empty:
                            stocks_data[symbol] = data
                        progress_bar.progress((i + 1) / len(symbols))
                    except Exception as e:
                        st.warning(f"Could not fetch data for {symbol}: {e}")
                
                progress_bar.empty()
                
                if len(stocks_data) < 3:
                    st.error(f"Insufficient data: got {len(stocks_data)} stocks, need at least 3 for cross-sectional analysis.")
                    st.stop()
                
                # Initialize and train cross-sectional alpha engine
                engine = CrossSectionalAlphaEngine(model_type=model_type)
                
                with st.expander("Model Training", expanded=False):
                    try:
                        train_results = engine.train(
                            stocks_data,
                            forward_days=forward_days,
                            validation_size=0.2
                        )
                        
                        st.success("Cross-sectional alpha model trained successfully!")
                        
                        # Display training metrics
                        st.markdown("""
                        <div style="margin: 2rem 0;">
                            <h3 style="color: #0A2463; margin-bottom: 1rem;">Training Performance</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rank IC", f"{train_results['rank_ic']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{train_results['rmse']:.6f}")
                        with col3:
                            st.metric("MAE", f"{train_results['mae']:.6f}")
                        with col4:
                            significance = "✓" if train_results.get('rank_ic', 0) > 0.05 else "✗"
                            st.metric("Signal Quality", significance)
                        
                        # Feature importance
                        if train_results.get('feature_importance'):
                            st.markdown("**Top Features:**")
                            importance_df = pd.DataFrame({
                                'Feature': list(train_results['feature_importance'].keys()),
                                'Importance': list(train_results['feature_importance'].values())
                            }).sort_values('Importance', ascending=False).head(10)
                            st.dataframe(importance_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Training error: {e}")
                        with st.expander("Show detailed error"):
                            st.code(traceback.format_exc())
                        st.stop()
                
                # Store engine in session state
                st.session_state.cross_sectional_engine = engine
                
                # Get most recent date
                all_dates = []
                for data in stocks_data.values():
                    if isinstance(data.index, pd.DatetimeIndex):
                        all_dates.extend(data.index.tolist())
                
                if not all_dates:
                    st.error("No valid dates found in data.")
                    st.stop()
                
                target_date = pd.Timestamp(max(all_dates))
                
                # Predict alpha for target date
                with st.spinner("Generating alpha signals..."):
                    alpha_raw_df = engine.predict_alpha(stocks_data, target_date=target_date)
                
                # Apply risk adjustment
                with st.spinner("Applying risk adjustment..."):
                    risk_adjustment_layer = RiskAdjustmentLayer()
                    risk_adjusted_df = risk_adjustment_layer.adjust_alpha(
                        alpha_raw_df,
                        stocks_data,
                        target_date=target_date
                    )
                
                # Portfolio allocation
                with st.spinner("Calculating portfolio weights..."):
                    allocator = PortfolioAllocator(
                        mode=allocation_mode,
                        max_weight=max_weight
                    )
                    allocation_df = allocator.allocate(
                        risk_adjusted_df,
                        target_date=target_date
                    )
                
                # Display results
                st.markdown("""
                <div style="margin: 2rem 0;">
                    <h2 style="color: #0A2463; margin-bottom: 1rem;">Cross-Sectional Alpha Results</h2>
                    <p style="color: #6B7280; font-size: 0.875rem;">
                        <strong>Date:</strong> {}</p>
                </div>
                """.format(target_date.strftime('%Y-%m-%d')), unsafe_allow_html=True)
                
                # Results table
                results_df = allocation_df[allocation_df['date'] == target_date].copy()
                results_df = results_df.sort_values('risk_adjusted_alpha', ascending=False)
                
                # Highlight top 20% and bottom 20%
                n = len(results_df)
                top_n = max(1, int(n * 0.2))
                bottom_n = max(1, int(n * 0.2))
                
                results_df['rank'] = range(1, len(results_df) + 1)
                results_df = results_df[['rank', 'ticker', 'alpha_raw', 'alpha_z', 'volatility', 'risk_adjusted_alpha', 'weight']]
                results_df.columns = ['Rank', 'Ticker', 'Alpha Raw', 'Alpha Z-Score', 'Volatility', 'Risk-Adj Alpha', 'Weight']
                
                # Format for display
                display_df = results_df.copy()
                display_df['Alpha Raw'] = display_df['Alpha Raw'].apply(lambda x: f"{x:.6f}")
                display_df['Alpha Z-Score'] = display_df['Alpha Z-Score'].apply(lambda x: f"{x:.4f}")
                display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.4f}")
                display_df['Risk-Adj Alpha'] = display_df['Risk-Adj Alpha'].apply(lambda x: f"{x:.4f}")
                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Summary statistics
                st.markdown("""
                <div style="margin: 2rem 0;">
                    <h3 style="color: #0A2463; margin-bottom: 1rem;">Allocation Summary</h3>
                </div>
                """, unsafe_allow_html=True)
                
                summary = allocator.get_allocation_summary(allocation_df, target_date=target_date)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Stocks", summary['total_stocks'])
                with col2:
                    st.metric("Long Positions", summary['long_positions'])
                with col3:
                    st.metric("Max Weight", f"{summary['max_weight']:.2%}")
                with col4:
                    st.metric("Concentration", f"{summary['concentration']:.4f}")
                
                # Top and bottom stocks
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top 5 by Risk-Adjusted Alpha:**")
                    top_5 = results_df.head(5)[['Ticker', 'Risk-Adj Alpha', 'Weight']]
                    st.dataframe(top_5, use_container_width=True)
                
                with col2:
                    st.markdown("**Bottom 5 by Risk-Adjusted Alpha:**")
                    bottom_5 = results_df.tail(5)[['Ticker', 'Risk-Adj Alpha', 'Weight']]
                    st.dataframe(bottom_5, use_container_width=True)
                
                # Visualization
                st.markdown("""
                <div style="margin: 2rem 0;">
                    <h3 style="color: #0A2463; margin-bottom: 1rem;">Alpha Distribution</h3>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df['Ticker'],
                    y=results_df['Risk-Adj Alpha'],
                    name='Risk-Adjusted Alpha',
                    marker_color='#0A2463'
                ))
                fig.update_layout(
                    title=dict(text="Risk-Adjusted Alpha by Stock", font=dict(size=16, color='#0A2463')),
                    xaxis=dict(title="Ticker"),
                    yaxis=dict(title="Risk-Adjusted Alpha"),
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Portfolio weights visualization
                st.markdown("""
                <div style="margin: 2rem 0;">
                    <h3 style="color: #0A2463; margin-bottom: 1rem;">Portfolio Weights</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Get weights from allocation_df (before formatting)
                weights_data = allocation_df[allocation_df['date'] == target_date].copy()
                weights_data = weights_data[weights_data['weight'].abs() > 0.001].copy()  # Filter near-zero weights
                weights_data = weights_data.sort_values('weight', ascending=False)
                
                if len(weights_data) > 0:
                    fig_weights = px.pie(
                        weights_data,
                        values='weight',
                        names='ticker',
                        title="Portfolio Weight Allocation"
                    )
                    fig_weights.update_layout(
                        title_font=dict(size=16, color='#0A2463'),
                        height=400,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                
                # Important disclaimer
                st.markdown("""
                <div style="margin: 2rem 0; padding: 1rem; background: #FEF3C7; border-radius: 8px; border-left: 4px solid #F59E0B;">
                    <p style="margin: 0; color: #92400E; font-size: 0.875rem;">
                        <strong>⚠️ Research Signal Only:</strong> This module generates relative ranking signals for risk allocation research. 
                        ML outputs are treated as noisy alpha signals that require risk adjustment. 
                        This is NOT investment advice, NOT price prediction, and NOT a trading system. 
                        Portfolio construction, not prediction accuracy, drives performance in practice.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"**Error**: {str(e)}")
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

# ============================================================================

st.markdown("""
<div class="footer">
    <p style="margin: 0; color: #6B7280;">
        <strong style="color: #0A2463;">FinAI Pro</strong> | Enterprise Financial Intelligence Platform<br>
        <span style="font-size: 0.75rem;">This platform is for educational and demonstration purposes only. 
        Stock predictions do not constitute investment advice.</span>
    </p>
</div>
""", unsafe_allow_html=True)
