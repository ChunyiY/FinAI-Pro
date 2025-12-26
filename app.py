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
import warnings
import traceback

warnings.filterwarnings('ignore')

# Import custom modules - Use Real Data Fetcher (only real market data)
try:
    from real_data_fetcher import RealDataFetcher
    USE_REAL_FETCHER = True
except ImportError:
    try:
        from smart_data_fetcher import SmartDataFetcher
        USE_REAL_FETCHER = False
        USE_SMART_FETCHER = True
    except ImportError:
        try:
            from multi_source_fetcher import MultiSourceDataFetcher
            USE_REAL_FETCHER = False
            USE_SMART_FETCHER = False
            USE_MULTI_SOURCE = True
        except ImportError:
            from data_fetcher import StockDataFetcher
            USE_REAL_FETCHER = False
            USE_SMART_FETCHER = False
            USE_MULTI_SOURCE = False

from stock_predictor import StockPredictor
from sentiment_analyzer import SentimentAnalyzer
from portfolio_optimizer import PortfolioOptimizer
from utils import format_currency, format_percentage

# ============================================================================
# BRAND DESIGN SYSTEM & PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinAI Pro | Enterprise Financial Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "FinAI Pro - Enterprise Financial Intelligence Platform"
    }
)

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
            <span style="font-size: 2.5rem;">💼</span>
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

st.sidebar.markdown("""
<div style="padding: 1rem 0; border-bottom: 2px solid #E5E7EB; margin-bottom: 1.5rem;">
    <h2 style="margin: 0; color: #0A2463; font-size: 1.25rem;">Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Module",
    ["🏠 Dashboard", "📈 Stock Prediction", "📰 Sentiment Analysis", "💼 Portfolio Optimization", "📊 Market Analysis"],
    label_visibility="collapsed"
)

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

# Dashboard / Home page
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="content-card">
        <h1 style="color: #0A2463; margin-bottom: 1rem;">Welcome to FinAI Pro</h1>
        <p style="font-size: 1.125rem; color: #6B7280; line-height: 1.75; margin-bottom: 2rem;">
            Your comprehensive AI-powered financial intelligence platform. Leverage advanced machine learning, 
            natural language processing, and quantitative analysis to make informed investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem;">📈</span>
                <h2 style="margin: 0; color: #0A2463;">Stock Prediction</h2>
            </div>
            <p style="color: #6B7280; line-height: 1.75;">
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

        st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem;">💼</span>
                <h2 style="margin: 0; color: #0A2463;">Portfolio Optimization</h2>
            </div>
            <p style="color: #6B7280; line-height: 1.75;">
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

    with col2:
        st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem;">📰</span>
                <h2 style="margin: 0; color: #0A2463;">Sentiment Analysis</h2>
            </div>
            <p style="color: #6B7280; line-height: 1.75;">
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

        st.markdown("""
        <div class="content-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem;">📊</span>
                <h2 style="margin: 0; color: #0A2463;">Market Analysis</h2>
            </div>
            <p style="color: #6B7280; line-height: 1.75;">
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

# Stock Prediction page
elif page == "📈 Stock Prediction":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #0A2463; margin-bottom: 0.5rem;">Stock Price Prediction</h1>
        <p style="color: #6B7280; font-size: 1.125rem;">AI-powered forecasting using advanced LSTM neural networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT", key="pred_symbol")
        with col2:
            period = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1, key="pred_period")
        with col3:
            predict_days = st.slider("Prediction Days", 7, 60, 30, key="pred_days")
    
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("🔄 Fetching data and training model..."):
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
                    
                    # Train model
                    with st.expander("🔧 Model Training Progress", expanded=False):
                        train_losses = st.session_state.predictor.train(data, epochs=30)
                        st.line_chart(train_losses)
                    
                    # Evaluate model
                    metrics = st.session_state.predictor.evaluate(data)
                    
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #0A2463; margin-bottom: 1rem;">Model Performance</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
                    with col2:
                        st.metric("MAE", f"${metrics['MAE']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    with col4:
                        accuracy = max(0, 100 - metrics['MAPE'])
                        st.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    # Predict future prices
                    predictions = st.session_state.predictor.predict(data, days=predict_days)
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
                    st.error("⚠️ **Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("⚠️ **Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Sentiment Analysis page
elif page == "📰 Sentiment Analysis":
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
        with st.spinner("🔄 Fetching and analyzing news..."):
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
                        with st.expander(f"📰 {row['Title']} - **{row['Sentiment']}** ({row['Score']:.3f})"):
                            st.write(f"**Publisher**: {row['Publisher']}")
                            st.write(f"**Sentiment Score**: {row['Score']:.3f}")
                            st.write(f"**Link**: [{row['Link']}]({row['Link']})")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    st.error("⚠️ **Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("⚠️ **Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Portfolio Optimization page
elif page == "💼 Portfolio Optimization":
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
        with st.spinner("🔄 Analyzing and optimizing portfolio..."):
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
                    st.error("⚠️ **Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("⚠️ **Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# Market Analysis page
elif page == "📊 Market Analysis":
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
    
    if st.button("📊 Load Market Data", type="primary", use_container_width=True):
        with st.spinner("🔄 Fetching market data..."):
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
                    st.error("⚠️ **Rate Limit Exceeded** - Please wait a moment and try again.")
                elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                    st.error("⚠️ **Network Error** - Please check your connection and try again.")
                else:
                    st.error(f"**Error**: {str(e)}")
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())

# ============================================================================
# FOOTER
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
