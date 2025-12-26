# 🚀 Quick Start Guide

## Installation Steps

### 1. Ensure Python Environment
```bash
python3 --version  # Requires Python 3.8+
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```bash
python setup.py
```

Or manually:
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

### 4. Run the Application

**Method 1: Using the startup script**
```bash
./run.sh
```

**Method 2: Direct run**
```bash
streamlit run app.py
```

The application will automatically open in your browser at: http://localhost:8501

## 📖 Usage Examples

### Stock Prediction Example
1. Select "Stock Prediction" page
2. Enter stock symbol (e.g., AAPL, TSLA, MSFT)
3. Select data period and prediction days
4. Click "Start Prediction"

### Sentiment Analysis Example
1. Select "Sentiment Analysis" page
2. Enter stock symbol
3. Set number of articles to analyze
4. View sentiment analysis results

### Portfolio Optimization Example
1. Select "Portfolio Optimization" page
2. Enter multiple stock symbols (comma-separated)
3. Select optimization method
4. View optimal weight allocation

## 🎯 Interview Demo Suggestions

### Technical Highlights Display Order:

1. **Home Page Introduction** (30 seconds)
   - Show overall project architecture and tech stack
   - Explain practical value of the project

2. **Stock Prediction Demo** (2-3 minutes)
   - Show LSTM deep learning model
   - Explain model training process and evaluation metrics
   - Display prediction result visualizations

3. **Sentiment Analysis Demo** (1-2 minutes)
   - Show NLP technology applications
   - Explain multi-model sentiment analysis
   - Display market sentiment impact assessment

4. **Portfolio Optimization Demo** (2 minutes)
   - Show quantitative finance knowledge
   - Explain Modern Portfolio Theory
   - Display efficient frontier analysis

5. **Technical Details Discussion** (as time permits)
   - Model architecture selection rationale
   - Data processing and feature engineering
   - Performance optimization and scalability

## 💡 Common Interview Questions Preparation

### Q: Why choose LSTM over other models?
**A**: LSTM is suitable for time series data and can capture long-term dependencies. For stock prices with temporal dependencies, LSTM is more effective than simple linear models or traditional statistical methods.

### Q: How do you evaluate model accuracy?
**A**: We use multiple metrics: RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MAPE (Mean Absolute Percentage Error). MAPE is particularly suitable for financial scenarios as it provides percentage error, which is more intuitive.

### Q: What NLP techniques are used for sentiment analysis?
**A**: We combine two methods: VADER (specifically optimized for social media and news sentiment analysis) and TextBlob (general-purpose sentiment analysis library). We get more accurate results through weighted averaging.

### Q: What is the theoretical basis for portfolio optimization?
**A**: Based on Markowitz's Modern Portfolio Theory, we find optimal portfolios by optimizing the risk-return ratio (Sharpe ratio). We implemented three optimization strategies: maximizing Sharpe ratio, minimizing volatility, and maximizing return.

### Q: How scalable is the project?
**A**: 
- Can add more technical indicators
- Can integrate more data sources (e.g., options data, macroeconomic data)
- Can implement real-time trading strategy backtesting
- Can add user authentication and personalized recommendations

## 🔧 Troubleshooting

### Issue: Unable to fetch stock data
- Check network connection
- Verify stock symbol format is correct (e.g., AAPL not AAPL.US)
- yfinance API may have rate limits, try again later

### Issue: NLTK data download fails
- Check network connection
- Manually run `python setup.py`
- If SSL error occurs, may need to configure proxy

### Issue: Model training is slow
- This is normal, LSTM training takes time
- Can reduce training epochs or data volume for faster demo
- Can consider using GPU acceleration (if available)

## 📚 Further Learning

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
