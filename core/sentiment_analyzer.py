"""
Financial News Sentiment Analysis Module

Analyzes sentiment of financial news articles using NLP techniques.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyzes sentiment of financial news articles."""
    
    def __init__(self) -> None:
        """Initialize sentiment analyzer with VADER."""
        self.vader = SentimentIntensityAnalyzer()
    
    def fetch_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch stock-related news articles from multiple sources.
        Tries Alpha Vantage first, then RSS feeds as fallback.
        
        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of article dictionaries
        """
        articles = []
        
        # Try Alpha Vantage first (if API key available)
        try:
            articles = self._fetch_alpha_vantage_news(symbol, max_articles)
            if articles:
                return articles[:max_articles]
        except Exception as e:
            print(f"Alpha Vantage news failed: {e}")
        
        # Fallback to RSS feeds (no rate limits)
        try:
            articles = self._fetch_rss_news(symbol, max_articles)
            if articles:
                return articles[:max_articles]
        except Exception as e:
            print(f"RSS news failed: {e}")
        
        return articles[:max_articles] if articles else []
    
    def _fetch_alpha_vantage_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """Fetch news from Alpha Vantage NEWS_SENTIMENT API."""
        import requests
        import os
        import time
        
        # Get API key
        api_key = None
        try:
            from config import ALPHA_VANTAGE_API_KEY as CONFIG_KEY
            if CONFIG_KEY and CONFIG_KEY != 'demo' and CONFIG_KEY != '':
                api_key = CONFIG_KEY
        except ImportError:
            pass
        
        if not api_key:
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not api_key or api_key == 'demo':
            raise Exception("Alpha Vantage API key not found")
        
        # Rate limit: wait 12 seconds
        time.sleep(12)
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key,
            'limit': min(max_articles, 50)  # Alpha Vantage allows up to 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise Exception(f"Rate limit: {data['Note']}")
            
            if 'feed' not in data:
                raise Exception("No news feed in response")
            
            articles = []
            for item in data['feed'][:max_articles]:
                articles.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('source', ''),
                    'link': item.get('url', ''),
                    'published': item.get('time_published', ''),
                    'summary': item.get('summary', '')
                })
            
            return articles
        except Exception as e:
            raise Exception(f"Alpha Vantage news API error: {str(e)}")
    
    def _fetch_rss_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """Fetch news from RSS feeds (no rate limits)."""
        import feedparser
        from datetime import datetime
        
        articles = []
        
        # Multiple RSS sources for better coverage
        rss_sources = [
            f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US',
            f'https://www.google.com/finance/quote/{symbol}:NASDAQ?output=rss',
        ]
        
        for rss_url in rss_sources:
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:max_articles]:
                    # Parse published date
                    published_time = 0
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published_time = int(datetime(*entry.published_parsed[:6]).timestamp())
                        except:
                            pass
                    
                    articles.append({
                        'title': entry.get('title', ''),
                        'publisher': entry.get('source', {}).get('title', 'RSS Feed') if hasattr(entry, 'source') else 'RSS Feed',
                        'link': entry.get('link', ''),
                        'published': published_time,
                        'summary': entry.get('summary', '')
                    })
                
                # If we got enough articles, break
                if len(articles) >= max_articles:
                    break
                    
            except Exception as e:
                print(f"RSS feed error ({rss_url}): {e}")
                continue
        
        return articles
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for analysis.
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with compound, pos, neu, neg scores
        """
        scores = self.vader.polarity_scores(text)
        return scores
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with polarity and subjectivity
        """
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a single article.
        
        Args:
            article: Article dictionary with title and other fields
        
        Returns:
            Dictionary with sentiment analysis results
        """
        title = article.get('title', '')
        cleaned_title = self.clean_text(title)
        
        # VADER analysis
        vader_scores = self.analyze_sentiment_vader(cleaned_title)
        
        # TextBlob analysis
        textblob_scores = self.analyze_sentiment_textblob(cleaned_title)
        
        # Combined score (weighted average)
        compound_score = vader_scores['compound']
        polarity_score = textblob_scores['polarity']
        combined_score = (compound_score * 0.6 + polarity_score * 0.4)
        
        # Determine sentiment category
        if combined_score >= 0.1:
            sentiment_label = 'Positive'
        elif combined_score <= -0.1:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'title': title,
            'publisher': article.get('publisher', ''),
            'link': article.get('link', ''),
            'vader_compound': compound_score,
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'textblob_polarity': polarity_score,
            'textblob_subjectivity': textblob_scores['subjectivity'],
            'combined_score': combined_score,
            'sentiment': sentiment_label
        }
    
    def analyze_stock_news(
        self, 
        symbol: str, 
        max_articles: int = 10
    ) -> Dict:
        """
        Analyze overall sentiment of stock-related news.
        
        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to analyze
        
        Returns:
            Dictionary with analysis results and summary statistics
        """
        articles = self.fetch_news(symbol, max_articles)
        
        if not articles:
            return {
                'articles': [],
                'summary': {
                    'total_articles': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'avg_sentiment': 0,
                    'overall_sentiment': 'No Data'
                }
            }
        
        analyzed_articles = []
        sentiment_scores = []
        
        for article in articles:
            analysis = self.analyze_article(article)
            analyzed_articles.append(analysis)
            sentiment_scores.append(analysis['combined_score'])
        
        # Calculate summary statistics
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        positive_count = sum(1 for a in analyzed_articles if a['sentiment'] == 'Positive')
        negative_count = sum(1 for a in analyzed_articles if a['sentiment'] == 'Negative')
        neutral_count = sum(1 for a in analyzed_articles if a['sentiment'] == 'Neutral')
        
        if avg_sentiment >= 0.1:
            overall_sentiment = 'Positive'
        elif avg_sentiment <= -0.1:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Neutral'
        
        return {
            'articles': analyzed_articles,
            'summary': {
                'total_articles': len(analyzed_articles),
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count,
                'avg_sentiment': avg_sentiment,
                'overall_sentiment': overall_sentiment
            }
        }
    
    def get_sentiment_impact(self, sentiment_score: float) -> str:
        """
        Assess market impact based on sentiment score.
        
        Args:
            sentiment_score: Sentiment score
        
        Returns:
            String describing market impact
        """
        if sentiment_score >= 0.3:
            return "Strongly Bullish - May drive stock price up"
        elif sentiment_score >= 0.1:
            return "Bullish - May have positive impact on stock price"
        elif sentiment_score >= -0.1:
            return "Neutral - Limited impact on stock price"
        elif sentiment_score >= -0.3:
            return "Bearish - May have negative impact on stock price"
        else:
            return "Strongly Bearish - May drive stock price down"
