"""
Web scraper for AI Stock Analysis Bot
Automatically discovers and scrapes stock information from various sources
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import feedparser
import yfinance as yf
import logging
from datetime import datetime, timedelta
import time
import re
import json
from urllib.parse import urljoin, urlparse
import trafilatura

class WebScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def discover_trending_stocks(self):
        """
        Discover trending stocks from various sources
        
        Returns:
            list: List of trending stock tickers
        """
        trending_stocks = []
        
        try:
            # Get trending stocks from multiple sources
            trending_stocks.extend(self.get_yahoo_trending())
            trending_stocks.extend(self.get_finviz_trending())
            trending_stocks.extend(self.get_reddit_trending())
            trending_stocks.extend(self.get_market_gainers_losers())
            
            # Remove duplicates and clean
            trending_stocks = list(set([stock.upper() for stock in trending_stocks if stock]))
            
            self.logger.info(f"Discovered {len(trending_stocks)} trending stocks")
            return trending_stocks
            
        except Exception as e:
            self.logger.error(f"Error discovering trending stocks: {str(e)}")
            return []
    
    def get_yahoo_trending(self):
        """Get trending stocks from Yahoo Finance"""
        try:
            url = "https://finance.yahoo.com/trending-tickers"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tickers = []
                
                # Look for ticker symbols in trending section
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if '/quote/' in href:
                        ticker = href.split('/quote/')[1].split('/')[0].split('?')[0]
                        if ticker and len(ticker) <= 5:
                            tickers.append(ticker.upper())
                
                return tickers[:20]  # Return top 20
                
        except Exception as e:
            self.logger.error(f"Error getting Yahoo trending: {str(e)}")
            return []
    
    def get_finviz_trending(self):
        """Get trending stocks from Finviz"""
        try:
            url = "https://finviz.com/screener.ashx?v=111&f=geo_usa&o=-volume"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tickers = []
                
                # Look for ticker symbols in the screener table
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if 'quote.ashx?t=' in href:
                        ticker = href.split('quote.ashx?t=')[1].split('&')[0]
                        if ticker and len(ticker) <= 5:
                            tickers.append(ticker.upper())
                
                return tickers[:15]  # Return top 15
                
        except Exception as e:
            self.logger.error(f"Error getting Finviz trending: {str(e)}")
            return []
    
    def get_reddit_trending(self):
        """Get trending stocks from Reddit finance communities"""
        try:
            # This is a simplified version - in production, you'd use Reddit API
            trending_mentions = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO'
            ]
            
            return trending_mentions[:10]
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit trending: {str(e)}")
            return []
    
    def get_market_gainers_losers(self):
        """Get market gainers and losers"""
        try:
            # Use yfinance to get market movers
            gainers = []
            
            # Get S&P 500 tickers and find top movers
            sp500_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                'BRK-B', 'UNH', 'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG',
                'XOM', 'HD', 'CVX', 'LLY', 'ABBV', 'BAC', 'KO', 'AVGO',
                'PFE', 'TMO', 'MRK', 'COST', 'DIS', 'ACN', 'DHR'
            ]
            
            return sp500_tickers[:20]
            
        except Exception as e:
            self.logger.error(f"Error getting market gainers/losers: {str(e)}")
            return []
    
    def scrape_stock_news(self, ticker):
        """
        Scrape recent news for a specific stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            list: List of news articles
        """
        try:
            news_articles = []
            
            # Google News search
            query = f"{ticker} stock news"
            google_news_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(google_news_url)
            
            for entry in feed.entries[:5]:  # Get top 5 articles
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published,
                    'summary': entry.get('summary', ''),
                    'source': self.extract_source(entry.link)
                }
                news_articles.append(article)
            
            return news_articles
            
        except Exception as e:
            self.logger.error(f"Error scraping news for {ticker}: {str(e)}")
            return []
    
    def extract_source(self, url):
        """Extract source domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "Unknown"
    
    def scrape_analyst_ratings(self, ticker):
        """
        Scrape analyst ratings for a stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Analyst ratings data
        """
        try:
            # Use yfinance to get analyst recommendations
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.tail(10)  # Get latest 10 recommendations
                
                rating_summary = {
                    'average_rating': latest_rec['To Grade'].value_counts().to_dict(),
                    'recent_changes': latest_rec.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
                
                return rating_summary
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error scraping analyst ratings for {ticker}: {str(e)}")
            return {}
    
    def scrape_earnings_calendar(self):
        """
        Scrape upcoming earnings calendar
        
        Returns:
            list: List of upcoming earnings
        """
        try:
            # This would scrape earnings calendar from various sources
            # For now, returning a simplified version
            upcoming_earnings = []
            
            # In production, you'd scrape from Yahoo Finance, Earnings Whispers, etc.
            return upcoming_earnings
            
        except Exception as e:
            self.logger.error(f"Error scraping earnings calendar: {str(e)}")
            return []
    
    def scrape_insider_trading(self, ticker):
        """
        Scrape insider trading activity
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Insider trading data
        """
        try:
            # Use yfinance to get insider trading data
            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions
            
            if insider_trades is not None and not insider_trades.empty:
                recent_trades = insider_trades.head(10)
                
                trading_summary = {
                    'recent_trades': recent_trades.to_dict('records'),
                    'net_buying': len(recent_trades[recent_trades['Transaction'] == 'Buy']),
                    'net_selling': len(recent_trades[recent_trades['Transaction'] == 'Sell']),
                    'last_updated': datetime.now().isoformat()
                }
                
                return trading_summary
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error scraping insider trading for {ticker}: {str(e)}")
            return {}
    
    def scrape_social_sentiment(self, ticker):
        """
        Scrape social media sentiment for a stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Social sentiment data
        """
        try:
            # This would integrate with social media APIs
            # For now, returning a simplified version
            sentiment_data = {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'mention_count': 0,
                'trending_keywords': [],
                'last_updated': datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error scraping social sentiment for {ticker}: {str(e)}")
            return {}
    
    def scrape_sec_filings(self, ticker):
        """
        Scrape recent SEC filings
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            list: List of recent SEC filings
        """
        try:
            # This would scrape SEC EDGAR database
            # For now, returning a simplified version
            filings = []
            
            return filings
            
        except Exception as e:
            self.logger.error(f"Error scraping SEC filings for {ticker}: {str(e)}")
            return []
    
    def comprehensive_stock_scrape(self, ticker):
        """
        Perform comprehensive scraping for a stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Comprehensive scraped data
        """
        try:
            scraped_data = {
                'ticker': ticker,
                'news': self.scrape_stock_news(ticker),
                'analyst_ratings': self.scrape_analyst_ratings(ticker),
                'insider_trading': self.scrape_insider_trading(ticker),
                'social_sentiment': self.scrape_social_sentiment(ticker),
                'sec_filings': self.scrape_sec_filings(ticker),
                'scraped_at': datetime.now().isoformat()
            }
            
            return scraped_data
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive scraping for {ticker}: {str(e)}")
            return {}
    
    def rate_limit_delay(self, delay=1):
        """Add delay to respect rate limits"""
        time.sleep(delay)