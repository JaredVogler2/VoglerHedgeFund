# ml_trading_sentiment.py

"""
ML Trading System - News Sentiment Analysis with OpenAI
Professional news analysis for market sentiment and event detection
"""

import openai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import feedparser
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import defaultdict
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    symbols: List[str]
    
@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float
    sentiment_label: str  # positive, negative, neutral
    key_events: List[str]
    price_impact: str  # high, medium, low
    reasoning: str

# =============================================================================
# NEWS FETCHER
# =============================================================================

class NewsDataFetcher:
    """Fetches news from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'yahoo_finance': self._fetch_yahoo_finance,
            'seeking_alpha': self._fetch_seeking_alpha,
            'benzinga': self._fetch_benzinga,
            'reuters': self._fetch_reuters_rss,
            'bloomberg': self._fetch_bloomberg_rss
        }
        
        # Rate limiting
        self.rate_limits = {
            'yahoo_finance': {'calls': 100, 'period': 3600},
            'seeking_alpha': {'calls': 50, 'period': 3600},
            'benzinga': {'calls': 100, 'period': 3600}
        }
        self.call_history = defaultdict(list)
        
    def fetch_news_for_symbols(self, symbols: List[str], 
                             hours_back: int = 24) -> Dict[str, List[NewsArticle]]:
        """Fetch news for multiple symbols"""
        all_news = defaultdict(list)
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for symbol in symbols:
                for source_name, fetch_func in self.sources.items():
                    if self._check_rate_limit(source_name):
                        future = executor.submit(
                            self._fetch_with_error_handling,
                            fetch_func, symbol, hours_back, source_name
                        )
                        futures.append((future, symbol))
            
            # Collect results
            for future, symbol in futures:
                try:
                    articles = future.result(timeout=10)
                    if articles:
                        all_news[symbol].extend(articles)
                except Exception as e:
                    logger.error(f"Error fetching news for {symbol}: {e}")
        
        # Sort by date and remove duplicates
        for symbol in all_news:
            all_news[symbol] = self._deduplicate_articles(all_news[symbol])
            all_news[symbol].sort(key=lambda x: x.published_date, reverse=True)
        
        return dict(all_news)
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're within rate limits"""
        if source not in self.rate_limits:
            return True
        
        limit = self.rate_limits[source]
        now = time.time()
        
        # Remove old calls
        self.call_history[source] = [
            call_time for call_time in self.call_history[source]
            if now - call_time < limit['period']
        ]
        
        # Check if we can make another call
        if len(self.call_history[source]) < limit['calls']:
            self.call_history[source].append(now)
            return True
        
        return False
    
    def _fetch_with_error_handling(self, fetch_func, symbol: str, 
                                 hours_back: int, source: str) -> List[NewsArticle]:
        """Fetch with error handling"""
        try:
            return fetch_func(symbol, hours_back)
        except Exception as e:
            logger.error(f"Error fetching from {source} for {symbol}: {e}")
            return []
    
    def _fetch_yahoo_finance(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        articles = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            cutoff_date = datetime.now() - timedelta(hours=hours_back)
            
            for item in news:
                # Parse publication date
                pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if pub_date < cutoff_date:
                    continue
                
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('summary', ''),
                    source='Yahoo Finance',
                    url=item.get('link', ''),
                    published_date=pub_date,
                    symbols=[symbol]
                )
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {symbol}: {e}")
        
        return articles
    
    def _fetch_seeking_alpha(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch from Seeking Alpha RSS"""
        articles = []
        
        try:
            # Seeking Alpha RSS feed
            url = f"https://seekingalpha.com/api/sa/combined/{symbol}.xml"
            feed = feedparser.parse(url)
            
            cutoff_date = datetime.now() - timedelta(hours=hours_back)
            
            for entry in feed.entries:
                # Parse date
                pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
                pub_date = pub_date.replace(tzinfo=None)  # Remove timezone
                
                if pub_date < cutoff_date:
                    continue
                
                # Extract content
                content = entry.summary if hasattr(entry, 'summary') else ''
                content = self._clean_html(content)
                
                article = NewsArticle(
                    title=entry.title,
                    content=content,
                    source='Seeking Alpha',
                    url=entry.link,
                    published_date=pub_date,
                    symbols=[symbol]
                )
                
                articles.append(article)
                
        except Exception as e:
            logger.debug(f"Error fetching Seeking Alpha news for {symbol}: {e}")
        
        return articles
    
    def _fetch_benzinga(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch from Benzinga RSS"""
        articles = []
        
        try:
            url = f"https://feeds.benzinga.com/benzinga/news/feed/{symbol}"
            feed = feedparser.parse(url)
            
            cutoff_date = datetime.now() - timedelta(hours=hours_back)
            
            for entry in feed.entries[:10]:  # Limit to recent articles
                # Parse date
                try:
                    pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
                    pub_date = pub_date.replace(tzinfo=None)
                except:
                    continue
                
                if pub_date < cutoff_date:
                    continue
                
                content = entry.summary if hasattr(entry, 'summary') else ''
                
                article = NewsArticle(
                    title=entry.title,
                    content=self._clean_html(content),
                    source='Benzinga',
                    url=entry.link,
                    published_date=pub_date,
                    symbols=[symbol]
                )
                
                articles.append(article)
                
        except Exception as e:
            logger.debug(f"Error fetching Benzinga news for {symbol}: {e}")
        
        return articles
    
    def _fetch_reuters_rss(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch from Reuters RSS feeds"""
        articles = []
        
        # Reuters business news feed
        feeds = [
            "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
            "https://www.reuters.com/business/rss"
        ]
        
        cutoff_date = datetime.now() - timedelta(hours=hours_back)
        
        for feed_url in feeds:
            try:
                url = feed_url.format(symbol=symbol)
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:5]:
                    # Check if symbol is mentioned
                    if symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                        continue
                    
                    # Parse date
                    try:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        else:
                            continue
                    except:
                        continue
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    article = NewsArticle(
                        title=entry.title,
                        content=entry.get('summary', ''),
                        source='Reuters/Google News',
                        url=entry.link,
                        published_date=pub_date,
                        symbols=[symbol]
                    )
                    
                    articles.append(article)
                    
            except Exception as e:
                logger.debug(f"Error fetching Reuters news: {e}")
        
        return articles
    
    def _fetch_bloomberg_rss(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch from Bloomberg RSS"""
        # Note: Bloomberg has limited free access
        # This is a placeholder for when API access is available
        return []
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title
            title_key = article.title.lower().strip()
            
            # Check for very similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_similarity(title_key, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title_key)
        
        return unique_articles
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        # Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0

# =============================================================================
# OPENAI SENTIMENT ANALYZER
# =============================================================================

class OpenAISentimentAnalyzer:
    """Analyzes news sentiment using OpenAI GPT"""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4"  # or "gpt-3.5-turbo" for lower cost
        
        # Cache for recent analyses
        self.sentiment_cache = {}
        self.cache_duration = 3600  # 1 hour
        
    def analyze_articles(self, articles: List[NewsArticle], 
                        symbol: str) -> SentimentResult:
        """Analyze sentiment for a collection of articles"""
        
        if not articles:
            return self._create_neutral_result(symbol)
        
        # Check cache
        cache_key = self._create_cache_key(articles, symbol)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Prepare articles for analysis
        article_summaries = self._prepare_article_summaries(articles[:10])  # Limit to 10 most recent
        
        # Create prompt
        prompt = self._create_analysis_prompt(symbol, article_summaries)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500
            )
            
            # Parse response
            result = self._parse_response(response.choices[0].message.content, symbol)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in OpenAI sentiment analysis: {e}")
            return self._create_neutral_result(symbol)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis"""
        return """You are a professional financial analyst specializing in news sentiment analysis 
        for stock trading. Your task is to analyze news articles and provide:
        
        1. Overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
        2. Confidence level (0 to 1)
        3. Sentiment label (positive, negative, or neutral)
        4. Key events mentioned (earnings, product launch, legal issues, etc.)
        5. Expected price impact (high, medium, low)
        6. Brief reasoning for your assessment
        
        Focus on information that would impact stock price in the next 1-5 trading days.
        Be objective and base your analysis on factual information.
        
        Respond in JSON format."""
    
    def _create_analysis_prompt(self, symbol: str, 
                              article_summaries: List[Dict]) -> str:
        """Create prompt for analysis"""
        prompt = f"Analyze the sentiment for {symbol} based on these recent news articles:\n\n"
        
        for i, summary in enumerate(article_summaries, 1):
            prompt += f"Article {i}:\n"
            prompt += f"Title: {summary['title']}\n"
            prompt += f"Summary: {summary['content']}\n"
            prompt += f"Source: {summary['source']}\n"
            prompt += f"Published: {summary['published']}\n\n"
        
        prompt += """Based on these articles, provide your analysis in the following JSON format:
        {
            "sentiment_score": float between -1 and 1,
            "confidence": float between 0 and 1,
            "sentiment_label": "positive" or "negative" or "neutral",
            "key_events": ["event1", "event2", ...],
            "price_impact": "high" or "medium" or "low",
            "reasoning": "brief explanation"
        }"""
        
        return prompt
    
    def _prepare_article_summaries(self, articles: List[NewsArticle]) -> List[Dict]:
        """Prepare article summaries for analysis"""
        summaries = []
        
        for article in articles:
            # Truncate content if too long
            content = article.content[:500] if len(article.content) > 500 else article.content
            
            summaries.append({
                'title': article.title,
                'content': content,
                'source': article.source,
                'published': article.published_date.strftime('%Y-%m-%d %H:%M')
            })
        
        return summaries
    
    def _parse_response(self, response_text: str, symbol: str) -> SentimentResult:
        """Parse OpenAI response into SentimentResult"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return SentimentResult(
                symbol=symbol,
                sentiment_score=float(response_data.get('sentiment_score', 0)),
                confidence=float(response_data.get('confidence', 0.5)),
                sentiment_label=response_data.get('sentiment_label', 'neutral'),
                key_events=response_data.get('key_events', []),
                price_impact=response_data.get('price_impact', 'low'),
                reasoning=response_data.get('reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return self._create_neutral_result(symbol)
    
    def _create_neutral_result(self, symbol: str) -> SentimentResult:
        """Create a neutral sentiment result"""
        return SentimentResult(
            symbol=symbol,
            sentiment_score=0.0,
            confidence=0.0,
            sentiment_label='neutral',
            key_events=[],
            price_impact='low',
            reasoning='Unable to analyze sentiment'
        )
    
    def _create_cache_key(self, articles: List[NewsArticle], symbol: str) -> str:
        """Create cache key for articles"""
        # Use titles and dates for cache key
        article_ids = [f"{a.title}_{a.published_date.timestamp()}" for a in articles[:5]]
        return f"{symbol}_{'_'.join(article_ids)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[SentimentResult]:
        """Get cached result if available and fresh"""
        if cache_key in self.sentiment_cache:
            cached_time, result = self.sentiment_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return result
        return None
    
    def _cache_result(self, cache_key: str, result: SentimentResult):
        """Cache analysis result"""
        self.sentiment_cache[cache_key] = (time.time(), result)
        
        # Clean old cache entries
        current_time = time.time()
        self.sentiment_cache = {
            k: v for k, v in self.sentiment_cache.items()
            if current_time - v[0] < self.cache_duration
        }

# =============================================================================
# SENTIMENT AGGREGATOR
# =============================================================================

class SentimentAggregator:
    """Aggregates sentiment across multiple timeframes and sources"""
    
    def __init__(self):
        self.history = defaultdict(list)
        
    def add_sentiment(self, symbol: str, sentiment: SentimentResult, 
                     timestamp: datetime = None):
        """Add a sentiment result to history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.history[symbol].append({
            'timestamp': timestamp,
            'sentiment': sentiment
        })
        
        # Keep only last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        self.history[symbol] = [
            item for item in self.history[symbol]
            if item['timestamp'] > cutoff
        ]
    
    def get_aggregated_sentiment(self, symbol: str, 
                               hours_back: int = 24) -> Dict:
        """Get aggregated sentiment metrics"""
        if symbol not in self.history:
            return self._get_default_metrics()
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours_back)
        recent_sentiments = [
            item['sentiment'] for item in self.history[symbol]
            if item['timestamp'] > cutoff
        ]
        
        if not recent_sentiments:
            return self._get_default_metrics()
        
        # Calculate metrics
        scores = [s.sentiment_score for s in recent_sentiments]
        confidences = [s.confidence for s in recent_sentiments]
        
        # Weighted average by confidence
        if sum(confidences) > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
        else:
            weighted_score = np.mean(scores)
        
        # Sentiment momentum (trend)
        if len(scores) >= 2:
            momentum = scores[-1] - scores[0]
        else:
            momentum = 0
        
        # Event aggregation
        all_events = []
        for sentiment in recent_sentiments:
            all_events.extend(sentiment.key_events)
        
        # Count event types
        event_counts = defaultdict(int)
        for event in all_events:
            event_counts[event.lower()] += 1
        
        # Determine overall impact
        high_impact_count = sum(1 for s in recent_sentiments if s.price_impact == 'high')
        if high_impact_count >= 2:
            overall_impact = 'high'
        elif high_impact_count >= 1:
            overall_impact = 'medium'
        else:
            overall_impact = 'low'
        
        return {
            'current_sentiment': weighted_score,
            'sentiment_momentum': momentum,
            'average_confidence': np.mean(confidences),
            'num_articles': len(recent_sentiments),
            'key_events': dict(event_counts),
            'overall_impact': overall_impact,
            'sentiment_distribution': {
                'positive': sum(1 for s in recent_sentiments if s.sentiment_label == 'positive'),
                'negative': sum(1 for s in recent_sentiments if s.sentiment_label == 'negative'),
                'neutral': sum(1 for s in recent_sentiments if s.sentiment_label == 'neutral')
            }
        }
    
    def _get_default_metrics(self) -> Dict:
        """Get default metrics when no data available"""
        return {
            'current_sentiment': 0.0,
            'sentiment_momentum': 0.0,
            'average_confidence': 0.0,
            'num_articles': 0,
            'key_events': {},
            'overall_impact': 'low',
            'sentiment_distribution': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        }
    
    def get_sentiment_features(self, symbol: str) -> pd.Series:
        """Get sentiment features for ML models"""
        metrics = self.get_aggregated_sentiment(symbol, hours_back=24)
        
        # Create features
        features = pd.Series({
            'news_sentiment_score': metrics['current_sentiment'],
            'news_sentiment_momentum': metrics['sentiment_momentum'],
            'news_confidence': metrics['average_confidence'],
            'news_volume': metrics['num_articles'],
            'news_impact_high': 1 if metrics['overall_impact'] == 'high' else 0,
            'news_impact_medium': 1 if metrics['overall_impact'] == 'medium' else 0,
            'news_positive_ratio': metrics['sentiment_distribution']['positive'] / max(metrics['num_articles'], 1),
            'news_negative_ratio': metrics['sentiment_distribution']['negative'] / max(metrics['num_articles'], 1),
            'news_has_earnings': 1 if 'earnings' in str(metrics['key_events']).lower() else 0,
            'news_has_upgrade': 1 if 'upgrade' in str(metrics['key_events']).lower() else 0,
            'news_has_downgrade': 1 if 'downgrade' in str(metrics['key_events']).lower() else 0,
            'news_has_product': 1 if 'product' in str(metrics['key_events']).lower() else 0,
            'news_has_legal': 1 if 'legal' in str(metrics['key_events']).lower() or 'lawsuit' in str(metrics['key_events']).lower() else 0
        })
        
        return features

# =============================================================================
# NEWS SENTIMENT INTEGRATION
# =============================================================================

class NewsSentimentIntegration:
    """Integrates news sentiment into the trading system"""
    
    def __init__(self, config):
        self.config = config
        self.news_fetcher = NewsDataFetcher()
        self.sentiment_analyzer = OpenAISentimentAnalyzer(config.OPENAI_API_KEY)
        self.sentiment_aggregator = SentimentAggregator()
        
    def update_sentiment_for_watchlist(self, symbols: List[str]):
        """Update sentiment for entire watchlist"""
        logger.info(f"Updating sentiment for {len(symbols)} symbols...")
        
        # Fetch news
        all_news = self.news_fetcher.fetch_news_for_symbols(symbols, hours_back=24)
        
        # Analyze sentiment
        results = {}
        for symbol, articles in all_news.items():
            if articles:
                logger.info(f"Analyzing {len(articles)} articles for {symbol}")
                sentiment = self.sentiment_analyzer.analyze_articles(articles, symbol)
                self.sentiment_aggregator.add_sentiment(symbol, sentiment)
                results[symbol] = sentiment
            else:
                logger.info(f"No recent news for {symbol}")
        
        return results
    
    def get_sentiment_features_for_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """Get sentiment features for multiple symbols"""
        features_list = []
        
        for symbol in symbols:
            features = self.sentiment_aggregator.get_sentiment_features(symbol)
            features['symbol'] = symbol
            features_list.append(features)
        
        if features_list:
            return pd.DataFrame(features_list).set_index('symbol')
        else:
            return pd.DataFrame()
    
    def check_breaking_news(self, symbols: List[str]) -> List[Dict]:
        """Check for breaking news that might impact trading"""
        breaking_news = []
        
        # Fetch very recent news (last 2 hours)
        recent_news = self.news_fetcher.fetch_news_for_symbols(symbols, hours_back=2)
        
        for symbol, articles in recent_news.items():
            if articles:
                # Quick sentiment check on most recent article
                sentiment = self.sentiment_analyzer.analyze_articles(articles[:1], symbol)
                
                # Check if it's significant
                if (abs(sentiment.sentiment_score) > 0.7 and 
                    sentiment.confidence > 0.7 and
                    sentiment.price_impact in ['high', 'medium']):
                    
                    breaking_news.append({
                        'symbol': symbol,
                        'headline': articles[0].title,
                        'sentiment': sentiment.sentiment_score,
                        'impact': sentiment.price_impact,
                        'events': sentiment.key_events,
                        'timestamp': articles[0].published_date
                    })
        
        return breaking_news

# =============================================================================
# REAL-TIME NEWS MONITOR
# =============================================================================

class RealTimeNewsMonitor:
    """Monitors news in real-time during trading hours"""
    
    def __init__(self, news_integration: NewsSentimentIntegration, 
                 callback_func=None):
        self.news_integration = news_integration
        self.callback_func = callback_func
        self.is_monitoring = False
        self.monitored_symbols = []
        self.check_interval = 300  # 5 minutes
        
    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring news for symbols"""
        self.monitored_symbols = symbols
        self.is_monitoring = True
        
        logger.info(f"Starting real-time news monitoring for {len(symbols)} symbols")
        
        while self.is_monitoring:
            try:
                # Check for breaking news
                breaking_news = self.news_integration.check_breaking_news(symbols)
                
                if breaking_news:
                    logger.info(f"Found {len(breaking_news)} breaking news items")
                    
                    # Call callback if provided
                    if self.callback_func:
                        for news_item in breaking_news:
                            self.callback_func(news_item)
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("Stopped real-time news monitoring")

# Example usage
if __name__ == "__main__":
    logger.info("News Sentiment Analysis module loaded successfully")