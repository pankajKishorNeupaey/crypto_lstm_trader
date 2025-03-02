from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from config import NEWS_API_KEY, SYMBOLS  # Ensure NEWS_API_KEY and SYMBOLS are defined in config.py

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, symbol):
        """Fetch news headlines and return average sentiment score."""
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url).json()
            articles = response.get("articles", [])
            if not articles:
                return 0.0
            sentiments = [self.analyzer.polarity_scores(a["title"])["compound"] for a in articles]
            return sum(sentiments) / len(sentiments)
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            return 0.0