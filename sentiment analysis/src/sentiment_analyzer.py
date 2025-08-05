"""
Module for sentiment analysis of text data.
"""
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    def __init__(self, algorithm="vader", verbose=False):
        """
        Initialize the sentiment analyzer.
        
        Args:
            algorithm (str): The sentiment analysis algorithm to use.
                Options: "vader", "textblob"
            verbose (bool): Whether to print verbose output during analysis
        """
        self.algorithm = algorithm.lower()
        self.verbose = verbose
        
        if self.verbose:
            print(f"Initializing SentimentAnalyzer with {self.algorithm} algorithm")
        
        if self.algorithm == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
            if self.verbose:
                print("VADER SentimentIntensityAnalyzer initialized")
        elif self.algorithm == "textblob":
            if self.verbose:
                print("TextBlob analyzer ready")
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'vader' or 'textblob'.")
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores and analysis
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        if self.verbose:
            print(f"Analyzing sentiment of text ({len(text)} characters)")
        
        if self.algorithm == "vader":
            return self._analyze_with_vader(text)
        elif self.algorithm == "textblob":
            return self._analyze_with_textblob(text)
    
    def _analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        if self.verbose:
            print("Running VADER sentiment analysis algorithm...")
            
        # Get sentiment scores
        sentiment_scores = self.analyzer.polarity_scores(text)
        
        if self.verbose:
            print("VADER analysis complete")
            print(f"Negative score: {sentiment_scores['neg']:.4f}")
            print(f"Neutral score: {sentiment_scores['neu']:.4f}")
            print(f"Positive score: {sentiment_scores['pos']:.4f}")
            print(f"Compound score: {sentiment_scores['compound']:.4f}")
            
            # Determine overall sentiment
            if sentiment_scores['compound'] >= 0.05:
                print("Overall sentiment: Positive")
                sentiment = "positive"
            elif sentiment_scores['compound'] <= -0.05:
                print("Overall sentiment: Negative")
                sentiment = "negative"
            else:
                print("Overall sentiment: Neutral")
                sentiment = "neutral"
        else:
            # Determine overall sentiment without printing
            if sentiment_scores['compound'] >= 0.05:
                sentiment = "positive"
            elif sentiment_scores['compound'] <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "algorithm": "vader",
            "scores": sentiment_scores,
            "sentiment": sentiment
        }
    
    def _analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        if self.verbose:
            print("Running TextBlob sentiment analysis algorithm...")
            
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentiment
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if self.verbose:
            print("TextBlob analysis complete")
            print(f"Polarity score: {polarity:.4f}")
            print(f"Subjectivity score: {subjectivity:.4f}")
            
            # Determine overall sentiment
            if polarity > 0.05:
                print("Overall sentiment: Positive")
                sentiment = "positive"
            elif polarity < -0.05:
                print("Overall sentiment: Negative")
                sentiment = "negative"
            else:
                print("Overall sentiment: Neutral")
                sentiment = "neutral"
        else:
            # Determine overall sentiment without printing
            if polarity > 0.05:
                sentiment = "positive"
            elif polarity < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "algorithm": "textblob",
            "scores": {
                "polarity": polarity,
                "subjectivity": subjectivity
            },
            "sentiment": sentiment
        }
    
    def analyze_text_chunks(self, text, chunk_size=1000):
        """
        Split text into chunks and analyze sentiment for each chunk.
        
        Args:
            text (str): Text to analyze
            chunk_size (int): Size of each chunk in characters
            
        Returns:
            list: List of sentiment analysis results for each chunk
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        if self.verbose:
            print(f"Text split into {len(chunks)} chunks of approximately {chunk_size} characters each")
        
        # Analyze each chunk
        results = []
        for i, chunk in enumerate(chunks):
            if self.verbose:
                print(f"\nAnalyzing chunk {i+1}/{len(chunks)}")
            
            result = self.analyze_text(chunk)
            results.append(result)
        
        return results