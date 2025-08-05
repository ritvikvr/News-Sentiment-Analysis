"""
Module for advanced transformer-based sentiment analysis.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

class TransformerSentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", verbose=False):
        """
        Initialize the transformer-based sentiment analyzer.
        
        Args:
            model_name (str): The name of the pretrained model to use
            verbose (bool): Whether to print verbose output during analysis
        """
        self.model_name = model_name
        self.verbose = verbose
        
        if self.verbose:
            print(f"Initializing TransformerSentimentAnalyzer with {model_name}")
            print("Loading tokenizer and model (this may take a moment)...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Check if CUDA is available and move model to GPU if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.verbose:
            print(f"Model loaded successfully and running on {self.device}")
            if self.device.type == "cuda":
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def analyze_text(self, text, max_length=512):
        """
        Analyze the sentiment of the given text using the transformer model.
        
        Args:
            text (str): Text to analyze
            max_length (int): Maximum sequence length for the model
            
        Returns:
            dict: Dictionary containing sentiment scores and analysis
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        if self.verbose:
            print(f"Analyzing sentiment of text ({len(text)} characters)")
        
        # Tokenize the text
        if self.verbose:
            print("Tokenizing text...")
        
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Forward pass
        if self.verbose:
            print("Running text through the model...")
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(output.logits, dim=-1)
        probs = probs.cpu().numpy()
        
        # Determine sentiment
        if probs[0][1] > 0.6:
            sentiment = "positive"
            confidence = float(probs[0][1])
        elif probs[0][0] > 0.6:
            sentiment = "negative"
            confidence = float(probs[0][0])
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(float(probs[0][1] - 0.5) * 2)
        
        if self.verbose:
            print("Transformer analysis complete")
            print(f"Negative probability: {probs[0][0]:.4f}")
            print(f"Positive probability: {probs[0][1]:.4f}")
            print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
        
        return {
            "algorithm": f"transformer-{self.model_name}",
            "scores": {
                "negative_prob": float(probs[0][0]),
                "positive_prob": float(probs[0][1]),
                "confidence": confidence
            },
            "sentiment": sentiment
        }
    
    def analyze_text_chunks(self, text, chunk_size=256, overlap=50):
        """
        Split text into overlapping chunks and analyze sentiment for each chunk.
        
        Args:
            text (str): Text to analyze
            chunk_size (int): Size of each chunk in words
            overlap (int): Number of words to overlap between chunks
            
        Returns:
            list: List of sentiment analysis results for each chunk
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Split text into words
        words = text.split()
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        if self.verbose:
            print(f"Text split into {len(chunks)} chunks of approximately {chunk_size} words each with {overlap} words overlap")
        
        # Analyze each chunk
        results = []
        for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks", disable=not self.verbose)):
            if self.verbose and not tqdm._instances:
                print(f"\nAnalyzing chunk {i+1}/{len(chunks)}")
            
            result = self.analyze_text(chunk)
            result["chunk_index"] = i
            results.append(result)
        
        # Calculate aggregate sentiment
        pos_count = sum(1 for r in results if r["sentiment"] == "positive")
        neg_count = sum(1 for r in results if r["sentiment"] == "negative")
        neu_count = sum(1 for r in results if r["sentiment"] == "neutral")
        
        # Calculate weighted sentiment (by confidence)
        weighted_sentiment = {
            "positive": sum(r["scores"]["positive_prob"] for r in results) / len(results),
            "negative": sum(r["scores"]["negative_prob"] for r in results) / len(results)
        }
        
        aggregate_result = {
            "algorithm": f"transformer-{self.model_name}-aggregate",
            "chunk_results": results,
            "sentiment_distribution": {
                "positive": pos_count / len(results),
                "negative": neg_count / len(results),
                "neutral": neu_count / len(results)
            },
            "weighted_sentiment": weighted_sentiment,
            "overall_sentiment": "positive" if weighted_sentiment["positive"] > weighted_sentiment["negative"] else "negative"
        }
        
        return aggregate_result