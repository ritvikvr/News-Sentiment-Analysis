# News-Sentiment-Analysis
Financial News Sentiment Analysis System

A modular Python-based system for analyzing sentiment in financial news headlines/articles, designed for exploratory market sentiment research and risk analysis.
Authors


This project provides a sentiment analysis platform for financial news. It analyzes single articles or batches of headlines, identifies entities, and measures sentiment directed towards those entities. The system is adaptable, efficient, and lightweight, offering modular tools for both overall and entity-based sentiment assessment.
Features

Supports Batch and Entity-based Sentiment Analysis
Multiple Sentiment Engines: VADER, TextBlob, and Transformers
Entity Extraction: Using spaCy for organization, person, geopolitical recognition
Visualization Output: Sentiment distribution and entity sentiment bar charts
Command-line Interface: Flexible & verbose logging levels
System Architecture

TextProcessor: Cleans and normalizes text.
Batch Analysis: Processes multiple documents for overall sentiment trends using different algorithms.
Entity-Based Analysis: Extracts entities and assesses sentiment towards each.
Visualizations: Plots for sentiment distribution and entity sentiment ranking.
Modules

Batch Analysis
Analyzes a directory or file of news stories (e.g. sample_news.txt).
Computes sentiment using VADER, TextBlob, and Transformer models (if available).
Results saved as CSV and visualized as count plots (matplotlib/seaborn).
Entity-Based Analysis
Finds named entities in news text (using spaCy).
Calculates average sentiment per entity (e.g., "Donald Trump", "China").
Output includes entity, type, sentiment score, and mention count in JSON.
Visualization: bar charts of entity sentiments.
Data and Preprocessing

Data: News article headlines and brief reports, e.g. "Markets soared today...", "Investors are pessimistic..."
Preprocessing Pipeline:
Lowercasing all text
Tokenization
Stopword removal
Lemmatization
Reconstruct text for analysis
Chunking long texts
Sample Data: Place your test data in data/sample_news.txt.
Usage & Arguments

Running the Script:
The system uses argparse for flexible CLI interaction.
Basic Example:
bash
python sentiment_analysis.py --batch data/sample_news.txt -vv
Arguments:
--batch <dir or file>: Process news items in batch mode.
--entity <file>: Run entity-based sentiment scoring.
-v: Increase verbosity (add more v's for greater detail).
Output & Visualization

Batch Output: CSV file with document-wise sentiment from each model.
Entity Output: JSON with sentiment per detected entity.
Visualizations: PNG/JPG images for sentiment distributions and average entity sentiment.
Performance

Execution Time: 0.03s on small test set.
Accuracy: 87.5% (Manual check over 8 test cases).
Tools Used: matplotlib, seaborn for plots.
Conclusion & Future Work

System Strengths
Efficient and robust for small-scale financial news sentiment analysis.
Modular: Easy to extend or plug-in new models.
Future Enhancements
Interactive dashboard for dynamic visual analytics.
Multilingual news handling.
Advanced model ensemble for better predictive accuracy.
For questions or contributions, please contact the authors.
