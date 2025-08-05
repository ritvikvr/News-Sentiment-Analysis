"""
Main script for running sentiment analysis on news text files.
"""
import os
import sys
import argparse
import json
import torch
from src.text_processor import TextProcessor
from src.sentiment_analyzer import SentimentAnalyze
from src.verbose_logger import VerboseLogger
from src.advanced_analyzer import TransformerSentimentAnalyzer

def main():
    """
    Main function to run the sentiment analysis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News Sentiment Analysis Tool')
    parser.add_argument('--input', '-i', required=True, help='Path to the input news text file')
    parser.add_argument('--algorithm', '-a', default='all', 
                       choices=['vader', 'textblob', 'transformer', 'all'],
                       help='Sentiment analysis algorithm to use')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--output', '-o', help='Output file path for the report (HTML)')
    parser.add_argument('--json', '-j', help='Output file path for JSON results')
    parser.add_argument('--plot', '-p', help='Output file path for the execution timeline plot')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare results from all algorithms')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='Size of text chunks for analysis (in words)')
    parser.add_argument('--model', '-m', default='distilbert-base-uncased-finetuned-sst-2-english',
                       help='Transformer model to use for analysis')
    parser.add_argument('--exclude-preprocessing', action='store_true',
                       help='Skip preprocessing steps (use raw text)')
    
    args = parser.parse_args()
    
    # Setup verbosity level
    verbose_level = min(args.verbose + 1, 3)  # Map to 1-3 range
    logger = VerboseLogger(level=verbose_level)
    
    logger.log(f"Starting sentiment analysis with algorithm(s): {args.algorithm}")
    logger.log(f"Input file: {args.input}")
    logger.log(f"Verbosity level: {verbose_level}")
    
    # Force comparison if 'all' algorithms are selected
    if args.algorithm == 'all':
        args.compare = True
    
    # Check if the input file exists
    if not os.path.exists(args.input):
        logger.log(f"ERROR: Input file not found: {args.input}", level=1)
        return 1
    
    try:
        # Step 1: Initialize the text processor
        logger.log_algorithm_step("Initializing text processor")
        text_processor = TextProcessor(verbose=(verbose_level >= 3))
        
        # Step 2: Process the input file
        logger.log_algorithm_step("Processing input file")
        text = text_processor.read_file(args.input)
        
        if not text:
            logger.log("ERROR: No text was read from the input file", level=1)
            return 1
        
        # Skip preprocessing if requested
        if args.exclude_preprocessing:
            logger.log("Skipping preprocessing steps (using raw text)", level=2)
            processed_text = text
        else:
            tokens = text_processor.preprocess_text(text)
            processed_text = ' '.join(tokens)
        
        # Step 3: Perform sentiment analysis with selected algorithm(s)
        results = {}
        
        if args.algorithm in ['vader', 'all']:
            logger.log_algorithm_step("Initializing VADER sentiment analyzer")
            vader_analyzer = SentimentAnalyzer(algorithm='vader', verbose=(verbose_level >= 2))
            
            logger.log_algorithm_step("Performing VADER sentiment analysis")
            vader_result = vader_analyzer.analyze_text(processed_text)
            results['vader'] = vader_result
            
            logger.log(f"VADER sentiment: {vader_result['sentiment']}")
        
        if args.algorithm in ['textblob', 'all']:
            logger.log_algorithm_step("Initializing TextBlob sentiment analyzer")
            textblob_analyzer = SentimentAnalyzer(algorithm='textblob', verbose=(verbose_level >= 2))
            
            logger.log_algorithm_step("Performing TextBlob sentiment analysis")
            textblob_result = textblob_analyzer.analyze_text(processed_text)
            results['textblob'] = textblob_result
            
            logger.log(f"TextBlob sentiment: {textblob_result['sentiment']}")
        
        if args.algorithm in ['transformer', 'all']:
            # Check if GPU is available
            if torch.cuda.is_available():
                logger.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.log("GPU not available, using CPU (this might be slow)")
            
            logger.log_algorithm_step(f"Initializing Transformer sentiment analyzer with model: {args.model}")
            transformer_analyzer = TransformerSentimentAnalyzer(
                model_name=args.model, 
                verbose=(verbose_level >= 2)
            )
            
            logger.log_algorithm_step("Performing Transformer sentiment analysis")
            
            # For longer texts, use chunk analysis
            if len(processed_text.split()) > args.chunk_size:
                logger.log(f"Text is long, analyzing in chunks of {args.chunk_size} words")
                transformer_result = transformer_analyzer.analyze_text_chunks(
                    processed_text, 
                    chunk_size=args.chunk_size
                )
            else:
                transformer_result = transformer_analyzer.analyze_text(processed_text)
            
            results['transformer'] = transformer_result
            
            if 'overall_sentiment' in transformer_result:
                sentiment = transformer_result['overall_sentiment']
            else:
                sentiment = transformer_result['sentiment']
                
            logger.log(f"Transformer sentiment: {sentiment}")
        
        # Step 4: Compare results if multiple algorithms were used
        if args.compare and len(results) > 1:
            logger.log_algorithm_step("Comparing sentiment analysis results")
            
            # Extract sentiments
            sentiments = {algo: res.get('overall_sentiment', res.get('sentiment')) 
                         for algo, res in results.items()}
            
            # Check agreement
            unique_sentiments = set(sentiments.values())
            if len(unique_sentiments) == 1:
                logger.log(f"All algorithms agree: {list(unique_sentiments)[0]}")
            else:
                logger.log("Algorithms disagree on sentiment:")
                for algo, sentiment in sentiments.items():
                    logger.log(f"  - {algo}: {sentiment}")
                
                # Majority voting
                sentiment_counts = {}
                for sentiment in sentiments.values():
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                majority_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
                logger.log(f"Majority sentiment: {majority_sentiment}")
        
        # Step 5: Save results to JSON if requested
        if args.json:
            logger.log_algorithm_step("Saving results to JSON")
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.log(f"Results saved to {args.json}")
        
        # Step 6: Generate report if requested
        if args.output:
            logger.log_algorithm_step("Generating HTML report")
            html_report = logger.generate_report(args.output)
            logger.log(f"Report saved to {args.output}")
        
        # Step 7: Generate execution timeline plot if requested
        if args.plot:
            logger.log_algorithm_step("Generating execution timeline plot")
            logger.plot_execution_timeline(args.plot)
            logger.log(f"Execution timeline plot saved to {args.plot}")
        
        logger.log("Sentiment analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.log(f"ERROR: An error occurred: {str(e)}", level=1)
        if verbose_level >= 3:
            import traceback
            logger.log(traceback.format_exc(), level=3)
        return 1

if __name__ == "__main__":
    sys.exit(main())