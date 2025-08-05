"""
This script demonstrates advanced usage of the sentiment analysis tool.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.text_processor import TextProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.advanced_analyzer import TransformerSentimentAnalyzer
from src.verbose_logger import VerboseLogger

def batch_analysis(directory, output_dir="results", verbose=False):
    """
    Perform batch analysis on multiple news files.
    
    Args:
        directory (str): Directory containing news files
        output_dir (str): Directory to save results
        verbose (bool): Whether to print verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzers
    text_processor = TextProcessor(verbose=verbose)
    vader_analyzer = SentimentAnalyzer(algorithm="vader", verbose=verbose)
    textblob_analyzer = SentimentAnalyzer(algorithm="textblob", verbose=verbose)
    
    try:
        transformer_analyzer = TransformerSentimentAnalyzer(verbose=verbose)
        use_transformer = True
    except Exception as e:
        print(f"Transformer model could not be loaded: {e}")
        print("Continuing without transformer model...")
        use_transformer = False
    
    # Process each file
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
            # Read and preprocess text
            text = text_processor.read_file(filepath)
            tokens = text_processor.preprocess_text(text)
            processed_text = " ".join(tokens)
            
            # Analyze with different algorithms
            vader_result = vader_analyzer.analyze_text(processed_text)
            textblob_result = textblob_analyzer.analyze_text(processed_text)
            
            if use_transformer:
                transformer_result = transformer_analyzer.analyze_text(processed_text)
                transformer_sentiment = transformer_result["sentiment"]
                transformer_confidence = transformer_result["scores"].get("confidence", 0)
            else:
                transformer_sentiment = "N/A"
                transformer_confidence = 0
            
            # Store results
            results.append({
                "filename": filename,
                "vader_sentiment": vader_result["sentiment"],
                "vader_compound": vader_result["scores"]["compound"],
                "textblob_sentiment": textblob_result["sentiment"],
                "textblob_polarity": textblob_result["scores"]["polarity"],
                "transformer_sentiment": transformer_sentiment,
                "transformer_confidence": transformer_confidence,
                "text_length": len(text),
                "word_count": len(tokens)
            })
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
    
    # Create visualizations
    if results:
        # Sentiment distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(data=results_df, x="vader_sentiment", palette=["red", "gray", "green"])
        plt.title("Distribution of Sentiment (VADER)")
        plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"), dpi=300)
        
        # Sentiment comparison
        if use_transformer:
            plt.figure(figsize=(12, 8))
            
            # Create a cross-tabulation of sentiments
            sentiment_matrix = pd.crosstab(
                results_df["vader_sentiment"], 
                results_df["transformer_sentiment"]
            )
            
            # Create heatmap
            sns.heatmap(sentiment_matrix, annot=True, cmap="YlGnBu", fmt="d")
            plt.title("Sentiment Comparison: VADER vs Transformer")
            plt.xlabel("Transformer Sentiment")
            plt.ylabel("VADER Sentiment")
            plt.savefig(os.path.join(output_dir, "sentiment_comparison.png"), dpi=300)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return results_df

def entity_based_analysis(file_path, output_dir="results", verbose=False):
    """
    Perform entity-based sentiment analysis.
    
    Args:
        file_path (str): Path to the news file
        output_dir (str): Directory to save results
        verbose (bool): Whether to print verbose output
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print("Spacy is not installed. Please install it with:")
        print("pip install spacy")
        print("python -m spacy download en_core_web_sm")
        return
    except OSError:
        print("Spacy model not found. Please download it with:")
        print("python -m spacy download en_core_web_sm")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzers
    text_processor = TextProcessor(verbose=verbose)
    vader_analyzer = SentimentAnalyzer(algorithm="vader", verbose=verbose)
    
    # Read file
    text = text_processor.read_file(file_path)
    
    # Process with spaCy to extract entities
    doc = nlp(text)
    
    # Extract entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Person, Organization, Geo-Political Entity
            if ent.text not in entities:
                entities[ent.text] = {
                    "type": ent.label_,
                    "mentions": []
                }
            
            # Get sentence containing the entity
            sentence = ent.sent.text
            entities[ent.text]["mentions"].append(sentence)
    
    # Analyze sentiment for each entity
    entity_sentiments = []
    for entity, data in entities.items():
        if len(data["mentions"]) < 2:  # Skip entities with too few mentions
            continue
        
        # Analyze sentiment of each mention
        mention_sentiments = []
        for mention in data["mentions"]:
            sentiment = vader_analyzer.analyze_text(mention)
            mention_sentiments.append({
                "text": mention,
                "compound": sentiment["scores"]["compound"],
                "sentiment": sentiment["sentiment"]
            })
        
        # Calculate average sentiment
        avg_compound = sum(m["compound"] for m in mention_sentiments) / len(mention_sentiments)
        
        entity_sentiments.append({
            "entity": entity,
            "type": data["type"],
            "mention_count": len(data["mentions"]),
            "avg_compound": avg_compound,
            "sentiment": "positive" if avg_compound >= 0.05 else ("negative" if avg_compound <= -0.05 else "neutral"),
            "mentions": mention_sentiments
        })
    
    # Sort by mention count
    entity_sentiments.sort(key=lambda x: x["mention_count"], reverse=True)
    
    # Save results
    import json
    with open(os.path.join(output_dir, "entity_sentiments.json"), "w") as f:
        json.dump(entity_sentiments, f, indent=2)
    
    # Create visualization
    if entity_sentiments:
        top_entities = entity_sentiments[:10]  # Top 10 entities by mention count
        
        # Create DataFrame for visualization
        viz_data = pd.DataFrame([
            {"Entity": e["entity"], "Sentiment": e["avg_compound"], "Mentions": e["mention_count"], "Type": e["type"]}
            for e in top_entities
        ])
        
        plt.figure(figsize=(14, 8))
        bars = sns.barplot(
            x="Entity", 
            y="Sentiment", 
            data=viz_data, 
            palette=sns.color_palette("RdYlGn", n_colors=len(viz_data)),
            hue="Sentiment",  # Color based on sentiment value
            dodge=False
        )
        
        # Add mention count as text on bars
        for i, row in enumerate(viz_data.itertuples()):
            bars.text(
                i, 
                0, 
                f"{row.Mentions} mentions",
                ha='center',
                va='bottom' if row.Sentiment >= 0 else 'top',
                color='black'
            )
        
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title("Entity Sentiment Analysis")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "entity_sentiment.png"), dpi=300)
        
        # Create entity type breakdown
        plt.figure(figsize=(10, 6))
        sns.countplot(x="Type", data=viz_data)
        plt.title("Entity Types")
        plt.savefig(os.path.join(output_dir, "entity_types.png"), dpi=300)
    
    print(f"Entity analysis complete. Results saved to {output_dir}")
    return entity_sentiments

def main():
    parser = argparse.ArgumentParser(description="Advanced usage of sentiment analysis tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Batch analysis command
    batch_parser = subparsers.add_parser("batch", help="Perform batch analysis on multiple files")
    batch_parser.add_argument("directory", help="Directory containing news files")
    batch_parser.add_argument("--output", default="batch_results", help="Output directory")
    batch_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Entity analysis command
    entity_parser = subparsers.add_parser("entity", help="Perform entity-based sentiment analysis")
    entity_parser.add_argument("file", help="Path to news file")
    entity_parser.add_argument("--output", default="entity_results", help="Output directory")
    entity_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.command == "batch":
        batch_analysis(args.directory, args.output, args.verbose)
    elif args.command == "entity":
        entity_based_analysis(args.file, args.output, args.verbose)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()