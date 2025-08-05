"""
Module for handling verbose logging and visualization of the sentiment analysis process.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import os
import seaborn as sns

class VerboseLogger:
    def __init__(self, level=2):
        """
        Initialize the verbose logger.
        
        Args:
            level (int): The verbosity level (1-3). Higher is more verbose.
        """
        self.level = level
        self.start_time = time.time()
        self.steps = []
        
        # Set up styling for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def log(self, message, level=1, timing=True):
        """
        Log a message if the logger's level is high enough.
        
        Args:
            message (str): The message to log
            level (int): The level of this message
            timing (bool): Whether to include timing information
        """
        if level <= self.level:
            timestamp = time.time() - self.start_time
            time_str = f"[{timestamp:.2f}s] " if timing else ""
            print(f"{time_str}{message}")
            
            # Store step for later analysis
            self.steps.append({
                "time": timestamp,
                "message": message,
                "level": level
            })
    
    def log_algorithm_step(self, step_name, details=None):
        """
        Log an algorithm step with optional details.
        
        Args:
            step_name (str): Name of the algorithm step
            details (dict): Additional details about the step
        """
        self.log(f"ALGORITHM STEP: {step_name}", level=2)
        if details and self.level >= 3:
            for key, value in details.items():
                self.log(f"  - {key}: {value}", level=3, timing=False)
    
    def generate_report(self, output_file=None, results=None):
        """
        Generate a report of the sentiment analysis process.
        
        Args:
            output_file (str): Path to save the report to
            results (dict): Dictionary containing sentiment analysis results
            
        Returns:
            str: HTML report
        """
        steps_df = pd.DataFrame(self.steps)
        total_time = time.time() - self.start_time
        
        # Create visualizations if results are provided
        visualization_html = ""
        if results:
            # Create output directory for visualizations
            vis_dir = "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate visualizations
            vis_files = self._generate_visualizations(results, vis_dir)
            
            # Add visualizations to report
            visualization_html = "<h2>Sentiment Analysis Visualizations</h2>"
            for name, file_path in vis_files.items():
                rel_path = os.path.relpath(file_path, os.path.dirname(output_file))
                visualization_html += f"""
                <div class="visualization">
                    <h3>{name}</h3>
                    <img src="{rel_path}" alt="{name}" style="max-width: 100%;">
                </div>
                """
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #212529; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #343a40; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0; }}
                .step {{ margin: 5px 0; padding: 10px; border-left: 4px solid #ccc; background-color: white; border-radius: 0 5px 5px 0; }}
                .level-1 {{ border-color: #007bff; }}
                .level-2 {{ border-color: #28a745; }}
                .level-3 {{ border-color: #ffc107; }}
                .step-time {{ font-weight: bold; color: #6c757d; }}
                .visualization {{ margin: 30px 0; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .summary {{ display: flex; margin: 20px 0; }}
                .summary-box {{ flex: 1; padding: 15px; background-color: white; border-radius: 5px; margin-right: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
                .summary-box:last-child {{ margin-right: 0; }}
                .summary-box h3 {{ margin-top: 0; color: #343a40; }}
                .summary-box p {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .summary-box.time {{ border-top: 4px solid #007bff; }}
                .summary-box.steps {{ border-top: 4px solid #28a745; }}
                .summary-box.algorithms {{ border-top: 4px solid #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Sentiment Analysis Execution Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <div class="summary-box time">
                        <h3>Execution Time</h3>
                        <p>{total_time:.2f} seconds</p>
                    </div>
                    <div class="summary-box steps">
                        <h3>Steps Executed</h3>
                        <p>{len(self.steps)}</p>
                    </div>
                    <div class="summary-box algorithms">
                        <h3>Algorithms Used</h3>
                        <p>{len(results) if results else 'N/A'}</p>
                    </div>
                </div>
                
                {visualization_html}
                
                <h2>Algorithm Execution Steps:</h2>
        """
        
        for step in self.steps:
            level_class = f"level-{step['level']}"
            html += f"""
            <div class="step {level_class}">
                <span class="step-time">[{step['time']:.2f}s]</span> {step['message']}
            </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
        
        return html
    
    def _generate_visualizations(self, results, output_dir):
        """
        Generate visualizations for sentiment analysis results.
        
        Args:
            results (dict): Dictionary containing sentiment analysis results
            output_dir (str): Directory to save visualizations to
            
        Returns:
            dict: Dictionary of visualization names and file paths
        """
        vis_files = {}
        
        # Sentiment comparison visualization
        if len(results) > 1:
            # Extract sentiment scores for each algorithm
            scores = {}
            for algo, result in results.items():
                # Handle different score formats across algorithms
                if algo == 'vader':
                    scores[algo] = {
                        'positive': result['scores']['pos'],
                        'negative': result['scores']['neg'],
                        'neutral': result['scores']['neu']
                    }
                elif algo == 'textblob':
                    polarity = result['scores']['polarity']
                    scores[algo] = {
                        'positive': max(0, polarity) if polarity > 0 else 0,
                        'negative': abs(min(0, polarity)) if polarity < 0 else 0,
                        'neutral': 1 - abs(polarity)
                    }
                elif algo == 'transformer':
                    # Check if aggregate or single result
                    if 'weighted_sentiment' in result:
                        scores[algo] = {
                            'positive': result['weighted_sentiment']['positive'],
                            'negative': result['weighted_sentiment']['negative'],
                            'neutral': 1 - (result['weighted_sentiment']['positive'] + 
                                          result['weighted_sentiment']['negative'])
                        }
                    elif 'scores' in result:
                        scores[algo] = {
                            'positive': result['scores'].get('positive_prob', 0),
                            'negative': result['scores'].get('negative_prob', 0),
                            'neutral': 1 - (result['scores'].get('positive_prob', 0) + 
                                          result['scores'].get('negative_prob', 0))
                        }
            
            # Create comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            algorithms = list(scores.keys())
            x = np.arange(len(algorithms))
            width = 0.25
            
            # Plot bars for each sentiment
            bar1 = ax.bar(x - width, [scores[algo]['positive'] for algo in algorithms], 
                         width, label='Positive', color='#28a745')
            bar2 = ax.bar(x, [scores[algo]['neutral'] for algo in algorithms], 
                         width, label='Neutral', color='#6c757d')
            bar3 = ax.bar(x + width, [scores[algo]['negative'] for algo in algorithms], 
                         width, label='Negative', color='#dc3545')
            
            # Add labels and title
            ax.set_xlabel('Algorithms')
            ax.set_ylabel('Score')
            ax.set_title('Sentiment Analysis Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms)
            ax.legend()
            
            # Add value labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
                               
            autolabel(bar1)
            autolabel(bar2)
            autolabel(bar3)
            
            plt.tight_layout()
            
            # Save figure
            comparison_file = os.path.join(output_dir, 'sentiment_comparison.png')
            plt.savefig(comparison_file, dpi=300)
            plt.close()
            
            vis_files['Sentiment Comparison'] = comparison_file
            
            # Create sentiment score heatmap
            # Prepare data for heatmap
            heatmap_data = []
            for algo in algorithms:
                for sentiment, score in scores[algo].items():
                    heatmap_data.append({
                        'Algorithm': algo,
                        'Sentiment': sentiment,
                        'Score': score
                    })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_pivot = heatmap_df.pivot(index='Algorithm', columns='Sentiment', values='Score')
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_pivot, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=.5)
            plt.title('Sentiment Score Heatmap')
            plt.tight_layout()
            
            # Save figure
            heatmap_file = os.path.join(output_dir, 'sentiment_heatmap.png')
            plt.savefig(heatmap_file, dpi=300)
            plt.close()
            
            vis_files['Sentiment Score Heatmap'] = heatmap_file
        
        # Check if we have chunk analysis from transformer
        transformer_result = results.get('transformer', {})
        if 'chunk_results' in transformer_result:
            # Plot sentiment scores for each chunk
            chunk_results = transformer_result['chunk_results']
            chunk_indices = [r['chunk_index'] for r in chunk_results]
            pos_scores = [r['scores'].get('positive_prob', 0) for r in chunk_results]
            neg_scores = [r['scores'].get('negative_prob', 0) for r in chunk_results]
            
            plt.figure(figsize=(14, 7))
            plt.plot(chunk_indices, pos_scores, 'g-', label='Positive Score', linewidth=2)
            plt.plot(chunk_indices, neg_scores, 'r-', label='Negative Score', linewidth=2)
            plt.fill_between(chunk_indices, pos_scores, alpha=0.3, color='green')
            plt.fill_between(chunk_indices, neg_scores, alpha=0.3, color='red')
            plt.xlabel('Text Chunk')
            plt.ylabel('Sentiment Score')
            plt.title('Sentiment Flow Through Text')
            plt.legend()
            plt.grid(True)
            
            # Add threshold line at 0.5
            plt.axhline(y=0.5, color='gray', linestyle='--')
            
            # Save figure
            flow_file = os.path.join(output_dir, 'sentiment_flow.png')
            plt.savefig(flow_file, dpi=300)
            plt.close()
            
            vis_files['Sentiment Flow'] = flow_file
        
        return vis_files
    
    def plot_execution_timeline(self, output_file=None):
        """
        Plot the execution timeline of the algorithm.
        
        Args:
            output_file (str): Path to save the plot to
        """
        steps_df = pd.DataFrame(self.steps)
        
        # Filter out level 3 messages for clarity if there are too many steps
        if len(steps_df) > 20:
            plot_df = steps_df[steps_df['level'] <= 2].copy()
        else:
            plot_df = steps_df.copy()
        
        # Create a readable label for each step
        plot_df['label'] = plot_df['message'].str.slice(0, 50)
        plot_df['label'] = plot_df['label'].apply(lambda x: x + '...' if len(x) >= 50 else x)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create colormap based on message level
        colors = plot_df['level'].map({1: '#007bff', 2: '#28a745', 3: '#ffc107'})
        
        # Plot horizontal bars
        bars = ax.barh(y=range(len(plot_df)), width=plot_df['time'], height=0.6, color=colors)
        
        # Add time labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.1:  # Only add label if bar is wide enough
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}s', 
                       va='center', ha='left', fontsize=9)
        
        # Set y-axis ticks and labels
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['label'])
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Sentiment Analysis Algorithm Execution Timeline')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#007bff', label='Level 1 (Basic)'),
            Patch(facecolor='#28a745', label='Level 2 (Algorithm Step)')
        ]
        if 3 in steps_df['level'].values:
            legend_elements.append(Patch(facecolor='#ffc107', label='Level 3 (Detailed)'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
        else:
            plt.show()