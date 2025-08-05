"""
Module for processing text input from news files.
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextProcessor:
    def __init__(self, verbose=False):
        """
        Initialize the text processor.
        
        Args:
            verbose (bool): Whether to print verbose output during processing
        """
        self.verbose = verbose
        # Download necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            if self.verbose:
                print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        if self.verbose:
            print(f"Initialized TextProcessor with {len(self.stop_words)} stop words")
    
    def read_file(self, file_path):
        """
        Read text from a file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: Content of the file
        """
        if self.verbose:
            print(f"Reading file from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if self.verbose:
                print(f"Successfully read {len(content)} characters")
                
            return content
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, converting to lowercase,
        tokenizing, and removing stop words.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            list: List of processed tokens
        """
        if self.verbose:
            print("Starting text preprocessing...")
            
        # Convert to lowercase
        text = text.lower()
        if self.verbose:
            print("Text converted to lowercase")
            
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        if self.verbose:
            print("Removed special characters and numbers")
            
        # Tokenize
        tokens = word_tokenize(text)
        if self.verbose:
            print(f"Text tokenized into {len(tokens)} tokens")
            
        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        if self.verbose:
            print(f"Removed {len(tokens) - len(filtered_tokens)} stop words")
            print(f"Final token count: {len(filtered_tokens)}")
            
        return filtered_tokens
    
    def process_file(self, file_path):
        """
        Read and preprocess a text file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            list: List of processed tokens
        """
        content = self.read_file(file_path)
        if not content:
            return []
        
        return self.preprocess_text(content)