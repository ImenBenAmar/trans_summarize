from transformers import BartTokenizer
from .utils import setup_logger
import re

class TextPreprocessor:
    """Handles text preprocessing for summarization."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_input_length: int = 1024):
        self.logger = setup_logger(__name__)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_input_length = max_input_length
    
    def clean_text(self, text: str) -> str:
        """Clean input text by removing extra whitespace and special characters."""
        if not isinstance(text, str):
            self.logger.error("Input must be a string.")
            raise ValueError("Input must be a string.")
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove standalone punctuation
        text = re.sub(r'\s*[.,!?]+\s*', ' ', text).strip()
        return text
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within the model's max token length."""
        tokens = self.tokenizer.encode(text, truncation=False)
        if len(tokens) > self.max_input_length:
            self.logger.warning(f"Text exceeds max length ({len(tokens)} tokens). Truncating.")
            tokens = tokens[:self.max_input_length-2]  
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        cleaned_text = self.clean_text(text)
        truncated_text = self.truncate_text(cleaned_text)
        return truncated_text