from transformers import BartForConditionalGeneration, BartTokenizer
from .data_preprocessing import TextPreprocessor
from .utils import setup_logger, load_config

class BartSummarizer:
    """Handles text summarization using BART-CNN."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = setup_logger(__name__)
        self.config = load_config(config_path)
        self.model_name = self.config.get("model_name", "facebook/bart-large-cnn")
        
        self.logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.preprocessor = TextPreprocessor(self.model_name)
        
        # Model parameters from config
        self.max_summary_length = self.config.get("max_summary_length", 150)
        self.min_summary_length = self.config.get("min_summary_length", 50)
        self.num_beams = self.config.get("num_beams", 4)
    
    def summarize(self, text: str) -> str:
        """Generate a summary for the input text."""
        try:
            # Preprocess the input text
            processed_text = self.preprocessor.preprocess(text)
            if not processed_text:
                self.logger.error("Empty text after preprocessing.")
                return ""
            
            # Tokenize and generate summary
            inputs = self.tokenizer([processed_text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_summary_length,
                min_length=self.min_summary_length,
                num_beams=self.num_beams,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode the summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            self.logger.info("Summary generated successfully.")
            return summary
        
        except Exception as e:
            self.logger.error(f"Error during summarization: {str(e)}")
            raise