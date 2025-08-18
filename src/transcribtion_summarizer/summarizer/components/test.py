import unittest
from transcribtion_summarizer.summarizer.components.data_preprocessing import TextPreprocessor
from transcribtion_summarizer.summarizer.components.bartcnn_model import BartSummarizer
from transcribtion_summarizer.summarizer.components.utils import setup_logger, load_config

class TestSummarizerComponents(unittest.TestCase):
    """Unit tests for summarizer components."""
    
    def setUp(self):
        self.logger = setup_logger(__name__)
        self.preprocessor = TextPreprocessor()
        self.summarizer = BartSummarizer()
        self.sample_text = (
            "This is a long text about artificial intelligence. AI is transforming industries "
            "by automating tasks and improving efficiency. Machine learning models, like BART, "
            "are used for tasks such as text summarization and generation."
        )
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = '  This   is a\ntext   with   !!@#  special chars  '
        cleaned = self.preprocessor.clean_text(dirty_text)
        self.assertEqual(cleaned, "This is a text with special chars")
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "word " * 2000
        truncated = self.preprocessor.truncate_text(long_text)
        tokens = self.preprocessor.tokenizer.encode(truncated)
        self.assertTrue(len(tokens) <= self.preprocessor.max_input_length)
    
    def test_summarization(self):
        """Test summarization output."""
        summary = self.summarizer.summarize(self.sample_text)
        self.assertTrue(len(summary) > 0)
        self.assertTrue(len(summary.split()) <= self.summarizer.max_summary_length)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = load_config("nonexistent.yaml")
        self.assertIsInstance(config, dict)

if __name__ == "__main__":
    unittest.main()