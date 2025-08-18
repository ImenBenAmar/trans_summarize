import argparse
import logging
import sys
from pathlib import Path
import sys



# Add the src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from transcribtion_summarizer.summarizer.components.data_preprocessing import TextPreprocessor
from transcribtion_summarizer.summarizer.components.bartcnn_model import BartSummarizer
from transcribtion_summarizer.summarizer.components.utils import ensure_dir, setup_logger, load_config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text Summarizer CLI")
    parser.add_argument("--input-file", "-i", type=str, help="Path to input text file")
    parser.add_argument("--output-file", "-o", type=str, help="Path to save output summary")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def read_input_text(input_file: str = None) -> str:
    """Read input text from file or user input."""
    if input_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            sys.exit(1)
    else:
        print("Enter text to summarize (press Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        except KeyboardInterrupt:
            logger.info("Input cancelled by user.")
            sys.exit(0)
        text = " ".join(lines).strip()
        if not text:
            logger.error("No input text provided.")
            sys.exit(1)
        return text

def save_output(summary: str, output_file: str = None, logger=None) -> None:
    """Save or print the summary."""
    if output_file:
        try:
            output_path = Path(output_file)
            ensure_dir(output_path.parent)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            if logger:
                logger.info(f"Summary saved to {output_file}")
        except Exception as e:
            if logger:
                logger.error(f"Error saving summary: {str(e)}")
            else:
                print(f"Error saving summary: {str(e)}")
            sys.exit(1)
    else:
        print("\n=== Summary ===")
        print(summary)
        print("===============\n")

def main():
    """Main function to summarize user-provided text."""
    # Set up logger
    logger = setup_logger(__name__)
    
    # Initialize the summarizer
    summarizer = BartSummarizer()

    # Prompt the user for input text
    print("Enter the text you want to summarize (press Enter when done):")
    input_text = input().strip()

    # Check if the input text is empty
    if not input_text:
        print("Error: Input text cannot be empty.")
        return

    try:
        # Generate the summary
        summary = summarizer.summarize(input_text)
        print("\nGenerated Summary:")
        print(summary)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        print("An error occurred while generating the summary. Please try again.")

if __name__ == "__main__":
    # Add the src directory to sys.path
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    main()