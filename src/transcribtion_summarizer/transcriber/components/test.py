import unittest
from transcribtion_summarizer.transcriber.components.whisper_model import transcribe
import warnings



class TestWhisperTranscriber(unittest.TestCase):
    def test_transcribe(self):
        warnings.filterwarnings("ignore")
        # Use a small local audio file for testing
        result = transcribe(r'D:\imen\LLM\Trans-summarizer\src\transcribtion_summarizer\transcriber\components\sample-000000.mp3')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        print("Transcription:", result)

if __name__ == "__main__":
    unittest.main()