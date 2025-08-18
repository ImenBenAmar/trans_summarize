# Add this to whisper_model.py

import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_finetuned_whisper(model_name="imen0/whisper-cv-finetuned"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return processor, model

# Load model and processor once (or do it inside the function if you prefer)
processor, model = load_finetuned_whisper()

def transcribe(audio_path):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
    input_features = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    predicted_ids = model.generate(
        input_features,
        task="transcribe",
        language="en"
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription