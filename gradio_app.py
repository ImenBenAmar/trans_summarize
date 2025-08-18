import gradio as gr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from transcribtion_summarizer.summarizer.components.bartcnn_model import BartSummarizer
from transcribtion_summarizer.transcriber.components.whisper_model import load_finetuned_whisper
import torchaudio
import torch


# Load Whisper model once
processor, model = load_finetuned_whisper()
model.generation_config.forced_decoder_ids = None
def transcribe_audio(audio):
    if audio is None:
        return "No audio provided."
    sr, data = audio
    data = torch.tensor(data).float()
    if sr != 16000:
        data = torchaudio.transforms.Resample(sr, 16000)(data)
    input_features = processor(data.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    predicted_ids = model.generate(input_features, task="transcribe", language="en")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def summarize_text(text, model_name, min_len, max_len, num_beams):
    if not text or not text.strip():
        return "No valid text provided."
    summarizer = BartSummarizer()
    summarizer.model_name = model_name
    summarizer.max_summary_length = max_len
    summarizer.min_summary_length = min_len
    summarizer.num_beams = num_beams
    summarizer.tokenizer = summarizer.preprocessor.tokenizer.from_pretrained(model_name)
    summarizer.model = summarizer.model.from_pretrained(model_name)
    return summarizer.summarize(text)


summarize_btn = gr.Button("Summarize", interactive=True)
transcribe_btn = gr.Button("Transcribe", interactive=True)


# Update button states: always clickable
def update_buttons(text, audio):
    return gr.update(interactive=True), gr.update(interactive=True)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "<h2 style='text-align: center;'>🎤📝 Voice & Text Summarizer</h2>"
        "<p style='text-align: center;'>Paste text or record your voice, then summarize or transcribe easily!</p>"
    )
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Enter text to summarize", lines=6, placeholder="Paste or type your text here...")
            audio_input = gr.Audio(type="numpy", label="Record audio")

            with gr.Row():
                summarize_btn = gr.Button("Summarize")
                transcribe_btn = gr.Button("Transcribe")
            model_name = gr.Dropdown(
                choices=["facebook/bart-large-cnn", "facebook/bart-base"],
                value="facebook/bart-large-cnn",
                label="Summarizer Model"
            )
            min_len = gr.Slider(10, 200, value=50, step=1, label="Min summary length")
            max_len = gr.Slider(20, 300, value=150, step=1, label="Max summary length")
            num_beams = gr.Slider(1, 8, value=4, step=1, label="Number of beams")
        with gr.Column(scale=1):
            output_box = gr.Textbox(label="Transcription / Summary", lines=12, interactive=False)

    # Update button states based on input
    text_input.change(
        update_buttons,
        inputs=[text_input, audio_input],
        outputs=[summarize_btn, transcribe_btn]
    )
    audio_input.change(
        update_buttons,
        inputs=[text_input, audio_input],
        outputs=[summarize_btn, transcribe_btn]
    )

    # Summarize button
    summarize_btn.click(
        summarize_text,
        inputs=[text_input, model_name, min_len, max_len, num_beams],
        outputs=output_box
    )

    transcribe_btn.click(
        transcribe_audio,
        inputs=audio_input,
        outputs=output_box
    )


if __name__ == "__main__":
   demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
