import torchaudio

def prepare_example(example, processor):
    speech_array, sr = torchaudio.load(example["audio"])
    resampler = torchaudio.transforms.Resample(sr, 16000)
    example["input_features"] = processor(resampler(speech_array).squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features[0]
    example["labels"] = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]
    return example