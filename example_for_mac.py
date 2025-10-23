# import torch
# import torchaudio as ta
# from chatterbox.tts import ChatterboxTTS
# import torchaudio.transforms as T

# # Detect device (Mac with M1/M2/M3/M4)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# map_location = torch.device(device)

# torch_load_original = torch.load
# def patched_torch_load(*args, **kwargs):
#     if 'map_location' not in kwargs:
#         kwargs['map_location'] = map_location
#     return torch_load_original(*args, **kwargs)

# torch.load = patched_torch_load

# model = ChatterboxTTS.from_pretrained(device=device)
# text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# # If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "/Users/lakhanpatidar/Downloads/Record (online-voice-recorder.com).wav"
# wav = model.generate(
#     text,
#     audio_prompt_path=AUDIO_PROMPT_PATH,
#     exaggeration=2.0,
#     cfg_weight=0.5
#     )
# # Change speed by resampling
# speed = 0.7  # >1 = faster, <1 = slower

# resampler = T.Resample(orig_freq=model.sr, new_freq=int(model.sr*speed))
# wav_modified = resampler(wav)
# ta.save("test-4.wav", wav_modified, model.sr)


# filename: app.py
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os

app = FastAPI(title="Chatterbox TTS API")

# Detect device
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Patch torch.load for M1/M2
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# Load model once
model = ChatterboxTTS.from_pretrained(device=device)
# Temporary output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/tts")
async def tts(
    text: str = Form(...),
    speed: float = Form(1.0),   # optional: <1 = slower, >1 = faster
    exaggeration: float = Form(1.0),
    audio_prompt_path: str = Form(None)
):
    """
    Generate TTS audio from text.
    - text: string input
    - speed: playback speed (default 1.0)
    - audio_prompt_path: optional path to a reference voice
    """
    # Generate audio
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=0.5
    )
    
    
    # Save temporary file
    output_path = os.path.join(OUTPUT_DIR, "output.wav")
    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)  # [samples]

    # STFT
    n_fft = 1024
    hop_length = n_fft // 4
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, return_complex=True)  # [freq, time]

    # Add batch dim for phase_vocoder
    spec = spec.unsqueeze(0)  # [1, freq, time]

    # Phase vocoder
    phase_advance = torch.linspace(0, torch.pi * hop_length, spec.shape[1], device=spec.device)  # [freq]
    spec_stretched = ta.functional.phase_vocoder(spec, rate=speed, phase_advance=phase_advance)

    # Remove batch dim BEFORE ISTFT
    spec_stretched = spec_stretched[0]  # [freq, time]

    # ISTFT
    wav_stretched = torch.istft(spec_stretched, n_fft=n_fft, hop_length=hop_length)  # [samples]

    # Add channel dim for saving
    wav_stretched = wav_stretched.unsqueeze(0)  # [1, samples]

    # Fix tensor dimensions before saving
    if wav_stretched.ndim == 3:
        wav_stretched = wav_stretched.squeeze(0)
    if wav_stretched.ndim == 1:
        wav_stretched = wav_stretched.unsqueeze(0)

    # Save output
    ta.save(output_path, wav_stretched, model.sr)
    
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
