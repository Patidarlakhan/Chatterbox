import torchaudio as ta
import torch
import torchaudio.transforms as T
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
# Change speed by resampling
speed = 0.7  # >1 = faster, <1 = slower

resampler = T.Resample(orig_freq=model.sr, new_freq=int(model.sr*speed))
wav_modified = resampler(wav)
ta.save("test-1.wav", wav_modified, model.sr)

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav = multilingual_model.generate(text, language_id="fr")
# Change speed by resampling
wav_modified = resampler(wav)
ta.save("test-2.wav", wav_modified, multilingual_model.sr)


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# Change speed by resampling
wav_modified = resampler(wav)
ta.save("test-3.wav", wav_modified, model.sr)
