import sounddevice as sd
from pathlib import Path
from scipy.io.wavfile import write

fs = 16000
seconds = 6
output_path = Path(__file__).resolve().parent / "input.wav"

print("Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
recording = recording *2
write(str(output_path), fs, recording)
print("Saved.")
