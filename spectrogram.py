import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

samplerate, data = wavfile.read("vocal_percussion_clipped3.wav")

if len(data.shape) == 2:
    data = data.mean(axis=1)  # ステレオ→モノラル

plt.figure()
plt.specgram(data, Fs=samplerate)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram")
plt.colorbar(label="Intensity [dB]")
plt.show()
