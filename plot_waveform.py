import numpy as np
import matplotlib.pyplot as plt
import wave
import contextlib

# WAVファイルのパスを指定
wav_file = "vp/1120/vq/beat01.wav"

# WAVファイルを開く
with contextlib.closing(wave.open(wav_file, 'r')) as wf:
    n_frames = wf.getnframes()  # 総フレーム数
    framerate = wf.getframerate()  # サンプリングレート
    audio_data = wf.readframes(n_frames)  # 生データ
    audio_array = np.frombuffer(audio_data, dtype=np.int16)  # numpy 配列に変換

# 時間軸を作成
time = np.linspace(0, len(audio_array) / framerate, num=len(audio_array))

# 波形をプロット
plt.figure(figsize=(10, 4))
plt.plot(time, audio_array, color="black", linewidth=0.5)
plt.axis("off")  # 軸を非表示にする
plt.savefig("waveform.png", bbox_inches='tight', pad_inches=0)  # 画像を保存
plt.show()
