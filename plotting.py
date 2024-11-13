import os

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.fft import fft

from data_preprocess import extract_features, extract_stft_features, compute_fft

from dotenv import load_dotenv

load_dotenv()

sample_rate = int(os.getenv("SAMPLE_RATE"))
n_fft = int(os.getenv("N_FFT"))
n_mels = int(os.getenv("N_MELS"))
mels_n_fft = int(os.getenv("MELS_N_FFT"))
mels_hop_length = int(os.getenv("MELS_HOP_LENGTH"))
stft_n_fft = int(os.getenv("STFT_N_FFT"))
stft_hop_length = int(os.getenv("STFT_HOP_LENGTH"))
desired_time = float(os.getenv("DESIRED_TIME"))

print('n_mels:', n_mels)
print('mels_n_fft:', mels_n_fft)
print('mels_hop_length:', mels_hop_length)

def plot_mel_stft_fft(wav, file_name, class_type='no-leak'):

    mel_spectrogram = extract_features(wav, sr=sample_rate, n_mels=n_mels, n_fft=mels_n_fft, hop_length=mels_hop_length, desired_time=desired_time, enhanced=1, transpose=False)
    S_db = extract_stft_features(wav, sr=sample_rate, n_fft=stft_n_fft, hop_length=stft_hop_length, desired_time=desired_time, transpose=False)
    frequencies, fft_magnitude = compute_fft(wav, sample_rate=sample_rate, n_fft=n_fft, desired_time=desired_time)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 繪製 Mel 頻譜圖
    mel_img = librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=mels_hop_length, x_axis='time', y_axis='mel', cmap='jet', ax=axs[0])
    axs[0].set_title(f'Mel Spectrogram {file_name}')
    axs[0].set_ylim(0, 1600)
    axs[0].set_yticks(np.arange(0, 1600, 150))
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(mel_img, ax=axs[0], format='%+2.0f dB')  # 添加 colorbar

    # 繪製 STFT 頻譜圖
    stft_img = librosa.display.specshow(S_db, sr=sample_rate, hop_length=128, x_axis='time', y_axis='log', cmap='jet', ax=axs[1])
    axs[1].set_title(f'STFT Magnitude {file_name}')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(stft_img, ax=axs[1], format='%+2.0f dB')  # 添加 colorbar

    # 繪製 FFT 頻譜圖
    axs[2].plot(frequencies, fft_magnitude)
    axs[2].set_xlim(0, 3600)
    axs[2].set_title(f"FFT of Audio Signal {file_name}")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Magnitude")
    axs[2].grid(True)

    # 確保目錄存在
    os.makedirs(f"images/{class_type}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{class_type}/{file_name}_mel_stft_fft.png")
    plt.show()
    plt.close()


def plot_training_history(history, model_name):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='val accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')

    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='val loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')

    os.makedirs('images/model_loss', exist_ok=True)
    plt.savefig(f'images/model_loss/{model_name}.png')

    plt.show()