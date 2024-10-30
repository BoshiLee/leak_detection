import librosa
import numpy as np
from tensorflow.keras.models import load_model


def extract_features(audio, sr, n_mels=128, n_fft=2048, hop_length=512, desired_time=2.0):
    # 計算 max_len 以符合 desired_time
    max_len = int(np.ceil((desired_time * sr) / hop_length))

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 轉置以符合 (時間, 頻率) 的形狀
    log_mel_spectrogram = log_mel_spectrogram.T  # 形狀變為 (時間, n_mels)

    # 填充或截斷至 max_len
    if log_mel_spectrogram.shape[0] < max_len:
        pad_width = max_len - log_mel_spectrogram.shape[0]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, pad_width), (0, 0)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:max_len, :]

    return log_mel_spectrogram


def preprocess_audio(file_path, desired_time=2.0, n_mels=128, n_fft=2048, hop_length=512):
    # 載入音訊檔案
    audio, sr = librosa.load(file_path, sr=None)

    # 預處理音訊
    feature = extract_features(audio, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, desired_time=desired_time)

    # 正規化特徵
    feature = (feature - np.mean(feature)) / np.std(feature)

    # 擴展維度以符合模型輸入 (1, max_len, n_mels, 1)
    feature = np.expand_dims(feature, axis=0)  # 批次維度
    feature = np.expand_dims(feature, axis=-1)  # 通道維度

    return feature
