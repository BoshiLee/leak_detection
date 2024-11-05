import librosa
import numpy as np
from tensorflow.keras.models import load_model

def extract_stft_features(audio, sr, n_fft=2048, hop_length=512, desired_time=2.0):

    max_len = int(np.ceil((desired_time * sr) / hop_length))
    # 計算下個 2 的冪次倍數
    target_len = int(np.ceil(max_len / 8) * 8)


    # 計算 STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # 計算幅度譜
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 填充或截斷至 target_len
    if S_db.shape[0] < target_len:
        pad_width = target_len - S_db.shape[0]
        S_db = np.pad(S_db, ((0, pad_width), (0, 0)), mode='constant')
    else:
        S_db = S_db[:target_len, :]

    return S_db

def extract_features(audio, sr, n_mels=128, n_fft=2048, hop_length=512, desired_time=2.0):
    """
    提取 Mel 頻譜圖特徵，並根據目標時間長度進行填充或截斷，
    確保時間幀數是 2 的冪次倍數。

    Args:
        audio (np.ndarray): 音訊資料。
        sr (int): 取樣率。
        n_mels (int): Mel 頻帶數。
        n_fft (int): FFT 大小。
        hop_length (int): hop length。
        desired_time (float): 目標時間長度（秒）。

    Returns:
        np.ndarray: 處理後的 Mel 頻譜圖，形狀為 (target_len, n_mels)。
    """
    max_len = int(np.ceil((desired_time * sr) / hop_length))
    # 計算下個 2 的冪次倍數
    target_len = int(np.ceil(max_len / 8) * 8)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # log_mel_spectrogram = log_mel_spectrogram.T  # 形狀變為 (時間, n_mels)

    # 填充或截斷至 target_len
    if log_mel_spectrogram.shape[0] < target_len:
        pad_width = target_len - log_mel_spectrogram.shape[0]
        log_mel_spectrogram = np.pad(
            log_mel_spectrogram, ((0, pad_width), (0, 0)), mode='constant'
        )
    else:
        log_mel_spectrogram = log_mel_spectrogram[:target_len, :]

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
