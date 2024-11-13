import random

import librosa
import numpy as np
import os

from scipy.fft import fft
from tqdm import tqdm
from scipy.signal import butter, lfilter


def compute_fft(data, sample_rate, n_fft=2048, desired_time=2.0):
    """
    計算給定數據的 FFT，並返回指定時間長度的頻率和幅值。

    參數:
        data: numpy array, 音頻數據樣本
        sample_rate: int, 音頻的採樣率
        n_fft: int, FFT 計算的樣本數 (默認為 2048)
        desired_time: float, 希望的音頻時長 (秒)

    返回:
        frequencies: numpy array, 頻率軸
        fft_magnitude: numpy array, 截取或填充後的 FFT 幅值矩陣
    """
    # 計算目標長度
    desired_samples = int(desired_time * sample_rate)

    # 如果數據長度不足，則填充到目標長度
    if len(data) < desired_samples:
        data = np.pad(data, (0, desired_samples - len(data)), mode='constant')
    else:
        data = data[:desired_samples]

    # 截取所需樣本長度進行 FFT
    fft_result = fft(data, n=n_fft)  # 計算 FFT，設置 n 參數以控制 FFT 長度
    fft_magnitude = np.abs(fft_result[:n_fft // 2])  # 僅保留前半部分

    # 計算頻率軸
    frequencies = np.fft.fftfreq(n_fft, 1 / sample_rate)[:n_fft // 2]

    return frequencies, fft_magnitude


def compute_fft_features(X, sample_rate, n_fft=2048, desired_time=2.0):
    """
    對一組音頻數據樣本進行 FFT 並提取幅值特徵，適用於 1D CNN。

    參數:
        X: list of numpy arrays, 音頻數據樣本的列表
        sample_rate: int, 音頻的採樣率
        n_fft: int, FFT 計算的樣本數 (默認為 2048)
        desired_time: float, 希望的音頻時長 (秒)

    返回:
        X_features: numpy array, CNN 所需的 1D 特徵陣列 (樣本數, 特徵長度, 1)
    """
    X_features = []
    desired_samples = int(desired_time * sample_rate)

    for data in X:
        # 填充或截取數據到指定長度
        if len(data) < desired_samples:
            data = np.pad(data, (0, desired_samples - len(data)), mode='constant')
        else:
            data = data[:desired_samples]

        # 計算 FFT 並提取幅值
        fft_result = fft(data, n=n_fft)
        fft_magnitude = np.abs(fft_result[:n_fft // 2])  # 取前半部分的幅值

        # 將每個音頻樣本的幅值特徵加入到列表中
        X_features.append(fft_magnitude)

    # 將列表轉為 numpy 陣列，並擴展維度以符合 CNN 輸入形狀 (樣本數, 特徵長度, 1)
    X_features = np.array(X_features)
    X_features = np.expand_dims(X_features, -1)  # 新形狀: (樣本數, 特徵長度, 1)

    return X_features


def extract_stft_features(audio, sr, n_fft=2048, hop_length=512, desired_time=2.0, transpose=True):

    max_len = int(np.ceil((desired_time * sr) / hop_length))
    # 計算下個 2 的冪次倍數
    target_len = int(np.ceil(max_len / 8) * 8)


    # 計算 STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hamming')
    # 計算幅度譜
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    if transpose:
        S_db = S_db.T  # 形狀變為 (時間, 頻率)

    # 填充或截斷至 target_len
    if S_db.shape[0] < target_len:
        pad_width = target_len - S_db.shape[0]
        S_db = np.pad(S_db, ((0, pad_width), (0, 0)), mode='constant')
    else:
        S_db = S_db[:target_len, :]

    return S_db


def extract_features(audio, sr, n_mels=128, n_fft=2048, hop_length=512, enhanced=1.5, desired_time=2.0, transpose=True):
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

    if transpose:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, window='hamming', fmin=10, fmax=1500
        )
    else:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, window='hamming'
        )


    mel_enhanced = mel_spectrogram * enhanced
    log_mel_spectrogram = librosa.power_to_db(mel_enhanced, ref=np.max)
    if transpose:
        log_mel_spectrogram = log_mel_spectrogram.T

    # 填充或截斷至 target_len
    if log_mel_spectrogram.shape[0] < target_len:
        pad_width = target_len - log_mel_spectrogram.shape[0]
        log_mel_spectrogram = np.pad(
            log_mel_spectrogram, ((0, pad_width), (0, 0)), mode='constant'
        )
    else:
        log_mel_spectrogram = log_mel_spectrogram[:target_len, :]

    return log_mel_spectrogram

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def bandpass_filter(data, sr, lowcut, highcut, order=6):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_audio(file_path, traget_sr=48000, desired_time=2.0, n_mels=128, n_fft=2048, hop_length=512):
    # 載入音訊檔案
    audio, sr = librosa.load(file_path, sr=None)

    if sr != traget_sr:
        print(f'Resampling audio... from {sr} to {traget_sr}')
        audio = librosa.resample(audio, orig_sr=sr, target_sr=traget_sr)

    # 預處理音訊
    feature = extract_features(audio, traget_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, desired_time=desired_time)

    # 正規化特徵
    feature = (feature - np.mean(feature)) / np.std(feature)

    # 擴展維度以符合模型輸入 (1, max_len, n_mels, 1)
    feature = np.expand_dims(feature, axis=0)  # 批次維度
    feature = np.expand_dims(feature, axis=-1)  # 通道維度

    return feature

def preprocess_stft_audio(file_path, traget_sr, desired_time=2.0, n_fft=2048, hop_length=512):
    # 載入音訊檔案
    audio, sr = librosa.load(file_path, sr=None)

    if sr != traget_sr:
        print(f'Resampling audio... from {sr} to {traget_sr}')
        audio = librosa.resample(audio, orig_sr=sr, target_sr=traget_sr)

    # 預處理音訊
    feature = extract_stft_features(audio, traget_sr, n_fft=n_fft, hop_length=hop_length, desired_time=desired_time)

    # 正規化特徵
    feature = (feature - np.mean(feature)) / np.std(feature)

    # 擴展維度以符合模型輸入 (1, max_len, n_mels, 1)
    feature = np.expand_dims(feature, axis=0)  # 批次維度
    feature = np.expand_dims(feature, axis=-1)  # 通道維度

    return feature

def load_segmented_files(directory, target_sr=32000):
    wav_files = []
    leak_wav_files = []
    print(f"共有 {len(os.listdir(directory))} 個資料夾")
    for dir in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, dir)):
            continue
        print(f"正在處理 {dir} 資料夾，共有 {len(os.listdir(os.path.join(directory, dir)))} 個檔案")
        for file in tqdm(os.listdir(os.path.join(directory, dir))):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(directory, dir, file)
            y, sr = librosa.load(file_path, sr=None, mono=True)
            if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            filename = f'{dir}_{file}'
            # y_filt = bandpass_filter(y, sr=target_sr, lowcut=10, highcut=1500)
            if 'Leak' in dir:
                leak_wav_files.append((y, target_sr, filename))
            else:
                wav_files.append((y, target_sr, filename))
    print(f"共有 {len(wav_files) + len(leak_wav_files)} 個 wav 檔案，其中 {len(leak_wav_files)} 個有泄漏，{len(wav_files)} 個沒有泄漏")
    return wav_files, leak_wav_files


# 比較數量，均衡資料
def balance_shuffle_data(wav_files, leak_wav_files):
    min_count = min(len(wav_files), len(leak_wav_files))
    wav_files = random.sample(wav_files, min_count)
    leak_wav_files = random.sample(leak_wav_files, min_count)

    print(f"均衡後的樣本數: {min_count}")
    return wav_files, leak_wav_files


