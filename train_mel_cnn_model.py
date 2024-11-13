import argparse
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cnn_model import create_model, train_model, evaluate_model, plot_loss
from data_preprocess import extract_features, load_segmented_files, balance_shuffle_data

def preprocess_data(X, desired_time=2.0, n_mels=128, n_fft=2048, hop_length=512):
    features = []
    for audio, sr in tqdm(X):
        feature = extract_features(audio, sr, desired_time=desired_time, n_mels=n_mels,
                                   n_fft=n_fft, hop_length=hop_length)
        features.append(feature)
    return np.array(features)


def create_dataset(directory):
    wav_files, leak_wav_files = load_segmented_files(directory)
    wav_files, leak_wav_files = balance_shuffle_data(wav_files, leak_wav_files)

    # 將資料轉為 NumPy 格式
    X = []
    y = []

    for audio, sr, filename in wav_files:
        X.append((audio, sr))
        y.append(0)  # wav_files 標記為 0

    for audio, sr, filename in leak_wav_files:
        X.append((audio, sr))
        y.append(1)  # leak_wav_files 標記為 1

    np.bincount(y)
    return np.array(X, dtype=object), np.array(y)


def create_training_data(X, y, desired_time=2.0, n_mels=128, n_fft=2048, hop_length=512):
    print(f"總樣本數: {len(X)}")
    print(f"標籤分佈: {np.bincount(y)}")

    X_features = preprocess_data(X, desired_time=desired_time, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    print(f"特徵形狀: {X_features.shape}")  # 預期形狀: (樣本數, max_len, n_mels)

    # 正規化特徵
    X_features = (X_features - np.mean(X_features)) / np.std(X_features)

    # 擴展維度以符合 CNN 輸入 (樣本數, 高, 寬, 通道)
    X_features = np.expand_dims(X_features, -1)  # 新形狀: (樣本數, max_len, n_mels, 1)

    # 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2)

    print(f"訓練集形狀: {X_train.shape}")
    print(f"測試集形狀: {X_test.shape}")

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train CNN model for leaking audio classification')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)

    print(f"Training model with dataset in {dataset_path}")

    load_dotenv()
    desired_time = float(os.getenv('DESIRED_TIME'))
    n_mels = int(os.getenv('N_MELS'))
    n_fft = int(os.getenv('N_FFT'))
    hop_length = int(os.getenv('HOP_LENGTH'))

    print(f"Desired time: {desired_time}")
    print(f"Number of mel bands: {n_mels}")
    print(f"Number of FFT points: {n_fft}")
    print(f"Hop length: {hop_length}")

    print("Creating dataset...")
    X, y = create_dataset(dataset_path)
    X_train, X_test, y_train, y_test = create_training_data(X, y,
                                                            desired_time=desired_time,
                                                            n_mels=n_mels,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length)

    print("Creating model...")
    cnn_model = create_model(X_train)
    model, history = train_model(cnn_model, X_train, X_test, y_train, y_test, epochs=100, batch_size=64)
    evaluate_model(cnn_model, X_test, y_test)
    date = datetime.now().strftime('%Y%m%d%H%M')
    val_acc = history.history['val_accuracy'][-1]
    model.save(f'models/model_mel_{date}_acc_{val_acc:.2f}.h5')
    print(f"Model saved as models/model_mel_{date}_acc_{val_acc:.2f}.h5")
    plot_loss(history)
