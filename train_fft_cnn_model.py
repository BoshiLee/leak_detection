import argparse
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cnn_model import create_model, train_model, evaluate_model, create_1d_cnn_model
from create_dataset import create_dataset
from feature_extraction import compute_fft
from plotting import plot_training_history

def preprocess_data(X, desired_time=2.0, n_fft=2048):
    features = []
    for audio, sr in tqdm(X):
        _, feature = compute_fft(audio, sr, desired_time=desired_time, n_fft=n_fft)
        features.append(feature)
    return np.array(features)

def create_training_data(X, y, desired_time=2.0, n_fft=2048):
    print(f"總樣本數: {len(X)}")
    print(f"標籤分佈: {np.bincount(y)}")

    X_features = preprocess_data(X, desired_time=desired_time, n_fft=n_fft)

    print(f"特徵形狀: {X_features.shape}")  # 預期形狀: (樣本數, max_len, n_mels)

    # 正規化特徵
    X_features = (X_features - np.mean(X_features)) / np.std(X_features)

    # 擴展維度以符合 CNN 輸入 (樣本數, 長度, 1)
    X_features = np.expand_dims(X_features, -1)

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
    n_fft = int(os.getenv('N_FFT'))

    print(f"Desired time: {desired_time}")
    print(f"Number of FFT points: {n_fft}")

    print("Creating dataset...")
    X, y = create_dataset(dataset_path)
    X_train, X_test, y_train, y_test = create_training_data(X, y,
                                                            desired_time=desired_time,
                                                            n_fft=n_fft,)

    print("Creating model...")
    cnn_model = create_1d_cnn_model(X_train.shape[1:])
    model, history = train_model(cnn_model, X_train, X_test, y_train, y_test, epochs=100, batch_size=512)
    acc, loss = evaluate_model(model, X_test, y_test)
    date = datetime.now().strftime('%Y%m%d%H%M')
    model_name = f'model_mel_{date}_acc_{acc:.2f}_loss_{loss:.2f}'
    model.save(f'models/{model_name}.h5')
    print(f"Model saved as models/{model_name}.h5")
    plot_training_history(history, model_name)
