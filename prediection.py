import argparse
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from data_preprocess import preprocess_audio

def process_audio(file_path, sr=4800, n_mels=40, n_fft=512, hop_length=64, desired_time=2.0):
    # 預處理音訊
    feature = preprocess_audio(file_path, traget_sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, desired_time=desired_time)
    # feature = preprocess_stft_audio(file_path, desired_time=desired_time)
    return feature

def predict_leak(model_path, feature):
    # 載入模型
    model = load_model(model_path)
    # 預測
    prediction = model.predict(feature)

    # 解析預測結果
    label = 1 if prediction[0][0] >= 0.60 else 0
    confidence = prediction[0][0]

    return label, confidence


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    sample_rate = int(os.getenv('SAMPLE_RATE'))
    desired_time = float(os.getenv('DESIRED_TIME'))
    n_mels = int(os.getenv("N_MELS"))
    mels_n_fft = int(os.getenv("MELS_N_FFT"))
    mels_hop_length = int(os.getenv("MELS_HOP_LENGTH"))

    parser = argparse.ArgumentParser(description='Predict if there is a leak in the audio file.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file.')
    parser.add_argument('--dir', type=str, help='Path to the directory containing audio files to predict.')
    args = parser.parse_args()
    # print model name only file name
    model_name = os.path.basename(args.model_path).split('.h5')[0]
    print(f'use model: {model_name}')

    audio_dir = os.path.abspath(args.dir)
    print(f'audio_dir: {audio_dir}')

    model_path = os.path.abspath(args.model_path)
    test_audio_dir = os.path.abspath(args.dir)

    result_dict: dict[int, list[str]] = {0: [], 1: []}

    # 檢查目錄是否存在
    if not os.path.isdir(test_audio_dir):
        print(f"目錄 {test_audio_dir} 不存在，請檢查路徑是否正確。")
    else:
        # 讀取資料夾內的每個檔案
        for file in os.listdir(test_audio_dir):
            file_path = os.path.join(test_audio_dir, file)
            if not file.endswith(('.wav', '.WAV')):
                print(f"檔案 {file_path} 不是 .wav 或 .WAV 檔案，已跳過該檔案。")
                continue
            try:
                feature = process_audio(file_path, sr=sample_rate, desired_time=desired_time, n_mels=n_mels, n_fft=mels_n_fft, hop_length=mels_hop_length)
                # 呼叫預測函數
                label, confidence = predict_leak(model_path, feature)
                if label == 1:
                    result_dict[1].append(f"偵測到漏水，信心度: {confidence:.2f} ({file})")
                else:
                    result_dict[0].append(f"未偵測到漏水，信心度: {1 - confidence:.2f} ({file})")
            except PermissionError:
                print(f"無法讀取檔案 {file_path}，可能是權限問題，已跳過該檔案。")
            except Exception as e:
                print(f"處理檔案 {file_path} 時發生錯誤：{e}，已跳過該檔案。")

    # 顯示結果
    print(f'共有 {len(result_dict[1])} 個檔案偵測到漏水，{len(result_dict[0])} 個檔案未偵測到漏水。')
    # 寫到 prediction_log_{date).txt
    date = datetime.now().strftime('%Y%m%d%H%M')
    with open(f'{audio_dir}\\{model_name}_prediction_log_{date}.txt', 'w', encoding='utf-8') as f:
        for label, results in result_dict.items():
            f.write(f"類別 {label}:\n")
            print(f"類別 {label}:")
            for result in results:
                f.write(result + '\n')
                print(result)
            f.write('\n')