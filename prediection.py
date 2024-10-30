import argparse
import os

from tensorflow.keras.models import load_model
from data_preprocess import preprocess_audio

def predict_leak(model_path, file_path, desired_time=2.0):
    # 載入模型
    model = load_model(model_path)

    # 預處理音訊
    feature = preprocess_audio(file_path, desired_time=desired_time)

    # 預測
    prediction = model.predict(feature)

    # 解析預測結果
    label = 1 if prediction[0][0] >= 0.5 else 0
    confidence = prediction[0][0]

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict if there is a leak in the audio file.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file.')
    parser.add_argument('--dir', type=str, help='Path to the audio file to predict.')
    args = parser.parse_args()
    model_path = args.model_path

    test_audio_dir = args.dir

    result_dict: dict[int, list[str]] = {0: [], 1: []}


    for file in os.listdir(test_audio_dir):
        file_path = os.path.join(test_audio_dir, file)
        label, confidence = predict_leak(model_path, file_path, desired_time=1.0)
        if label == 1:
            result_dict[1].append(f"偵測到漏水，信心度: {confidence:.2f} ({file})")
        else:
            result_dict[0].append(f"未偵測到漏水，信心度: {1 - confidence:.2f} ({file})")

    print(f'共有 {len(result_dict[1])} 個檔案偵測到漏水，{len(result_dict[0])} 個檔案未偵測到漏水。')
    for label, results in result_dict.items():
        print(f"類別 {label}:")
        for result in results:
            print(result)
        print()