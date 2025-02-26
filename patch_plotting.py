import argparse
import pandas as pd
import librosa
from tqdm import tqdm
from query_wav_data import export_query_results
from plotting import plot_mel_stft_fft_1d_3d
from download_file_from_ftp import read_line, download_file, connect_sftp, close_sftp


def load_wav_images(local_file_path, target_sr=32000):
    y, sr = librosa.load(local_file_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def main(date_range, serial_number):
    # 匯出 CSV 檔案
    csv_path = export_query_results(date_range=date_range, serial_number=serial_number)
    print(f"CSV 檔案已匯出至 {csv_path}")

    # 讀取 CSV 檔案
    df = pd.read_csv(csv_path)

    # 連接 SFTP
    sftp = connect_sftp()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        uid, serial_number, date, wav_data, ai_result, dip, env, pvc, tfc = read_line(row)
        local_file_path, local_image_1d_path, local_image_3d_path = download_file(sftp, row)

        y = load_wav_images(local_file_path)

        plot_mel_stft_fft_1d_3d(
            y,
            one_d_path=local_image_1d_path,
            three_d_path=local_image_3d_path,
            class_type=date_range,
            uid=uid,
            serial_number=serial_number,
            leak_rate=ai_result,
            env_noise=env,
            dip_noise=dip,
            pvc_pipe=pvc,
            tfc_noise=tfc,
            file_name=wav_data,
            dev=False
        )

    # 關閉 SFTP 連線
    close_sftp(sftp)
    print("處理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="處理 WAV 數據並繪製頻譜圖")
    parser.add_argument("date_range", type=str, help="指定查詢的日期範圍 (格式：YYYY-MM-DD)")
    parser.add_argument("serial_number", type=str, help="設備序列號")

    args = parser.parse_args()
    main(args.date_range, args.serial_number)