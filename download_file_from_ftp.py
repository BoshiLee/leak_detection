import argparse
import os

import pandas as pd
from pathlib import Path
import paramiko
from paramiko import SSHException
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_PORT = int(os.getenv('SFTP_PORT'))
SFTP_USERNAME = os.getenv('SFTP_USERNAME')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD')
REMOTE_BASE_DIR = '/itri/Taiwan Water Local/Hsinchu City'
LOCAL_BASE_DIR = Path('dataset/ftp')

# 讀取 CSV 檔案

def get_ai_result_dir(ai_result):
    """
    根據 ai_result 計算 ai_result_dir。
    ai_result 在 50~59 為 50，60~69 為 60，以此類推。
    """
    try:
        ai_result_int = int(float(ai_result))
        ai_result_dir = (ai_result_int // 10) * 10
        return str(ai_result_dir)
    except ValueError:
        return 'unknown'


def download_wav_files(csv_path):
    # 讀取 CSV
    df = pd.read_csv(csv_path)

    # 初始化 SFTP 連線
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    try:
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("SFTP 連線成功")

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                serial_number = row['serial_number']
                date = row['date']  # 假設 date 格式為 'YYYY-MM-DD'
                wav_data = row['wav_data']
                ai_result = row['ai_result']

                # 計算 ai_result_dir
                ai_result_dir = get_ai_result_dir(ai_result)

                # 設定本地儲存路徑
                local_dir = LOCAL_BASE_DIR / ai_result_dir / serial_number
                local_dir.mkdir(parents=True, exist_ok=True)

                # 解析 wav_data（假設 wav_data 包含檔案名稱或路徑）
                # 根據您的 CSV 範例，wav_data 似乎是 JSON 格式的字串
                # 您需要根據實際情況調整解析方式

                wav_file_name = wav_data.strip()

                # 構建遠端檔案路徑
                remote_path = f"{REMOTE_BASE_DIR}/{serial_number}/WAV/{date}/{wav_file_name}.wav"

                # 設定本地檔案路徑
                local_file_path = local_dir / f"{wav_file_name}.wav"

                # 下載檔案
                tqdm.write(f"下載第 {index + 1} 行：{remote_path} 到 {local_file_path}")
                sftp.get(remote_path, str(local_file_path))
                tqdm.write(f"第 {index + 1} 行下載完成")

            except Exception as e:
                print(f"第 {index + 1} 行下載失敗：{e}")

    except SSHException as ssh_e:
        print(f"SFTP 連線失敗：{ssh_e}")
    finally:
        transport.close()
        print("SFTP 連線已關閉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download wav files from SFTP server.')
    parser.add_argument('csv_dir', type=str, help='Path to the CSV file containing the file information.')

    args = parser.parse_args()
    csv_dir = os.path.abspath(args.csv_dir)

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            download_wav_files(csv_path)
            print(f"下載 {csv_file} 完成")