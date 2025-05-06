import argparse
import os
import pandas as pd
from pathlib import Path
import paramiko
from paramiko import SSHException
from dotenv import load_dotenv
from tqdm import tqdm
from tkinter import filedialog, Tk

# 讀取 .env 檔案中的環境變數
load_dotenv()
SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_PORT = int(os.getenv('SFTP_PORT'))
SFTP_USERNAME = os.getenv('SFTP_USERNAME')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD')
REMOTE_BASE_DIR = '/itri/Taiwan Water Local/Hsinchu City'
LOCAL_BASE_DIR = Path('dataset')

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

def connect_sftp():
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    try:
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("SFTP 連線成功")
        return sftp
    except SSHException as ssh_e:
        print(f"SFTP 連線失敗：{ssh_e}")
        return None

def close_sftp(transport):
    transport.close()
    print("SFTP 連線已關閉")

def download_wav_and_images(csv_path, sftp):
    # 讀取 CSV
    df = pd.read_csv(csv_path)

    try:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            download_file(sftp, row)
    except SSHException as ssh_e:
        print(f"SFTP 連線失敗：{ssh_e}")

def read_line(row):
    uid = row['id']
    serial_number = row['serial_number']
    date = row['date']  # 假設 date 格式為 'YYYY-MM-DD'
    wav_data = row['wav_data']
    ai_result = row['ai_result']
    # dip, env, pvc, tfc
    dip = row['dip']
    env = row['env']
    pvc = row['pvc']
    tfc = row['tfc']

    return uid, serial_number, date, wav_data, ai_result, dip, env, pvc, tfc


def download_file(sftp, row):
    serial_number = row['serial_number']
    date = row['date']  # 假設 date 格式為 'YYYY-MM-DD'
    wav_data = row['wav_data']
    ai_result = row['ai_result']

    # 計算 ai_result_dir
    ai_result_dir = get_ai_result_dir(ai_result)
    local_root_dir = LOCAL_BASE_DIR / date

    # 設定本地儲存路徑
    local_dir = local_root_dir / ai_result_dir / serial_number
    local_dir.mkdir(parents=True, exist_ok=True)

    # 解析 wav_data（假設 wav_data 包含檔案名稱或路徑）
    # 根據您的 CSV 範例，wav_data 似乎是 JSON 格式的字串
    # 您需要根據實際情況調整解析方式

    wav_file_name = wav_data.strip()

    # 構建遠端檔案路徑
    remote_wav_path = f"{REMOTE_BASE_DIR}/{serial_number}/WAV/{date}/{wav_file_name}.wav"
    remote_img_1d_path = f"{REMOTE_BASE_DIR}/{serial_number}/IMG/{date}/{wav_file_name}_1d.png"
    remote_img_3d_path = f"{REMOTE_BASE_DIR}/{serial_number}/IMG/{date}/{wav_file_name}_3d.png"
    try:
        # 設定本地檔案路徑

        local_file_path = str(local_dir / f"{wav_file_name}.wav")
        local_image_1d_path = str(local_dir / f"{wav_file_name}_1d.png")
        local_image_3d_path = str(local_dir / f"{wav_file_name}_3d.png")

        # 下載檔案
        # tqdm.write(f"下載 {remote_wav_path} 至 {local_file_path}")
        sftp.get(remote_wav_path, local_file_path)
        # tqdm.write(f"下載 {remote_img_1d_path} 至 {local_image_1d_path}")
        sftp.get(remote_img_1d_path, local_image_1d_path)
        # tqdm.write(f"下載 {remote_img_3d_path} 至 {local_image_3d_path}")
        sftp.get(remote_img_3d_path, local_image_3d_path)

        return local_file_path, local_image_1d_path, local_image_3d_path

    except Exception as e:
        print(f"下載 {remote_wav_path} 失敗：{e}")

def main():
    # GUI 選擇資料夾
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='請選擇含 CSV 的資料夾')
    if not folder_path:
        print("❌ 未選擇資料夾，程式結束")
        return
    csv_dir = os.path.abspath(folder_path)

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            print(f"下載 {csv_file} 開始")
            sftp = connect_sftp()
            if sftp:
                download_wav_and_images(csv_path, sftp)
                close_sftp(sftp)
            print(f"下載 {csv_file} 完成")
    print("✅ 所有檔案處理完成，SFTP 連線已關閉")

if __name__ == "__main__":
    main()
