from tqdm import tqdm
import os
import shutil



def format_files(root_dir, leak_dir, no_leak_dir):
    # 遍歷各個區處
    for area in os.listdir(root_dir):
        area_path = os.path.join(root_dir, area)
        if os.path.isdir(area_path):  # 確認是資料夾
            # 遍歷區處內的資料夾
            for folder in os.listdir(area_path):
                folder_path = os.path.join(area_path, folder)
                if os.path.isdir(folder_path) and folder.isdigit():  # 確認是資料夾且資料夾名稱是數字
                    # 將資料夾名稱轉為整數
                    folder_num = int(folder)
                    if folder_num < 30 or folder_num > 80 or folder_num == 50:  # 過濾 < 30, > 80, = 50 的資料夾 ，只處理 30 ~ 80 的資料夾
                        continue
                    # 遍歷該資料夾內的子資料夾
                    for subfolder in tqdm(os.listdir(folder_path), desc=f'處理 {area_path}-{folder}'):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.isdir(subfolder_path):  # 確認是子資料夾
                            # 遍歷子資料夾內的檔案
                            for file in os.listdir(subfolder_path):
                                # 只處理 .wav 和 .WAV 檔案

                                if file.endswith(('.wav', '.WAV')):
                                    file_path = os.path.join(subfolder_path, file)
                                    # 判斷資料夾名稱來分類
                                    if folder_num >= 60:
                                        target_dir = leak_dir
                                    if folder_num <= 40:
                                        target_dir = no_leak_dir

                                    # 新的檔名格式：裝置名稱 (subfolder) + rate_(folder_num) + 原檔名
                                    new_file_name = f"{subfolder}_rate_{folder_num}_{file}"
                                    new_file_path = os.path.join(target_dir, new_file_name)

                                    # print(f'複製 {file_path} 到 {new_file_path}')

                                    # 複製並重新命名檔案到相應的目標資料夾
                                    shutil.copy(file_path, new_file_path)

if __name__ == '__main__':
    # 定義根目錄
    root_dir = 'dataset'  # 根據你的情況，調整根目錄的名稱
    leak_dir = 'training_data/Leak'
    no_leak_dir = 'training_data/no-leak'

    # 創建目標目錄
    os.makedirs(leak_dir, exist_ok=True)
    os.makedirs(no_leak_dir, exist_ok=True)
    format_files(root_dir, leak_dir, no_leak_dir)