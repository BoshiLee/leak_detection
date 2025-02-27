import os

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import load_dotenv

from feature_extraction import extract_features, extract_stft_features, compute_fft

load_dotenv()

sample_rate = int(os.getenv("SAMPLE_RATE"))
n_fft = int(os.getenv("N_FFT"))
n_mels = int(os.getenv("N_MELS"))
mels_n_fft = int(os.getenv("MELS_N_FFT"))
mels_hop_length = int(os.getenv("MELS_HOP_LENGTH"))
stft_n_fft = int(os.getenv("STFT_N_FFT"))
stft_hop_length = int(os.getenv("STFT_HOP_LENGTH"))
desired_time = float(os.getenv("DESIRED_TIME"))

print('n_mels:', n_mels)
print('mels_n_fft:', mels_n_fft)
print('mels_hop_length:', mels_hop_length)

# Windows 用 'Microsoft JhengHei'，Mac 用 'PingFang TC'，Linux 可試 'Noto Sans CJK TC'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 適用於 Windows
plt.rcParams['axes.unicode_minus'] = False  # 避免負號變成方塊

def plot_mel_stft_fft(wav, file_name, class_type='no-leak'):

    mel_spectrogram = extract_features(wav, sr=sample_rate, n_mels=n_mels, n_fft=mels_n_fft, hop_length=mels_hop_length, desired_time=desired_time, enhanced=1, transpose=False)
    S_db = extract_stft_features(wav, sr=sample_rate, n_fft=stft_n_fft, hop_length=stft_hop_length, desired_time=desired_time, transpose=False)
    frequencies, fft_magnitude = compute_fft(wav, sample_rate=sample_rate, n_fft=n_fft, desired_time=desired_time)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 繪製 Mel 頻譜圖
    mel_img = librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=mels_hop_length, x_axis='time', y_axis='mel', cmap='jet', ax=axs[0])
    axs[0].set_title(f'Mel Spectrogram {file_name}')
    axs[0].set_ylim(0, 1600)
    axs[0].set_yticks(np.arange(0, 1600, 150))
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(mel_img, ax=axs[0], format='%+2.0f dB')  # 添加 colorbar

    # 繪製 STFT 頻譜圖
    stft_img = librosa.display.specshow(S_db, sr=sample_rate, hop_length=128, x_axis='time', y_axis='log', cmap='jet', ax=axs[1])
    axs[1].set_title(f'STFT Magnitude {file_name}')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(stft_img, ax=axs[1], format='%+2.0f dB')  # 添加 colorbar

    # 繪製 FFT 頻譜圖
    axs[2].plot(frequencies, fft_magnitude)
    axs[2].set_xlim(0, 3600)
    axs[2].set_title(f"FFT of Audio Signal {file_name}")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Magnitude")
    axs[2].grid(True)

    # 確保目錄存在
    os.makedirs(f"images/{class_type}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{class_type}/{file_name}_mel_stft_fft.png")
    plt.show()
    plt.close()


def plot_mel_stft_fft_1d_3d(wav, file_name, one_d_path, three_d_path, class_type='no-leak', uid='12345', serial_number="ITRI000000", leak_rate=0,
                            env_noise=0, dip_noise=0, pvc_pipe=0, tfc_noise=0, dev=False):
    mel_spectrogram = extract_features(wav, sr=sample_rate, n_mels=n_mels, n_fft=mels_n_fft, hop_length=mels_hop_length,
                                       desired_time=desired_time, enhanced=1, transpose=False)
    S_db = extract_stft_features(wav, sr=sample_rate, n_fft=stft_n_fft, hop_length=stft_hop_length,
                                 desired_time=desired_time, transpose=False)
    frequencies, fft_magnitude = compute_fft(wav, sample_rate=sample_rate, n_fft=n_fft, desired_time=desired_time)

    # 設定 2 行 3 列的子圖
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # **在 axs[0, 0] 顯示表格**
    axs[0, 0].axis('off')  # 隱藏座標軸
    table_data = [
        ["UID", uid],
        ["序號", serial_number],
        ["擷取時間", file_name],
        ["洩漏率", f"{leak_rate}%"],
        ["環境音", f"{env_noise}%"],
        ["金屬音", f"{dip_noise}%"],
        ["複合管", f"{pvc_pipe}%"],
        ["雜訊音", f"{tfc_noise}%"]
    ]

    table = axs[0, 0].table(cellText=table_data, colWidths=[0.3, 0.4], cellLoc='center', loc='center',
                            edges='horizontal')

    # **調整表格樣式**
    table.auto_set_font_size(False)
    table.set_fontsize(16)  # 設定字體大小為 16
    table.scale(1.2, 1.2)  # 調整表格大小

    # **讓每一列有 1px 空間**
    for i, key in enumerate(table._cells):
        cell = table._cells[key]
        cell.set_height(1 / (len(table_data) + 1) - 0.01)  # 保持 1px 的間隔
        cell.set_fontsize(16)  # 設定字體大小
        if key[1] == 0:  # 讓左邊的第一列 (標題) 加粗
            cell.set_text_props(weight='bold')

    axs[0, 0].set_title("數據資訊", fontsize=16)

    # **繪製 1D 頻譜圖（移到 axs[0, 1]）**
    one_d_img = mpimg.imread(one_d_path)
    axs[0, 1].imshow(one_d_img)
    axs[0, 1].set_title('1D 頻譜圖', fontsize=16)
    axs[0, 1].axis('off')

    # **繪製 3D 頻譜圖（移到 axs[0, 2]）**
    three_d_img = mpimg.imread(three_d_path)
    axs[0, 2].imshow(three_d_img)
    axs[0, 2].set_title('3D 頻譜圖', fontsize=16)
    axs[0, 2].axis('off')

    # **繪製 Mel 頻譜圖（移到 axs[1, 0]）**
    mel_img = librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=mels_hop_length, x_axis='time',
                                       y_axis='mel', cmap='jet', ax=axs[1, 0])
    axs[1, 0].set_title(f'Mel Spectrogram {file_name}', fontsize=16)
    axs[1, 0].set_ylim(0, 1600)
    axs[1, 0].set_yticks(np.arange(0, 1600, 150))
    axs[1, 0].set_xlabel('Time', fontsize=14)
    axs[1, 0].set_ylabel('Frequency (Hz)', fontsize=14)
    fig.colorbar(mel_img, ax=axs[1, 0], format='%+2.0f dB')

    # **繪製 STFT 頻譜圖（移到 axs[1, 1]）**
    stft_img = librosa.display.specshow(S_db, sr=sample_rate, hop_length=128, x_axis='time', y_axis='log', cmap='jet',
                                        ax=axs[1, 1])
    axs[1, 1].set_title(f'STFT Magnitude {file_name}', fontsize=16)
    axs[1, 1].set_xlabel('Time', fontsize=14)
    axs[1, 1].set_ylabel('Frequency (Hz)', fontsize=14)
    fig.colorbar(stft_img, ax=axs[1, 1], format='%+2.0f dB')

    # **繪製 FFT 頻譜圖（移到 axs[1, 2]）**
    axs[1, 2].plot(frequencies, fft_magnitude)
    axs[1, 2].set_xlim(0, 6600)
    axs[1, 2].set_title(f"FFT of Audio Signal {file_name}", fontsize=16)
    axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=14)
    axs[1, 2].set_ylabel("Magnitude", fontsize=14)
    axs[1, 2].grid(True)

    # 確保目錄存在
    os.makedirs(f"images/{serial_number}_{class_type}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{serial_number}_{class_type}/{file_name}_mel_stft_fft_1d_3d.png")
    if dev:
        plt.show()

    plt.close()

def plot_training_history(history, model_name):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='val accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')

    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='val loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')

    os.makedirs('images/model_loss', exist_ok=True)
    plt.savefig(f'images/model_loss/{model_name}.png')

    plt.show()