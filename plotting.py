import os
import tempfile

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import matplotlib.image as mpimg
import librosa
import librosa.display
from matplotlib import gridspec
from scipy.fft import fft, ifft
from scipy.io import wavfile
from feature_extraction import extract_features, extract_stft_features, compute_fft
from sklearn.decomposition import PCA

load_dotenv()

sample_rate = int(os.getenv("SAMPLE_RATE"))
n_fft = int(os.getenv("N_FFT"))
n_mels = int(os.getenv("N_MELS"))
mels_n_fft = int(os.getenv("MELS_N_FFT"))
mels_hop_length = int(os.getenv("MELS_HOP_LENGTH"))
stft_n_fft = int(os.getenv("STFT_N_FFT"))
stft_hop_length = int(os.getenv("STFT_HOP_LENGTH"))
desired_time = float(os.getenv("DESIRED_TIME"))
# ✅ 頻段參數
freq_ranges = [(0, 1000), (1000, 3000)]
print('n_mels:', n_mels)
print('mels_n_fft:', mels_n_fft)
print('mels_hop_length:', mels_hop_length)
print(f'freq_ranges:', freq_ranges)

# Windows 用 'Microsoft JhengHei'，Mac 用 'PingFang TC'，Linux 可試 'Noto Sans CJK TC'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 適用於 Windows
plt.rcParams['axes.unicode_minus'] = False  # 避免負號變成方塊

def generate_segments(sig, window=1.0):
    """
    根據輸入的音訊資料切成每段 window 秒的區段

    Args:
        sig (np.ndarray): 音訊資料（1D or 2D）
        window (float): 每段長度（秒）

    Returns:
        list: [[start, end], ...] 的 segment_table
    """
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    duration = len(sig) / sample_rate

    segments = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        segments.append([start, end])
        start = end
    return segments

def FFT_filter(sig, segment_table, bandpass=None):
    """
    對給定的音訊資料進行 FFT 分析與 IFFT 還原，可加 bandpass 遮罩

    Args:
        sig (np.ndarray): 音訊訊號
        segment_table (list): [[start, end], ...]（秒為單位）
        bandpass (tuple or None): 頻率遮罩區間 (low_freq, high_freq)

    Returns:
        Xf_list, Yf_list, X_list, Y_list: 頻率、頻譜、還原時間序列等資訊
    """
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    time = len(sig) / sample_rate

    Yf_list = []
    Xf_list = []
    X_list = []
    Y_list = []

    for start, end in segment_table:
        if end > time:
            print(f"⚠️ 區段 {start}-{end} 超出音檔長度 {time:.2f}s，已修正 end 為 {time:.2f}s")
            end = time
        if start >= end:
            print(f"⚠️ 無效區段：start={start}, end={end}，已跳過")
            continue

        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment = sig[start_idx:end_idx]

        N = len(segment)
        Yf = fft(segment) / sample_rate
        freqs = np.fft.fftfreq(N, d=1/sample_rate)

        if bandpass is not None:
            low, high = bandpass
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
            Yf_filtered = np.zeros_like(Yf, dtype=complex)
            Yf_filtered[mask] = Yf[mask]
        else:
            Yf_filtered = Yf

        freqs_pos = freqs[:N//2]
        Yf_pos = np.abs(Yf[:N//2])
        y_rec = ifft(Yf_filtered).real
        x_rec = np.linspace(0, end - start, N)

        Xf_list.append(freqs_pos)
        Yf_list.append(Yf_pos)
        X_list.append(x_rec)
        Y_list.append(y_rec)

        # print(f"✅ 處理 segment: {start:.2f} → {end:.2f} 秒, 長度：{end - start:.2f} 秒")

    return Xf_list, Yf_list, X_list, Y_list

def get_audio(wav):
    if wav.ndim > 1:
        sig = wav.mean(axis=1)  # 轉單聲道
    time = len(sig) / sample_rate
    return sig, time

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
    axs[1, 2].set_xlim(0, 1000)
    axs[1, 2].set_title(f"FFT of Audio Signal {file_name}", fontsize=16)
    axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=14)
    axs[1, 2].set_ylabel("Magnitude", fontsize=14)
    axs[1, 2].grid(True)




    # ✅ 分段參數
    segment_table = generate_segments(wav, window=1.0)
    xf, yf, x, y = FFT_filter(wav, segment_table, bandpass=None)

    # ✅ 全段 bandpass 重建
    time = len(wav) / sample_rate
    N = len(wav)
    t = np.linspace(0, time, N)
    Yf = fft(wav) / sample_rate
    freqs = np.fft.fftfreq(N, d=1 / sample_rate)

    bandpassed_signals = []
    for (low, high) in freq_ranges:
        mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
        Yf_masked = np.zeros_like(Yf, dtype=complex)
        Yf_masked[mask] = Yf[mask]
        y_filtered = ifft(Yf_masked).real
        bandpassed_signals.append((f"{low}-{high}Hz", y_filtered))

    # 🔄 新圖形整合（額外加一張大圖，不塞進原 fig）
    fig2 = plt.figure(figsize=(16, 2 + len(x) * 2.5 + len(bandpassed_signals) * 2.5))
    gs = gridspec.GridSpec(len(x) + len(bandpassed_signals) + 1, len(freq_ranges) + 1, figure=fig2)

    # 🔹 多段頻譜圖
    for i in range(len(x)):
        seg_start, seg_end = segment_table[i]
        for j, (f_start, f_end) in enumerate(freq_ranges):
            ax = fig2.add_subplot(gs[i, j])
            ax.plot(xf[i], yf[i])
            ax.set_xlim(f_start, f_end)
            ax.set_title(f"{seg_start:.1f}-{seg_end:.1f}s\n{f_start}-{f_end}Hz")
        ax = fig2.add_subplot(gs[i, -1])
        ax.plot(x[i], y[i], color='red')
        ax.set_title(f"{seg_start:.1f}-{seg_end:.1f}s - IFFT")

    # 🔹 原始音訊 + 各頻段還原
    ax = fig2.add_subplot(gs[len(x), :])
    ax.plot(t, wav, color="gray")
    ax.set_title("Full Audio - Original Signal")

    for i, (label, yfilt) in enumerate(bandpassed_signals):
        ax = fig2.add_subplot(gs[len(x) + i + 1, :])
        ax.plot(t, yfilt)
        ax.set_title(f"Bandpassed: {label}")

    # 儲存主圖（Mel + STFT + FFT + 表格 + 1D/3D）
    os.makedirs(f"images/{serial_number}_{class_type}", exist_ok=True)
    main_fig_path = f"images/{serial_number}_{class_type}/uid_{uid}_{file_name}_mel_stft_fft_1d_3d.png"
    fig.tight_layout()
    fig.savefig(main_fig_path)  # ✅ 用 fig 儲存，而非 plt
    if dev:
        plt.show()
    plt.close(fig)
    # print(f"✅ 主圖儲存於：{main_fig_path}")

    # 儲存副圖（Segment FFT + bandpassed）
    bandpass_fig_path = f"images/{serial_number}_{class_type}/uid_{uid}_{file_name}_segment_bandpass.png"
    fig2.tight_layout()
    fig2.savefig(bandpass_fig_path)
    if dev:
        plt.show()
    plt.close(fig2)
    # print(f"✅ Bandpass 圖儲存於：{bandpass_fig_path}")

    analyze_pca_from_audio(y=wav, sr=sample_rate, duration=1,
                           n_fft=n_fft,
                           hop_length=stft_hop_length,
                           n_components=3,
                           n_mels=n_mels,
                           output_path=f"images/{serial_number}_{class_type}/uid_{uid}_{file_name}_pca.png")

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




def analyze_pca_from_audio(
    y: np.ndarray,
    sr: int,
    duration: float = 1.0,
    n_fft: int = 2048,
    hop_length: int = 256,
    n_components: int = 3,
    n_mels: int = 40,
    output_path: str = None
):
    # 取前 duration 秒資料
    sample_count = int(sr * duration)
    y_segment = y[:sample_count]

    # === STFT ===
    S = np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # === STFT PCA ===
    S_T = S_db.T
    pca_stft = PCA(n_components=n_components)
    S_pca_stft = pca_stft.fit_transform(S_T)
    time_axis = np.arange(S_pca_stft.shape[0]) * hop_length / sr

    # === Mel Spectrogram ===
    S_mel = librosa.feature.melspectrogram(
        y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels)

    # === Mel PCA ===
    S_mel_T = S_mel_db.T
    pca_mel = PCA(n_components=n_components)
    S_pca_mel = pca_mel.fit_transform(S_mel_T)

    # === Plotting ===
    fig, axes = plt.subplots(n_components * 4, 1, figsize=(12, 4 * n_components * 2), sharex=False)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i in range(n_components):
        color = colors[i % len(colors)]
        axes[4 * i].plot(time_axis, S_pca_stft[:, i], label=f'STFT PC{i+1} (Time)', color=color, linestyle='-')
        axes[4 * i].set_ylabel("Value")
        axes[4 * i].set_title(f'STFT PCA - PC{i+1} - Time Domain')
        axes[4 * i].legend()

        axes[4 * i + 1].plot(freqs, pca_stft.components_[i], label=f'STFT PC{i+1} (Freq)', color=color)
        axes[4 * i + 1].set_xlabel("Frequency (Hz)")
        axes[4 * i + 1].set_ylabel("Weight")
        axes[4 * i + 1].set_title(f'STFT PCA - PC{i+1} - Frequency Domain')
        axes[4 * i + 1].legend()

        axes[4 * i + 2].plot(time_axis, S_pca_mel[:, i], label=f'Mel PC{i+1} (Time)', color=color, linestyle='--')
        axes[4 * i + 2].set_ylabel("Value")
        axes[4 * i + 2].set_title(f'Mel PCA - PC{i+1} - Time Domain')
        axes[4 * i + 2].legend()

        axes[4 * i + 3].plot(mel_freqs, pca_mel.components_[i], label=f'Mel PC{i+1} (Mel Freq)', color=color, linestyle='--')
        axes[4 * i + 3].set_xlabel("Frequency (Hz)")
        axes[4 * i + 3].set_ylabel("Weight")
        axes[4 * i + 3].set_title(f'Mel PCA - PC{i+1} - Frequency Domain')
        axes[4 * i + 3].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        # print(f"📁 圖片已儲存至：{output_path}")
    else:
        plt.show()
    plt.close(fig)
