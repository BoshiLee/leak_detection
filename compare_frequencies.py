import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def analyze_and_compare(background_path, test_path, output_dir=None, no_show=False):
    """
    載入背景音和待測音，分析其頻譜並進行視覺化比較。
    此版本會同時產生 dB 刻度和對數刻度兩種圖表。

    Args:
        background_path (str): 背景音檔的路徑。
        test_path (str): 待測音檔的路徑。
        output_dir (str, optional): 儲存結果圖檔的資料夾路徑. Defaults to None.
        no_show (bool, optional): 若為 True，則不顯示互動視窗. Defaults to False.
    """
    try:
        # 載入音檔，保留原始取樣率
        y_bg, sr_bg = librosa.load(background_path, sr=None)
        y_test, sr_test = librosa.load(test_path, sr=None)

        # 確保取樣率一致，若不一致則重取樣 (以背景音為準)
        if sr_test != sr_bg:
            y_test = librosa.resample(y_test, orig_sr=sr_test, target_sr=sr_bg)
            sr_test = sr_bg

        # 設定 Matplotlib 使用支援中文的字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # --- 1. 計算靜態頻譜 (FFT) 並找出主頻率 ---
        def get_spectrum_and_peak(y, sr):
            fft_result = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(y), d=1 / sr)
            magnitude = np.abs(fft_result)
            positive_freq_indices = np.where(freqs >= 0)
            freqs_pos = freqs[positive_freq_indices]
            magnitude_pos = magnitude[positive_freq_indices]
            peak_index = np.argmax(magnitude_pos)
            peak_freq = freqs_pos[peak_index]
            return freqs_pos, magnitude_pos, peak_freq

        freqs_bg, mag_bg, peak_bg = get_spectrum_and_peak(y_bg, sr_bg)
        freqs_test, mag_test, peak_test = get_spectrum_and_peak(y_test, sr_test)

        # --- 內部輔助函式，用於產生指定類型的圖表 ---
        def _generate_plot(yscale_type):
            """根據指定的 yscale_type ('db' 或 'log') 產生並回傳一個圖表物件。"""
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
            fig.suptitle(f'環境音與洩漏音頻率比較 (Y軸: {yscale_type.upper()})', fontsize=20)

            # --- 上方圖：根據 yscale_type 決定繪圖方式 ---
            ax0 = fig.add_subplot(gs[0, :])
            if yscale_type == 'db':
                db_bg = librosa.amplitude_to_db(mag_bg, ref=np.max)
                db_test = librosa.amplitude_to_db(mag_test, ref=np.max)
                ax0.plot(freqs_bg, db_bg, label=f'背景音 (主頻率: {peak_bg:.0f} Hz)', color='blue', alpha=0.7)
                ax0.plot(freqs_test, db_test, label=f'待測音 (主頻率: {peak_test:.0f} Hz)', color='red', alpha=0.7)
                ax0.set_ylabel('幅度 (dB)')
                ax0.set_ylim(-80, max(np.max(db_bg), np.max(db_test)) + 5)
            elif yscale_type == 'log':
                ax0.semilogy(freqs_bg, mag_bg, label=f'背景音 (主頻率: {peak_bg:.0f} Hz)', color='blue', alpha=0.7)
                ax0.semilogy(freqs_test, mag_test, label=f'待測音 (主頻率: {peak_test:.0f} Hz)', color='red', alpha=0.7)
                ax0.set_ylabel('幅度 (對數刻度)')

            # 上方圖的通用設定
            ax0.axvline(x=peak_bg, color='blue', linestyle='--', alpha=0.8, label=f'背景音峰值')
            ax0.axvline(x=peak_test, color='red', linestyle='--', alpha=0.8, label=f'待測音峰值')
            ax0.set_title('頻譜能量分佈比較 (聚焦 0-1000 Hz)')
            ax0.set_xlabel('頻率 (Hz)')
            ax0.legend()
            ax0.grid(True, which="both", ls="--", alpha=0.5)
            ax0.set_xlim(0, 1000)

            # --- 下方圖：頻譜圖 (Spectrograms) ---
            ax1 = fig.add_subplot(gs[1, 0])
            D_bg = librosa.amplitude_to_db(np.abs(librosa.stft(y_bg)), ref=np.max)
            img1 = librosa.display.specshow(D_bg, sr=sr_bg, x_axis='time', y_axis='log', ax=ax1, cmap='jet')
            fig.colorbar(img1, ax=ax1, format='%+2.0f dB', label='能量強度 (dB)')
            ax1.set_title('背景音頻譜圖 (Spectrogram)')
            ax1.set_xlabel('時間 (秒)')
            ax1.set_ylabel('頻率 (Hz)')

            ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
            D_test = librosa.amplitude_to_db(np.abs(librosa.stft(y_test)), ref=np.max)
            img2 = librosa.display.specshow(D_test, sr=sr_test, x_axis='time', y_axis='log', ax=ax2, cmap='jet')
            fig.colorbar(img2, ax=ax2, format='%+2.0f dB', label='能量強度 (dB)')
            ax2.set_title('待測音頻譜圖 (Spectrogram)')
            ax2.set_xlabel('時間 (秒)')
            plt.setp(ax2.get_yticklabels(), visible=False)
            ax2.set_ylabel("")

            # 分析結果文字
            analysis_text = (
                f"頻率分析結果:\n"
                f"--------------------\n"
                f"背景音主頻率: {peak_bg:.2f} Hz\n"
                f"待測音主頻率: {peak_test:.2f} Hz"
            )
            fig.text(0.02, 0.9, analysis_text,
                     transform=fig.transFigure, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            return fig

        # --- 2. 循環產生、儲存並顯示兩種圖表 ---
        for scale_type in ['db', 'log']:
            print(f"\n--- 正在處理 '{scale_type.upper()}' 刻度圖表 ---")

            # 產生圖表
            fig = _generate_plot(scale_type)

            # 如果需要，儲存圖表
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                bg_name = os.path.splitext(os.path.basename(background_path))[0]
                test_name = os.path.splitext(os.path.basename(test_path))[0]
                # 在檔名中加入刻度類型
                output_filename = f"{bg_name}_vs_{test_name}_{scale_type}.png"
                output_filepath = os.path.join(output_dir, output_filename)
                fig.savefig(output_filepath, bbox_inches='tight', dpi=150)
                print(f"圖表已儲存至: {output_filepath}")

            # 如果需要，顯示圖表
            if not no_show:
                print("正在顯示圖表... (關閉視窗以繼續處理下一個)")
                plt.show()

            # 關閉圖表以釋放記憶體
            plt.close(fig)

    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案！請檢查路徑是否正確。 {e}")
    except Exception as e:
        print(f"處理音檔時發生錯誤: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="比較背景音和待測音的頻譜，並可選擇性儲存結果圖。\n此版本會同時產生 dB 和 Log 兩種刻度的圖表。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bg', type=str, required=True, help="背景音檔 (.wav) 的路徑。")
    parser.add_argument('--test', type=str, required=True, help="待測音檔 (.wav) 的路徑。")
    parser.add_argument('--outdir', type=str, help="儲存結果圖檔的資料夾路徑。")
    parser.add_argument('--no-show', action='store_true', help="僅儲存圖檔，不顯示互動視窗 (需搭配 --outdir 使用)。")

    args = parser.parse_args()

    analyze_and_compare(
        background_path=args.bg,
        test_path=args.test,
        output_dir=args.outdir,
        no_show=args.no_show
    )