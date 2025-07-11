# spectral_subtraction.py

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def analyze_spectral_subtraction(background_path, test_path, output_file=None):
    """
    透過頻譜減法，分析並視覺化待測音去除背景影響後的頻譜。

    Args:
        background_path (str): 背景音檔的路徑。
        test_path (str): 待測音檔的路徑。
        output_file (str, optional): 儲存結果圖檔的路徑. Defaults to None.
    """
    try:
        # --- 1. 載入與預處理 ---
        y_bg, sr_bg = librosa.load(background_path, sr=None)
        y_test, sr_test = librosa.load(test_path, sr=None)

        if sr_test != sr_bg:
            y_test = librosa.resample(y_test, orig_sr=sr_test, target_sr=sr_bg)
        sr = sr_bg  # 使用統一的取樣率

        # 為了讓 FFT 結果可比較，將較短的音檔補零至與較長音檔相同長度
        if len(y_bg) > len(y_test):
            y_test = np.pad(y_test, (0, len(y_bg) - len(y_test)))
        elif len(y_test) > len(y_bg):
            y_bg = np.pad(y_bg, (0, len(y_test) - len(y_bg)))

        n_fft = len(y_bg)  # FFT 點數
        freqs = np.fft.rfftfreq(n_fft, d=1 / sr)  # 取得實數 FFT 的頻率軸

        # --- 2. 計算頻譜 ---
        mag_bg = np.abs(np.fft.rfft(y_bg))
        mag_test = np.abs(np.fft.rfft(y_test))

        # --- 3. 執行頻譜減法 ---
        # 從待測音頻譜中減去背景音頻譜，並確保結果不為負
        mag_subtracted = np.maximum(0, mag_test - mag_bg)

        # --- 4. 視覺化 ---
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle('頻譜減法分析', fontsize=20)

        # 上方圖: 原始頻譜比較
        axes[0].set_title('原始頻譜比較')
        axes[0].plot(freqs, mag_bg, color='blue', alpha=0.6, label='背景音頻譜')
        axes[0].plot(freqs, mag_test, color='red', alpha=0.6, label='待測音頻譜 (含背景)')
        axes[0].set_ylabel('線性幅度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.5)

        # 下方圖: 減去背景後的純淨頻譜
        axes[1].set_title('純淨信號頻譜 (估算)')
        axes[1].plot(freqs, mag_subtracted, color='green', label='純淨信號頻譜 (待測音 - 背景音)')
        axes[1].set_xlabel('頻率 (Hz)')
        axes[1].set_ylabel('線性幅度')
        axes[1].legend()
        axes[1].grid(True, alpha=0.5)

        # 根據您的需求，可以設定觀察的頻率範圍
        focus_freq_limit = 1000
        plt.xlim(0, focus_freq_limit)
        # 如果想看全頻段，可以註解掉上面兩行

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            print(f"\n頻譜減法分析圖已儲存至: {output_file}")

        plt.show()
        plt.close(fig)

    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案！請檢查路徑是否正確。 {e}")
    except Exception as e:
        print(f"處理音檔時發生錯誤: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="透過頻譜減法，分析並視覺化待測音去除背景影響後的頻譜。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bg', type=str, required=True, help="作為基準的單一背景音檔 (.wav) 路徑。")
    parser.add_argument('--test', type=str, required=True, help="待測音檔 (.wav) 的路徑。")
    parser.add_argument('--outfile', type=str, help="儲存結果圖檔的路徑 (例如: ./results/subtracted.png)。")

    args = parser.parse_args()

    analyze_spectral_subtraction(
        background_path=args.bg,
        test_path=args.test,
        output_file=args.outfile
    )