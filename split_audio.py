import os
import librosa
import soundfile as sf
import numpy as np

# --- 參數設定 ---

# 1. 來源資料夾：包含您原始音檔的根目錄
SOURCE_ROOT_DIR = 'raw_train'

# 2. 目標資料夾：用來儲存切割後音檔的根目錄 (如果不存在，程式會自動建立)
DEST_ROOT_DIR = f'{SOURCE_ROOT_DIR}_split'

# 3. 切割長度：每個音檔片段的長度（秒）
SEGMENT_SECONDS = 2.0

# 4. 最小長度比例：如果最後一段的長度小於 (切割長度 * 這個比例)，就捨棄它
#    例如，設為 0.5，代表最後一段如果短於 1 秒 (2 * 0.5)，就不儲存。
MIN_SEGMENT_RATIO = 0.5


def split_audio_files(source_root, dest_root, segment_len_s, min_len_ratio):
    """
    遍歷來源資料夾，切割音檔，並以相同結構儲存到目標資料夾。

    Args:
        source_root (str): 原始音檔的根目錄。
        dest_root (str): 儲存切割後檔案的目標根目錄。
        segment_len_s (float): 每個片段的目標長度（秒）。
        min_len_ratio (float): 決定是否保留最後一個片段的最小長度比例。
    """
    # 確保來源資料夾存在
    if not os.path.exists(source_root):
        print(f"錯誤：來源資料夾 '{source_root}' 不存在。請檢查路徑。")
        return

    # os.walk 會深度遍歷所有子資料夾
    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            # 只處理 .wav 檔案 (不分大小寫)
            if not filename.lower().endswith('.wav'):
                continue

            source_filepath = os.path.join(dirpath, filename)
            print(f"\n正在處理: {source_filepath}")

            try:
                # 載入音檔，sr=None 表示保留原始取樣率
                y, sr = librosa.load(source_filepath, sr=None)

                # 計算每個片段的樣本數
                segment_samples = int(segment_len_s * sr)
                min_segment_samples = int(segment_samples * min_len_ratio)

                # --- 建立對應的輸出路徑 ---
                # 取得相對於來源根目錄的路徑，以保留結構
                relative_path = os.path.relpath(dirpath, source_root)
                dest_subdir = os.path.join(dest_root, relative_path)

                # 建立目標子資料夾 (如果不存在)
                os.makedirs(dest_subdir, exist_ok=True)

                # --- 開始切割與儲存 ---
                num_segments_created = 0
                for i, start_sample in enumerate(range(0, len(y), segment_samples)):
                    end_sample = start_sample + segment_samples
                    segment = y[start_sample:end_sample]

                    # 如果最後一段太短，就捨棄
                    if len(segment) < min_segment_samples:
                        print(f"  - 最後一段長度不足 ({len(segment)} samples)，已捨棄。")
                        continue

                    # 建立新檔名
                    base_name, extension = os.path.splitext(filename)
                    new_filename = f"{base_name}_part{i + 1}{extension}"
                    dest_filepath = os.path.join(dest_subdir, new_filename)

                    # 使用 soundfile 儲存音檔片段
                    sf.write(dest_filepath, segment, sr)
                    num_segments_created += 1

                print(f"  -> 完成！共建立 {num_segments_created} 個片段。")

            except Exception as e:
                print(f"  處理檔案 {source_filepath} 時發生錯誤: {e}")

    print("\n--- 所有檔案處理完畢 ---")


if __name__ == '__main__':
    # 執行主函式
    split_audio_files(
        source_root=SOURCE_ROOT_DIR,
        dest_root=DEST_ROOT_DIR,
        segment_len_s=SEGMENT_SECONDS,
        min_len_ratio=MIN_SEGMENT_RATIO
    )