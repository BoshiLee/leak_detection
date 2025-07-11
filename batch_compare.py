import os
import argparse
import subprocess
import sys


def batch_process(bg_file, test_root, output_root):
    """
    遍歷測試資料夾，將其中每個音檔與指定的背景音檔進行比較，
    並將結果圖儲存到結構對應的輸出資料夾中。

    Args:
        bg_file (str): 單一的、作為基準的背景音檔路徑。
        test_root (str): 包含待測音檔的根目錄。
        output_root (str): 儲存比較結果圖的根目錄。
    """
    # 1. 驗證輸入路徑是否存在
    if not os.path.isfile(bg_file):
        print(f"錯誤：背景音檔 '{bg_file}' 不存在或不是一個檔案。")
        return

    if not os.path.isdir(test_root):
        print(f"錯誤：測試資料夾 '{test_root}' 不存在或不是一個目錄。")
        return

    print(f"基準背景音: {bg_file}")
    print(f"掃描測試資料夾: {test_root}")
    print(f"輸出至: {output_root}")
    print("-" * 30)

    # 找到 compare_frequencies.py 的路徑
    # 假設批次腳本和目標腳本在同一個目錄下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(script_dir, 'compare_frequencies.py')

    if not os.path.isfile(target_script):
        print(f"錯誤: 找不到分析腳本 '{target_script}'。請確保此腳本與 compare_frequencies.py 在同一個資料夾中。")
        return

    # 2. 遍歷測試資料夾
    file_count = 0
    for dirpath, _, filenames in os.walk(test_root):
        for filename in filenames:
            # 只處理 .wav 檔案
            if not filename.lower().endswith('.wav'):
                continue

            file_count += 1
            test_filepath = os.path.join(dirpath, filename)
            print(f"[{file_count}] 正在處理: {test_filepath}")

            # 3. 建立對應的輸出路徑
            # 取得相對於測試根目錄的路徑，以保留結構
            relative_path = os.path.relpath(dirpath, test_root)
            # 如果 relative_path 是 '.', 代表在根目錄，os.path.join 會正確處理
            output_subdir = os.path.join(output_root, relative_path)

            # 4. 準備並執行子程序命令
            # 使用 sys.executable 確保用的是當前的 Python 解譯器
            command = [
                sys.executable,
                target_script,
                '--bg', bg_file,
                '--test', test_filepath,
                '--outdir', output_subdir,
                '--no-show'  # 在批次處理中，不彈出視窗，僅儲存檔案
            ]

            try:
                # 執行命令，並捕獲輸出
                result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                # 如果需要偵錯，可以取消下面這行的註解來查看子腳本的輸出
                # print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"  處理失敗! '{filename}' 發生錯誤:")
                # 印出子腳本的錯誤訊息，方便排查問題
                print(e.stderr)
            except Exception as e:
                print(f"  執行子程序時發生未知錯誤: {e}")

    print("-" * 30)
    print(f"批次處理完成！共處理了 {file_count} 個檔案。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="批次執行 compare_frequencies.py，將指定資料夾中的所有音檔與一個基準背景音比較。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bg_file', type=str, required=True,
                        help="作為基準的單一背景音檔 (.wav) 路徑。")
    parser.add_argument('--test_root', type=str, required=True,
                        help="包含所有待測音檔的根目錄。")
    parser.add_argument('--output_root', type=str, required=True,
                        help="儲存所有比較結果圖的根目錄。")

    args = parser.parse_args()

    batch_process(
        bg_file=args.bg_file,
        test_root=args.test_root,
        output_root=args.output_root
    )