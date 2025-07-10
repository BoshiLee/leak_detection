import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 參數設定 ---
SAMPLE_DURATION = 2
SR = 22050
N_MFCC = 13

# 設定 Matplotlib 使用支援中文的字體，並解決負號顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']  # 優先使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False


def extract_features(file_path):
    """從音頻檔案中提取 MFCC 特徵"""
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=SAMPLE_DURATION)
        if len(y) < SAMPLE_DURATION * sr:
            y = np.pad(y, (0, SAMPLE_DURATION * sr - len(y)))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"處理檔案 {file_path} 時出錯: {e}")
        return None


def load_features_from_directory(directory, label, features_list, labels_list):
    """從指定目錄載入所有 .wav 檔案的特徵"""
    print(f"正在從 '{directory}' 載入資料，標籤為: {label}")
    if not os.path.exists(directory):
        print(f"警告：目錄 '{directory}' 不存在。")
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            filepath = os.path.join(directory, filename)
            features = extract_features(filepath)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)


# --- PCA 特徵空間分析 ---
print("\n--- 執行 PCA 分析 ---")

all_features = []
labels = []

# 定義類別與對應的目錄和標籤
# 標籤 0: 非洩漏 (正常)
# 標籤 1: 洩漏
data_map = {
    'training_data/no-leak/': 0,
    'training_data/Leak/': 1
}

for directory, label in data_map.items():
    load_features_from_directory(directory, label, all_features, labels)

if len(all_features) < 3:
    print("錯誤：需要至少三個音檔才能進行 3D PCA 分析。請在資料夾中放入足夠的音檔。")
else:
    # 將特徵列表和標籤列表轉換為 numpy 陣列
    X = np.array(all_features)
    y_labels = np.array(labels)

    # 初始化 PCA，將維度降到 3 維
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X)

    # --- 視覺化：產生三個 2D 投影視圖 (XY, YZ, XZ) ---

    # 建立一個 1x3 的子圖畫布
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('PCA 三視圖投影', fontsize=20)

    # 定義視圖的軸索引和名稱
    views = [
        (0, 1, 'XY 視圖 (PC1 vs PC2)'),
        (1, 2, 'YZ 視圖 (PC2 vs PC3)'),
        (0, 2, 'XZ 視圖 (PC1 vs PC3)')
    ]

    target_names = ['非洩漏音 (No-Leak)', '氣體洩漏音 (Leak)']
    colors = ['blue', 'red']

    # 循環繪製三個視圖
    for i, (x_idx, y_idx, title) in enumerate(views):
        ax = axes[i]
        for label_idx, target_name in enumerate(target_names):
            # 根據標籤篩選出對應的點
            ax.scatter(principal_components[y_labels == label_idx, x_idx],
                       principal_components[y_labels == label_idx, y_idx],
                       c=colors[label_idx],
                       label=target_name,
                       alpha=0.7,
                       s=50)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f'主成分 {x_idx + 1}', fontsize=12)
        ax.set_ylabel(f'主成分 {y_idx + 1}', fontsize=12)
        ax.legend()
        ax.grid(True)

    # 自動調整子圖間距，避免標籤重疊
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 調整佈局以容納主標題
    plt.show()

    print(f"PCA 解釋的變異數比例: {pca.explained_variance_ratio_}")
    print(f"總解釋變異數 (前三個主成分): {sum(pca.explained_variance_ratio_):.2f}")
