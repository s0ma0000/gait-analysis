#ステップ数を解析します。歩行サイクルを分析し、各ステップの長さや歩幅を計算します。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter1d

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
SPEED_OUTPUT_PATH = "dataset/walking_speed.csv"
VIDEO_OUTPUT_PATH = "dataset/walking_speed_animation.mp4"

# キーポイントインデックス（例として左足首を使用）
keypoints_indices = {
    'ankle_left': 15
}

# キーポイントデータの読み込み
df = pd.read_csv(CSV_INPUT_PATH)

# フレームごとにキーポイントインデックスを追加
num_keypoints = 17  # 1フレームあたりのキーポイント数
df['keypoint'] = np.tile(np.arange(num_keypoints), len(df) // num_keypoints)

# キーポイントデータをフレームごとに処理
keypoints = {k: [] for k in keypoints_indices.keys()}
frames = []

for frame, group in df.groupby('frame'):
    frames.append(frame)
    try:
        for k, idx in keypoints_indices.items():
            point = group[group['keypoint'] == idx][['x', 'y']].values[0]
            keypoints[k].append(point)
    except IndexError:
        keypoints[k].append([np.nan, np.nan])  # キーポイントが欠落している場合

# 歩行速度の計算
speeds = []
time_intervals = np.diff(frames)
for i in range(1, len(frames)):
    if np.isnan(keypoints['ankle_left'][i]).any() or np.isnan(keypoints['ankle_left'][i-1]).any():
        speeds.append(np.nan)
    else:
        distance = np.linalg.norm(np.array(keypoints['ankle_left'][i]) - np.array(keypoints['ankle_left'][i-1]))
        speed = distance / time_intervals[i-1]
        speeds.append(speed)

# スピードデータをフィルタリング
speeds_filtered = gaussian_filter1d(speeds, sigma=2)

# 歩数の計算
step_threshold = 5  # スピードの閾値（この値を調整）
steps = np.where(np.diff(np.sign(speeds_filtered - step_threshold)) > 0)[0]
num_steps = len(steps)

# スピードデータをCSVに保存
speeds_df = pd.DataFrame({
    'frame': frames[1:],
    'speed': speeds,
    'speed_filtered': speeds_filtered,
    'steps': np.isin(np.arange(len(speeds)), steps).astype(int)
})
speeds_df.to_csv(SPEED_OUTPUT_PATH, index=False)

# プロットの作成
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# スピードのプロット
axs[0].plot(frames[1:], speeds, label='Walking Speed')
axs[0].plot(frames[1:], speeds_filtered, label='Filtered Walking Speed', linestyle='--')

# stepsのインデックスを整数配列として扱う
steps_frames = np.array(frames)[steps + 1]
axs[0].scatter(steps_frames, speeds_filtered[steps], color='red', label='Steps')

axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Speed (pixels/frame)')
axs[0].set_title('Walking Speed Over Time')
axs[0].legend()
axs[0].grid(True)

# スピードのヒストグラムプロット
axs[1].hist(speeds, bins=30, alpha=0.5, label='Walking Speed')
axs[1].hist(speeds_filtered, bins=30, alpha=0.5, label='Filtered Walking Speed')
axs[1].set_xlabel('Speed (pixels/frame)')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distribution of Walking Speed')
axs[1].legend()
axs[1].grid(True)

# 歩数の表示
text = axs[2].text(0.5, 0.5, f'Total Steps: {num_steps}', horizontalalignment='center', verticalalignment='center', fontsize=20)
axs[2].axis('off')

# アニメーションの更新関数
def update(frame):
    scat.set_offsets([keypoints['ankle_left'][frame]])
    speed_line.set_data(frames[1:frame+1], speeds_filtered[:frame])
    step_points = np.array([[frames[s + 1], speeds_filtered[s]] for s in steps if s <= frame])
    if len(step_points) > 0:
        step_scat.set_offsets(step_points)
    else:
        step_scat.set_offsets(np.empty((0, 2)))

    # 更新された歩数の表示
    current_steps = np.sum(np.isin(steps, np.arange(frame)))
    text.set_text(f'Total Steps: {current_steps}')
    
    return scat, speed_line, step_scat, text

# アニメーションの作成
scat = axs[0].scatter([], [], color='blue')
speed_line = axs[0].plot([], [], 'b-')[0]
step_scat = axs[0].scatter([], [], color='red')

ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.tight_layout()
plt.show()
