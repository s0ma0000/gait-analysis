#関節の角度データを用いてヒートマップを作成します。関節の動きを視覚的に解析するために使用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation, FFMpegWriter

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
HEATMAP_OUTPUT_PATH = "dataset/joint_angle_heatmap.png"
VIDEO_OUTPUT_PATH = "dataset/joint_angle_animation.mp4"

# キーポイントインデックス
keypoints_indices = {
    'hip_left': 11,
    'knee_left': 13,
    'ankle_left': 15,
    'hip_right': 12,
    'knee_right': 14,
    'ankle_right': 16
}

# フレームごとの角度を計算する関数
def calculate_angle(point1, point2, point3):
    v1 = point1 - point2
    v2 = point3 - point2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# キーポイントデータの読み込み
df = pd.read_csv(CSV_INPUT_PATH)

# フレームごとにキーポイントインデックスを追加
num_keypoints = 17  # 1フレームあたりのキーポイント数
df['keypoint'] = np.tile(np.arange(num_keypoints), len(df) // num_keypoints)

# キーポイントデータをフレームごとに処理
angles = []
keypoints = {k: [] for k in keypoints_indices.keys()}
frames = []

for frame, group in df.groupby('frame'):
    frames.append(frame)
    try:
        for k, idx in keypoints_indices.items():
            point = group[group['keypoint'] == idx][['x', 'y']].values[0]
            keypoints[k].append(point)
        
        angle_left_knee = calculate_angle(keypoints['hip_left'][-1], keypoints['knee_left'][-1], keypoints['ankle_left'][-1])
        angle_right_knee = calculate_angle(keypoints['hip_right'][-1], keypoints['knee_right'][-1], keypoints['ankle_right'][-1])
        
        angle_left_hip = calculate_angle(keypoints['knee_left'][-1], keypoints['hip_left'][-1], keypoints['ankle_left'][-1])
        angle_right_hip = calculate_angle(keypoints['knee_right'][-1], keypoints['hip_right'][-1], keypoints['ankle_right'][-1])
        
        angle_left_ankle = calculate_angle(keypoints['knee_left'][-1], keypoints['ankle_left'][-1], keypoints['hip_left'][-1])
        angle_right_ankle = calculate_angle(keypoints['knee_right'][-1], keypoints['ankle_right'][-1], keypoints['hip_right'][-1])
        
        angles.append([angle_left_knee, angle_right_knee, angle_left_hip, angle_right_hip, angle_left_ankle, angle_right_ankle])
    except IndexError:
        angles.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  # キーポイントが欠落している場合

# 角度データをデータフレームに変換
angles_df = pd.DataFrame(angles, columns=['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle', 'left_ankle_angle', 'right_ankle_angle'])

# 欠落データの補間
angles_df = angles_df.interpolate(method='linear', axis=0).ffill().bfill()

# データのフィルタリング
angles_df_filtered = angles_df.apply(lambda x: gaussian_filter1d(x, sigma=2))

# ヒートマップのプロット
plt.figure(figsize=(12, 8))
sns.heatmap(angles_df_filtered.T, cmap='coolwarm', cbar=True, annot=False)
plt.xlabel('Frame')
plt.ylabel('Joint')
plt.title('Joint Angle Heatmap Over Time')
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip', 'Left Ankle', 'Right Ankle'])
plt.savefig(HEATMAP_OUTPUT_PATH)
plt.show()

# 角度変化のアニメーションプロット
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# アニメーションの設定
lines = {col: axs[i//2].plot([], [], label=col)[0] for i, col in enumerate(angles_df_filtered.columns)}

axs[0].set_xlim(0, max(frames))
axs[0].set_ylim(min(angles_df_filtered.min()), max(angles_df_filtered.max()))
axs[0].set_title('Knee Angles Over Time')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Angle (degrees)')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlim(0, max(frames))
axs[1].set_ylim(min(angles_df_filtered.min()), max(angles_df_filtered.max()))
axs[1].set_title('Hip Angles Over Time')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Angle (degrees)')
axs[1].legend()
axs[1].grid(True)

axs[2].set_xlim(0, max(frames))
axs[2].set_ylim(min(angles_df_filtered.min()), max(angles_df_filtered.max()))
axs[2].set_title('Ankle Angles Over Time')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Angle (degrees)')
axs[2].legend()
axs[2].grid(True)

# 更新関数
def update(frame):
    for col in angles_df_filtered.columns:
        lines[col].set_data(frames[:frame], angles_df_filtered[col][:frame])
    return list(lines.values())

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.tight_layout()
plt.show()
