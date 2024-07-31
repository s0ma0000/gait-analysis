#股関節の角度を解析します。データを読み込み、股関節の角度を計算し、解析結果を保存します。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter1d

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
CSV_OUTPUT_PATH = "dataset/hip_angles.csv"
VIDEO_OUTPUT_PATH = "dataset/hip_angles_animation.mp4"

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
hip_angles_left = []
hip_angles_right = []
keypoints = {k: [] for k in keypoints_indices.keys()}
frames = []

for frame, group in df.groupby('frame'):
    frames.append(frame)
    try:
        for k, idx in keypoints_indices.items():
            point = group[group['keypoint'] == idx][['x', 'y']].values[0]
            keypoints[k].append(point)
        hip_angle_left = calculate_angle(keypoints['knee_left'][-1], keypoints['hip_left'][-1], keypoints['ankle_left'][-1])
        hip_angle_right = calculate_angle(keypoints['knee_right'][-1], keypoints['hip_right'][-1], keypoints['ankle_right'][-1])
        hip_angles_left.append(hip_angle_left)
        hip_angles_right.append(hip_angle_right)
    except IndexError:
        hip_angles_left.append(np.nan)  # キーポイントが欠落している場合
        hip_angles_right.append(np.nan)  # キーポイントが欠落している場合

# 角度データのフィルタリング
hip_angles_left_filtered = gaussian_filter1d(hip_angles_left, sigma=2)
hip_angles_right_filtered = gaussian_filter1d(hip_angles_right, sigma=2)

# 角度データをCSVに保存
hip_angles_df = pd.DataFrame({
    'frame': frames,
    'hip_angle_left': hip_angles_left_filtered,
    'hip_angle_right': hip_angles_right_filtered
})
hip_angles_df.to_csv(CSV_OUTPUT_PATH, index=False)

# プロットの作成
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# アニメーションの設定
lines = {k: axs[0].plot([], [], marker='o', linestyle='-', label=k.replace('_', ' ').capitalize())[0] for k in keypoints.keys()}
lines_segments = {
    'left_hip-knee': axs[0].plot([], [], 'b-')[0],
    'left_knee-ankle': axs[0].plot([], [], 'b-')[0],
    'right_hip-knee': axs[0].plot([], [], 'r-')[0],
    'right_knee-ankle': axs[0].plot([], [], 'r-')[0]
}

angle_lines = {
    'left_hip': axs[1].plot([], [], 'b-', label='Left Hip Angle')[0],
    'right_hip': axs[1].plot([], [], 'r-', label='Right Hip Angle')[0]
}

axs[0].set_xlim(0, 1500)
axs[0].set_ylim(0, 900)
axs[0].invert_yaxis()  # Y軸を反転して画像と一致させる
axs[0].set_title('Keypoints Movement Over Time')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlim(0, max(frames))
axs[1].set_ylim(min(min(hip_angles_left_filtered), min(hip_angles_right_filtered)), max(max(hip_angles_left_filtered), max(hip_angles_right_filtered)))
axs[1].set_title('Hip Flexion/Extension Angle Over Time')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Angle (degrees)')
axs[1].legend()
axs[1].grid(True)

# 更新関数
def update(frame):
    for k in keypoints.keys():
        if len(keypoints[k]) > frame:
            lines[k].set_data(keypoints[k][frame][0], keypoints[k][frame][1])
    
    if len(keypoints['hip_left']) > frame and len(keypoints['knee_left']) > frame and len(keypoints['ankle_left']) > frame:
        lines_segments['left_hip-knee'].set_data(
            [keypoints['hip_left'][frame][0], keypoints['knee_left'][frame][0]],
            [keypoints['hip_left'][frame][1], keypoints['knee_left'][frame][1]]
        )
        lines_segments['left_knee-ankle'].set_data(
            [keypoints['knee_left'][frame][0], keypoints['ankle_left'][frame][0]],
            [keypoints['knee_left'][frame][1], keypoints['ankle_left'][frame][1]]
        )
    
    if len(keypoints['hip_right']) > frame and len(keypoints['knee_right']) > frame and len(keypoints['ankle_right']) > frame:
        lines_segments['right_hip-knee'].set_data(
            [keypoints['hip_right'][frame][0], keypoints['knee_right'][frame][0]],
            [keypoints['hip_right'][frame][1], keypoints['knee_right'][frame][1]]
        )
        lines_segments['right_knee-ankle'].set_data(
            [keypoints['knee_right'][frame][0], keypoints['ankle_right'][frame][0]],
            [keypoints['knee_right'][frame][1], keypoints['ankle_right'][frame][1]]
        )
    
    angle_lines['left_hip'].set_data(frames[:frame], hip_angles_left_filtered[:frame])
    angle_lines['right_hip'].set_data(frames[:frame], hip_angles_right_filtered[:frame])

    return list(lines.values()) + list(lines_segments.values()) + list(angle_lines.values())

# アニメーションの作成と保存
ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.tight_layout()
plt.show()
