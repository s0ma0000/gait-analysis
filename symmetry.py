import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation, FFMpegWriter
import plotly.graph_objects as go

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
SYMMETRY_OUTPUT_PATH = "dataset/symmetry_analysis.csv"
VIDEO_OUTPUT_PATH = "dataset/symmetry_animation.mp4"

# キーポイントインデックス
keypoints_indices = {
    'hip_left': 11,
    'knee_left': 13,
    'ankle_left': 15,
    'hip_right': 12,
    'knee_right': 14,
    'ankle_right': 16
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

# データの検証と欠損値の処理
for k in keypoints.keys():
    keypoints[k] = np.array(keypoints[k])
    keypoints[k] = pd.DataFrame(keypoints[k]).interpolate().values  # 線形補間による欠損値処理

# 対称性の指標を計算する関数
def calculate_symmetry_index(left_points, right_points):
    differences = []
    for left, right in zip(left_points, right_points):
        difference = np.linalg.norm(np.array(left) - np.array(right))
        differences.append(difference)
    return differences

# 角度の計算
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# 左右の膝および足首の対称性を計算
knee_symmetry = calculate_symmetry_index(keypoints['knee_left'], keypoints['knee_right'])
ankle_symmetry = calculate_symmetry_index(keypoints['ankle_left'], keypoints['ankle_right'])

# 角度の対称性を計算
knee_angles_left = [calculate_angle(np.array(keypoints['ankle_left'][i]), np.array(keypoints['knee_left'][i]), np.array(keypoints['hip_left'][i])) for i in range(len(frames))]
knee_angles_right = [calculate_angle(np.array(keypoints['ankle_right'][i]), np.array(keypoints['knee_right'][i]), np.array(keypoints['hip_right'][i])) for i in range(len(frames))]
angle_symmetry = [abs(left - right) for left, right in zip(knee_angles_left, knee_angles_right)]

# データをフィルタリング
knee_symmetry_filtered = gaussian_filter1d(knee_symmetry, sigma=2)
ankle_symmetry_filtered = gaussian_filter1d(ankle_symmetry, sigma=2)
angle_symmetry_filtered = gaussian_filter1d(angle_symmetry, sigma=2)

# 対称性データを保存
symmetry_df = pd.DataFrame({
    'frame': frames,
    'knee_symmetry': knee_symmetry,
    'knee_symmetry_filtered': knee_symmetry_filtered,
    'ankle_symmetry': ankle_symmetry,
    'ankle_symmetry_filtered': ankle_symmetry_filtered,
    'angle_symmetry': angle_symmetry,
    'angle_symmetry_filtered': angle_symmetry_filtered
})
symmetry_df.to_csv(SYMMETRY_OUTPUT_PATH, index=False)

# プロットの作成
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# 膝の対称性のプロット
axs[0].plot(frames, knee_symmetry, label='Knee Symmetry')
axs[0].plot(frames, knee_symmetry_filtered, label='Filtered Knee Symmetry', linestyle='--')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Symmetry Index')
axs[0].set_title('Knee Symmetry Over Time')
axs[0].legend()
axs[0].grid(True)

# 足首の対称性のプロット
axs[1].plot(frames, ankle_symmetry, label='Ankle Symmetry')
axs[1].plot(frames, ankle_symmetry_filtered, label='Filtered Ankle Symmetry', linestyle='--')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Symmetry Index')
axs[1].set_title('Ankle Symmetry Over Time')
axs[1].legend()
axs[1].grid(True)

# 角度の対称性のプロット
axs[2].plot(frames, angle_symmetry, label='Angle Symmetry')
axs[2].plot(frames, angle_symmetry_filtered, label='Filtered Angle Symmetry', linestyle='--')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Symmetry Index')
axs[2].set_title('Angle Symmetry Over Time')
axs[2].legend()
axs[2].grid(True)

# アニメーションの設定
def update(frame):
    knee_sym_line.set_data(frames[:frame], knee_symmetry_filtered[:frame])
    ankle_sym_line.set_data(frames[:frame], ankle_symmetry_filtered[:frame])
    angle_sym_line.set_data(frames[:frame], angle_symmetry_filtered[:frame])
    return knee_sym_line, ankle_sym_line, angle_sym_line

knee_sym_line, = axs[0].plot([], [], 'b-')
ankle_sym_line, = axs[1].plot([], [], 'r-')
angle_sym_line, = axs[2].plot([], [], 'g-')

ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.tight_layout()
plt.show()

# インタラクティブなプロットを作成
fig = go.Figure()

fig.add_trace(go.Scatter(x=frames, y=knee_symmetry_filtered, mode='lines', name='Filtered Knee Symmetry'))
fig.add_trace(go.Scatter(x=frames, y=ankle_symmetry_filtered, mode='lines', name='Filtered Ankle Symmetry'))
fig.add_trace(go.Scatter(x=frames, y=angle_symmetry_filtered, mode='lines', name='Filtered Angle Symmetry'))

fig.update_layout(
    title="Symmetry Analysis Over Time",
    xaxis_title="Frame",
    yaxis_title="Symmetry Index",
    legend_title="Symmetry Type"
)

fig.show()
