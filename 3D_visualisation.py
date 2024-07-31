#キーポイントデータを用いて歩行の3Dビジュアル化を行います。CSVファイルからデータを読み込み、欠損値を補間し、フィルタリングした後、3Dプロットを作成してアニメーション化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation, FFMpegWriter

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
THREED_OUTPUT_PATH = "dataset/3d_gait_visualization.png"
VIDEO_OUTPUT_PATH = "dataset/3d_gait_animation.mp4"

# キーポイントデータの読み込み
df = pd.read_csv(CSV_INPUT_PATH)

# フレームごとにキーポイントインデックスを追加
num_keypoints = 17  # 1フレームあたりのキーポイント数
df['keypoint'] = np.tile(np.arange(num_keypoints), len(df) // num_keypoints)

# キーポイントデータをフレームごとに処理
keypoints = []
frames = []

for frame, group in df.groupby('frame'):
    frames.append(frame)
    points = []
    try:
        for i in range(num_keypoints):
            point = group[group['keypoint'] == i][['x', 'y']].values[0]
            points.extend(point)
    except IndexError:
        points.extend([np.nan] * 34)  # キーポイントが欠落している場合
    keypoints.append(points)

keypoints = np.array(keypoints)

# 欠落データの補間
keypoints_df = pd.DataFrame(keypoints)
keypoints_df = keypoints_df.interpolate(method='linear', axis=0).ffill().bfill()
keypoints = keypoints_df.values

# データのフィルタリング
keypoints_filtered = gaussian_filter1d(keypoints, sigma=2, axis=0)

# 3Dプロットの作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# プロットの初期化
scat = ax.scatter([], [], [], c='b', marker='o')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frame')
ax.set_title('3D Gait Visualization')

# 軸のスケーリングを設定
ax.set_xlim(np.nanmin(keypoints_filtered[:, ::2]), np.nanmax(keypoints_filtered[:, ::2]))
ax.set_ylim(np.nanmin(keypoints_filtered[:, 1::2]), np.nanmax(keypoints_filtered[:, 1::2]))
ax.set_zlim(0, len(frames))

# 視点の設定
ax.view_init(elev=20., azim=120)

# アニメーションの更新関数
def update(frame):
    scat._offsets3d = (keypoints_filtered[frame, ::2], keypoints_filtered[frame, 1::2], np.full(num_keypoints, frames[frame]))
    return scat,

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.savefig(THREED_OUTPUT_PATH)
plt.show()
