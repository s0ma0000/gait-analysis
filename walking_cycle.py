#歩行サイクルの解析を行います。歩行のフェーズを識別し、歩行パターンを解析します。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# CSVファイルのパス
CSV_INPUT_PATH = "dataset/keypoints.csv"
STEP_OUTPUT_PATH = "dataset/step_analysis.csv"
VIDEO_OUTPUT_PATH = "dataset/step_analysis_animation.mp4"

# キーポイントインデックス
keypoints_indices = {
    'ankle_left': 15,
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
        continue  # キーポイントが欠落している場合

# ステップ長と歩幅の計測
steps_left = []
steps_right = []
stride_lengths = []
walking_phase = []
last_left_foot = None
last_right_foot = None

for i in range(1, len(frames)):
    if keypoints['ankle_left'][i] is not None and keypoints['ankle_left'][i-1] is not None:
        step_length_left = np.linalg.norm(keypoints['ankle_left'][i] - keypoints['ankle_left'][i-1])
        steps_left.append(step_length_left)
    else:
        steps_left.append(0)
        
    if keypoints['ankle_right'][i] is not None and keypoints['ankle_right'][i-1] is not None:
        step_length_right = np.linalg.norm(keypoints['ankle_right'][i] - keypoints['ankle_right'][i-1])
        steps_right.append(step_length_right)
    else:
        steps_right.append(0)
    
    if keypoints['ankle_left'][i] is not None and keypoints['ankle_right'][i] is not None:
        stride_length = np.linalg.norm(keypoints['ankle_left'][i] - keypoints['ankle_right'][i])
        stride_lengths.append(stride_length)
    else:
        stride_lengths.append(0)
    
    if keypoints['ankle_left'][i][1] < keypoints['ankle_right'][i][1]:  # 左足が右足よりも上にある（接地）
        walking_phase.append('left')
        if last_left_foot is not None:
            stride_length = np.linalg.norm(keypoints['ankle_left'][i] - last_left_foot)
            last_left_foot = keypoints['ankle_left'][i]
        else:
            last_left_foot = keypoints['ankle_left'][i]
    else:  # 右足が接地
        walking_phase.append('right')
        if last_right_foot is not None:
            stride_length = np.linalg.norm(keypoints['ankle_right'][i] - last_right_foot)
            last_right_foot = keypoints['ankle_right'][i]
        else:
            last_right_foot = keypoints['ankle_right'][i]

# 歩行解析結果をCSVに保存
step_analysis_df = pd.DataFrame({
    'frame': frames[1:],
    'step_length_left': steps_left,
    'step_length_right': steps_right,
    'stride_length': stride_lengths,
    'walking_phase': walking_phase
})
step_analysis_df.to_csv(STEP_OUTPUT_PATH, index=False)

# プロットの作成
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# アニメーションの設定
line_step_left, = axs[0].plot([], [], label='Left Step Length', color='b')
line_step_right, = axs[0].plot([], [], label='Right Step Length', color='r')
line_stride, = axs[1].plot([], [], label='Stride Length', color='g')
line_phase, = axs[2].plot([], [], label='Walking Phase', color='purple')

axs[0].set_xlim(0, max(frames))
axs[0].set_ylim(0, max(max(steps_left), max(steps_right)))
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Step Length (pixels)')
axs[0].set_title('Step Length Over Time')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlim(0, max(frames))
axs[1].set_ylim(0, max(stride_lengths))
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Stride Length (pixels)')
axs[1].set_title('Stride Length Over Time')
axs[1].legend()
axs[1].grid(True)

axs[2].set_xlim(0, max(frames))
axs[2].set_ylim(-0.5, 1.5)
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Phase (1: Left, 0: Right)')
axs[2].set_title('Walking Phase Over Time')
axs[2].legend()
axs[2].grid(True)

# 更新関数
def update(frame):
    line_step_left.set_data(frames[:frame], steps_left[:frame])
    line_step_right.set_data(frames[:frame], steps_right[:frame])
    line_stride.set_data(frames[:frame], stride_lengths[:frame])
    line_phase.set_data(frames[:frame], [1 if phase == 'left' else 0 for phase in walking_phase[:frame]])
    return line_step_left, line_step_right, line_stride, line_phase

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50)

# FFMpegWriterを使用してMP4ファイルとして保存
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(VIDEO_OUTPUT_PATH, writer=writer)

plt.tight_layout()
plt.show()
