## モデルの初期化
#MODEL_POSE = "model/yolov8n-pose.pt"
#model_pose = YOLO(MODEL_POSE)
#
## ビデオのパス
#SOURCE_VIDEO_PATH = "dataset/p01.mov"
#TARGET_VIDEO_PATH = "dataset/output_p01.mov"


#全体のプログラムのエントリーポイント.各モジュールを統合して，歩行データの解析と視覚化を実行


import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# モデルの初期化
MODEL_POSE = "model/yolov8n-pose.pt"
model_pose = YOLO(MODEL_POSE)

# ビデオのパス
SOURCE_VIDEO_PATH = "dataset/taku.mov"
TARGET_VIDEO_PATH = "dataset/output_taku.mp4"
CSV_OUTPUT_PATH = "dataset/keypoints.csv"  # CSVファイルの出力パスを設定

# ビデオの読み込み
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ビデオライターの設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (width, height))

# キーポイントデータを保存するリスト
keypoints_data = []

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 姿勢推定
    results = model_pose(frame)
    
    # キーポイントの描画およびデータ収集
    for result in results:
        keypoints = result.keypoints.xy[0]  # keypoints.xy はテンソル内の最初のエントリを取得
        frame_keypoints = []
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            frame_keypoints.append((frame_index, x.item(), y.item()))  # フレームインデックスと座標を保存
        keypoints_data.extend(frame_keypoints)
    
    # フレームを表示
    cv2.imshow('Pose Estimation', frame)
    
    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # フレームをファイルに書き込む
    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# キーポイントデータをCSVに保存
df = pd.DataFrame(keypoints_data, columns=['frame', 'x', 'y'])
df.to_csv(CSV_OUTPUT_PATH, index=False)
