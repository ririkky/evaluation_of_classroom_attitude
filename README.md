# 授業態度評価システム

YOLOv7とMediaPipeを使用した、授業中の姿勢と集中度を自動評価するシステムです。写真や動画から顔を検出し、姿勢（Pitch/Yaw角度）と目の開き具合（EAR値）を分析して評価スコアを算出します。

## 主な機能

### 📷 画像評価
- 静止画像から顔を検出し、姿勢と集中度を評価
- 複数人の同時検出に対応
- 顔のランドマーク（キーポイント）を可視化

### 🎥 動画評価
- 動画を1秒ごとにフレーム分割して解析
- 各フレームの評価スコアをグラフ表示
- 時系列での姿勢変化を可視化
- 平均値・最大値・最小値の統計表示

### 📊 評価項目
1. **Pitchスコア（うなずき）**: 頭の前後の傾き（0-10点）
2. **Yawスコア（首振り）**: 頭の左右の傾き（0-10点）
3. **EARスコア（目の開き）**: Eye Aspect Ratio（0-10点）
4. **総合スコア**: 上記3項目の合計（0-30点）

## セットアップ

### 必要な環境
- Python 3.8以上
- CUDA対応GPU（推奨、CPU でも動作）

### インストール

1. リポジトリのクローン
```bash
git clone https://github.com/ririkky/evaluation_of_classroom_attitude.git
cd evaluation_of_classroom_attitude
```

2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

3. YOLOv7の重みファイルをダウンロード
```bash
cd detect_yolo
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## 使い方

### Webアプリケーションの起動

```bash
cd UI
python app3.py
```

ブラウザで `http://127.0.0.1:5001` にアクセスします。

### 画像のアップロードと評価

1. 「画像をアップロード」タブを選択
2. 評価したい画像ファイルを選択
3. 自動的に検出が実行され、結果が表示されます

**表示内容:**
- 検出された顔画像（ランドマーク付き）
- 各評価項目のスコアと検出値
- EAR値の推移グラフ

### 動画のアップロードと評価

1. 「動画をアップロード」タブを選択
2. 評価したい動画ファイルを選択
3. 動画が1秒ごとに分割され、各フレームが解析されます

**表示内容:**
- **グラフ表示**: 
  - 評価スコアの時系列グラフ
  - 各項目の平均値・最大値・最小値の統計表
- **写真表示**: 
  - 各フレームの顔画像（ランドマーク付き）
  - フレームごとのスコアと検出値

## プロジェクト構造

```
evaluation_of_classroom_attitude/
├── UI/
│   ├── app3.py                 # Flaskサーバー（メインアプリケーション）
│   └── templates/
│       └── index.html          # Webインターフェース
├── detect_yolo/
│   ├── detect.py               # YOLOv7 + MediaPipe 検出スクリプト
│   ├── yolov7.pt              # YOLOv7の重みファイル
│   └── images/
│       ├── faces/              # 検出された顔画像
│       └── results.csv         # 検出結果のCSV
└── requirements.txt            # Python依存パッケージ
```

## 技術スタック

- **物体検出**: YOLOv7
- **顔ランドマーク検出**: MediaPipe Face Mesh
- **Webフレームワーク**: Flask
- **フロントエンド**: HTML/CSS/JavaScript
- **画像処理**: OpenCV, NumPy

## スコアリング基準

### Pitch（うなずき）
- 基準角度との差分を元にスコア化
- 前傾姿勢（20度以上）を検出

### Yaw（首振り）
- 左右の角度を評価
- 正面を向いているほど高スコア

### EAR（目の開き）
- Eye Aspect Ratioが一定値以上で高スコア
- まばたきや目を閉じている状態を検出

## 注意事項

- 動画処理は時間がかかる場合があります（フレーム数による）
- 複数人が映っている場合、各人物が個別に評価されます
- 顔が検出されないフレームはスキップされます
- GPU使用を推奨（CPU でも動作しますが処理が遅くなります）

## ライセンス

このプロジェクトは研究・教育目的で使用してください。

## 参考

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [MediaPipe](https://google.github.io/mediapipe/)
