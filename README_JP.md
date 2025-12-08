# 授業態度評価ビューア - 実行ガイド

## 概要
このプロジェクトは、YOLOv7 + MediaPipe を使用して、顔画像から以下を自動検出・評価します：

- **顔の向き角度** (Pitch / Yaw)
- **目の開き具合** (EAR: Eye Aspect Ratio)
- **総合スコア** (100点満点)
- **評価ランク** (良好 / 普通 / 不良)

## 🚀 クイックスタート

### 方法 1: スタートアップスクリプト（推奨）

```bash
./start_app.sh
```

または、以下のコマンドで直接実行：

```bash
cd UI
python app3.py
```

Flask サーバーが起動したら、ブラウザで以下にアクセス：

```
http://localhost:5000
```

### 方法 2: コマンドラインで直接実行

顔画像を検出して評価：

```bash
cd detect_yolo
python detect.py --weights yolov7.pt --source input/your_image.jpg --save-faces
```

## 📊 ウェブインターフェース

### 機能
1. **画像アップロード** - ファイルを選択してアップロード
2. **自動検出** - 「検出実行」ボタンで即座に検出開始
3. **結果表示** - 2つのビューモード：
   - **テーブルビュー** - 詳細なスコア情報を一覧表示
   - **フォトビュー** - 検出された顔画像をギャラリー表示

### スコアシステム

| 項目 | 配点 | 評価基準 |
|------|------|----------|
| **Pitch角度** | 30点 | ≤20° = 30点、≥45° = 0点、線形計算 |
| **Yaw角度** | 30点 | ≤20° = 30点、≥45° = 0点、線形計算 |
| **EAR（目開き）** | 40点 | ≥0.25 = 40点、<0.15 = 0点、線形計算 |
| **総合スコア** | 100点 | Pitch + Yaw + EAR |

### 評価ランク

- **良好** ✅ - 70点以上
- **普通** 📊 - 50～69点
- **不良** ❌ - 50点未満

## 📁 ディレクトリ構造

```
evaluate_of_classroom_attitude/
├── UI/
│   ├── app3.py                 # Flask メインアプリケーション
│   ├── templates/
│   │   └── index.html          # ウェブインターフェース
│   ├── static/
│   │   └── script.js
│   └── uploads/
│
├── detect_yolo/
│   ├── detect.py               # YOLOv7 + MediaPipe 検出スクリプト
│   ├── yolov7.pt               # YOLOv7 モデルウェイト（事前学習済み）
│   ├── models/                 # YOLOv7 モデル定義
│   ├── utils/                  # ユーティリティ関数
│   ├── images/
│   │   ├── results.csv         # 検出結果（CSV）
│   │   └── faces/              # 検出された顔画像（自動保存）
│   ├── input/                  # 入力画像ディレクトリ
│   └── temp_input/             # 一時入力（自動管理）
│
├── start_app.sh                # スタートアップスクリプト
├── main.py                     # 従来の実行方式（オプション）
└── requirements.txt            # Python 依存ライブラリ

```

## ⚙️ 依存関係

```bash
pip install -r requirements.txt
```

主要なライブラリ：
- **torch** - PyTorch（YOLOv7推論用）
- **opencv-python** - 画像処理
- **mediapipe** - 顔のランドマーク検出
- **Flask** - ウェブサーバー
- **numpy** - 数値計算

## 🔧 トラブルシューティング

### ⚠️ モデルファイルが見つからない
```
yolov7.pt がない場合は、以下でダウンロード：
cd detect_yolo
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### ⚠️ Flask が起動しない
```bash
# ポート 5000 が使用中の場合
lsof -i :5000
kill -9 <PID>
```

### ⚠️ 顔が検出されない
- 画像の解像度が十分か確認（最小 480x640 推奨）
- 照明条件が良いか確認
- 顔がカメラに向かっているか確認
- `--conf` パラメータを調整（デフォルト: 0.25）

## 📝 実行例

### 例 1: ウェブUIで画像をアップロード

1. `./start_app.sh` を実行
2. ブラウザで `http://localhost:5000` を開く
3. 顔画像をアップロード
4. 結果が自動表示される

### 例 2: コマンドラインで直接実行

```bash
cd detect_yolo

# 単一画像を処理
python detect.py --weights yolov7.pt --source input/photo.jpg --save-faces

# フォルダ内の全画像を処理
python detect.py --weights yolov7.pt --source input/ --save-faces

# カメラからリアルタイム処理
python detect.py --weights yolov7.pt --source 0 --save-faces
```

### 例 3: スコア詳細表示

```bash
cd detect_yolo
python << 'EOF'
import csv

with open('images/results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"ID: {row['メッシュID']}")
        print(f"  総合スコア: {row['総合スコア']}点")
        print(f"  Pitch: {row['Pitchスコア']}点（角度評価）")
        print(f"  Yaw: {row['Yawスコア']}点（左右角度）")
        print(f"  EAR: {row['EARスコア']}点（目開き）")
        print()
EOF
```

## 🔍 アルゴリズム詳細

### 1. 顔検出
- YOLOv7 で人物クラス（class 0）を検出
- 検出信度 25% 以上を採用

### 2. 顔ランドマーク抽出
- MediaPipe FaceMesh で 478 個のランドマークを検出
- 高精度フィルタリング有効

### 3. 顔の向き角度計測
- `cv2.solvePnP()` で 3D ヘッド ポーズ推定
- Pitch（上下）、Yaw（左右）を度単位で計測

### 4. 目の開き度計測
- 両眼のランドマークから EAR（Eye Aspect Ratio）を計算
- 正規化: 0.0～1.0

### 5. スコア計算
```
各スコア = 正規化値 × 配点
総合スコア = Pitch配点 + Yaw配点 + EAR配点
```

### 6. 評価ランク決定
```
if 総合スコア >= 70:
    ランク = "良好" ✅
elif 総合スコア >= 50:
    ランク = "普通" 📊
else:
    ランク = "不良" ❌
```

## 📊 出力形式

### CSV ファイル (`detect_yolo/images/results.csv`)

```csv
メッシュID,パス,総合スコア,Pitchスコア,Yawスコア,EARスコア
33,face_id_33_20251208-164747_0.jpg,60,30,30,0
```

### 顔画像
検出された顔は自動的に以下に保存されます：
```
detect_yolo/images/faces/face_id_<ID>_<TIMESTAMP>_<INDEX>.jpg
```

## 🐛 最近の修正

### ✅ スコア計算バグ修正（2024/12/08）
**問題**: CSV に保存される個別スコア（Pitch/Yaw/EAR）の合計が、総合スコアと異なっていた

**原因**: 個別スコアが 25 点満点で計算されていたが、総合スコアは 30+30+40 点で計算されていた

**修正**: `detect.py` の CSV 出力ロジックを更新
- 個別スコアを正しく 30/30/40 点スケールで保存
- `int(round(30*pitch_score))`, `int(round(30*yaw_score))`, `int(round(40*ear_score))` に変更

**検証**: 
- スコア合計が一致するようになりました
- 例: 総合スコア 60 = 30（Pitch） + 30（Yaw） + 0（EAR）

## 📞 技術サポート

問題が発生した場合：

1. **ログを確認** - ターミナルの出力メッセージを確認
2. **CSV を確認** - `detect_yolo/images/results.csv` が存在するか確認
3. **画像を確認** - `detect_yolo/images/faces/` に顔画像が保存されているか確認
4. **依存関係を確認** - `pip install -r requirements.txt`

## 📄 ライセンス

- YOLOv7: [GPL-3.0](https://github.com/WongKinYiu/yolov7)
- MediaPipe: [Apache 2.0](https://github.com/google/mediapipe)

