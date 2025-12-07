import os
import csv
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# === ★ここを修正（隣のフォルダを見に行く設定） ===
# 現在の app3.py のあるフォルダの一つ上(..)の、detect_yolo/images を指すようにする
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # UIフォルダ
DETECT_DIR = os.path.join(BASE_DIR, '..', 'detect_yolo') # detect_yoloフォルダ
BASE_OUTPUT_DIR = os.path.join(DETECT_DIR, 'images') # detect_yolo/images

CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'results.csv')
FACES_DIR = os.path.join(BASE_OUTPUT_DIR, 'faces')
# ============================================

def load_results_from_csv():
    image_map = []

    try:
        # detect.pyが出力したCSVを読み込む
        with open(CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                mesh_id      = row.get('メッシュID', '').strip()
                path_value   = row.get('パス', '').strip() # ここにはファイル名が入っている想定
                total_score  = row.get('総合スコア', '').strip()
                pitch_score  = row.get('Pitchスコア', '').strip()
                yaw_score    = row.get('Yawスコア', '').strip()
                ear_score    = row.get('EARスコア', '').strip()

                # パスが絶対パスや重複パスになっていないか考慮し、ファイル名だけ抽出
                filename = os.path.basename(path_value)
                
                # 実際の画像ファイルの場所を確認
                image_path = os.path.join(FACES_DIR, filename)
                
                if not os.path.exists(image_path):
                    # 画像がない場合はスキップ（またはデバッグ表示）
                    continue

                # ブラウザ用URL (Flaskのルート経由)
                image_url = f"/images/faces/{filename}"

                image_map.append({
                    'mesh_id': mesh_id,
                    'filename': filename,
                    'url': image_url,
                    'total_score': total_score,
                    'pitch_score': pitch_score,
                    'yaw_score': yaw_score,
                    'ear_score': ear_score,
                })

    except FileNotFoundError:
        print(f"⚠️ {CSV_PATH} が見つかりません。検出を実行しましたか？")
    except Exception as e:
        print(f"⚠️ CSV読み込みエラー: {e}")

    return image_map


@app.route('/')
def index():
    image_data = load_results_from_csv()
    return render_template('index.html', images=image_data)


@app.route('/api/images')
def get_images():
    image_data = load_results_from_csv()
    return jsonify(image_data)


# ★修正: 画像ファイルを提供するためのルート
@app.route('/images/faces/<path:filename>')
def serve_faces(filename):
    return send_from_directory(FACES_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)