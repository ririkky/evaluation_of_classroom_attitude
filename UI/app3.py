import os
import csv
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request

app = Flask(__name__)

# === ★ここを修正（隣のフォルダを見に行く設定） ===
# 現在の app3.py のあるフォルダの一つ上(..)の、detect_yolo/images を指すようにする
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # UIフォルダ
DETECT_DIR = os.path.join(BASE_DIR, '..', 'detect_yolo') # detect_yoloフォルダ
BASE_OUTPUT_DIR = os.path.join(DETECT_DIR, 'images') # detect_yolo/images

CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'results.csv')
FACES_DIR = os.path.join(BASE_OUTPUT_DIR, 'faces')

# 入力画像用のテンポラリディレクトリ
TEMP_INPUT_DIR = os.path.join(DETECT_DIR, 'temp_input')
# ============================================

def load_results_from_csv():
    image_map = []

    try:
        # detect.pyが出力したCSVを読み込む（ヘッダと行をインデックスで扱い、カラムずれに耐性を持たせる）
        with open(CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)

            # ヘッダのインデックスを探す（存在しない場合は -1）
            def idx(name):
                try:
                    return header.index(name)
                except ValueError:
                    return -1

            im_idx = idx('メッシュID')
            path_idx = idx('パス')
            total_idx = idx('総合スコア')
            pitch_score_idx = idx('Pitchスコア')
            yaw_score_idx = idx('Yawスコア')
            ear_score_idx = idx('EARスコア')
            pitch_angle_idx = idx('Pitch角度(度)')
            yaw_angle_idx = idx('Yaw角度(度)')
            ear_idx = idx('EAR値')
            landmark_idx = idx('ランドマーク画像')

            def get_at(row, i):
                return row[i].strip() if (0 <= i < len(row)) else ''

            for row in reader:
                mesh_id = get_at(row, im_idx)
                path_value = get_at(row, path_idx)
                total_score = get_at(row, total_idx)
                pitch_score = get_at(row, pitch_score_idx)
                yaw_score = get_at(row, yaw_score_idx)
                ear_score = get_at(row, ear_score_idx)
                pitch_angle = get_at(row, pitch_angle_idx)
                yaw_angle = get_at(row, yaw_angle_idx)
                ear_value = get_at(row, ear_idx)
                landmark_img = get_at(row, landmark_idx)

                # ヘッダとデータのずれの可能性に備え、ランドマークの位置に数値がある場合は
                # それが実際の EAR 値（もしくはその他の数値）であり、次のカラムがランドマーク画像名である
                # ことが多いため補正する。
                def is_float_str(s):
                    try:
                        float(str(s))
                        return True
                    except Exception:
                        return False

                if landmark_img and is_float_str(landmark_img):
                    # 例: landmark_img が '0.109' のような数値になっている場合
                    # 次のカラムに画像ファイル名が入っている可能性が高い
                    next_idx = landmark_idx + 1
                    next_val = get_at(row, next_idx)
                    if next_val:
                        # もし ear_value が空または 0 系なら、ここで補正する
                        if ear_value in ('', '0', '0.0'):
                            ear_value = landmark_img
                        landmark_img = next_val

                # パスが絶対パスや重複パスになっていないか考慮し、ファイル名だけ抽出
                filename = os.path.basename(path_value)

                # 実際の画像ファイルの場所を確認
                image_path = os.path.join(FACES_DIR, filename)

                if not os.path.exists(image_path):
                    # 画像がない場合はデバッグ表示してスキップ（顔が検出されなかったフレーム）
                    print(f"⚠️ Image file not found (face not detected): {image_path}")
                    print(f"  CSV row: mesh_id={mesh_id}, filename={filename}")
                    continue

                # ブラウザ用URL (Flaskのルート経由)
                image_url = f"/images/faces/{filename}"

                # ランドマーク画像のURLも生成
                landmark_url = ""
                if landmark_img:
                    # landmark_imgはCSVに記録されたファイル名
                    landmark_path = os.path.join(FACES_DIR, landmark_img)
                    if os.path.exists(landmark_path):
                        landmark_url = f"/images/faces/{landmark_img}"
                    else:
                        # デバッグ：ファイルが見つからない場合
                        print(f"⚠️ Landmark image not found: {landmark_path}")
                        print(f"  Expected file: {landmark_img}")
                        print(f"  FACES_DIR: {FACES_DIR}")

                image_map.append({
                    'mesh_id': mesh_id,
                    'filename': filename,
                    'url': image_url,
                    'total_score': total_score,
                    'pitch_score': pitch_score,
                    'yaw_score': yaw_score,
                    'ear_score': ear_score,
                    'pitch_angle': pitch_angle,
                    'yaw_angle': yaw_angle,
                    'ear_value': ear_value,
                    'landmark_url': landmark_url,
                })

    except FileNotFoundError:
        print(f"⚠️ {CSV_PATH} が見つかりません。検出を実行しましたか？")
    except Exception as e:
        print(f"⚠️ CSV読み込みエラー: {e}")

    return image_map


def run_detect(image_path):
    """
    detect.py を実行して検出結果を生成
    """
    try:
        # CSVとfaces ディレクトリをクリア
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        if os.path.exists(FACES_DIR):
            shutil.rmtree(FACES_DIR)

        # 入力画像ディレクトリを作成
        os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
        temp_image_path = os.path.join(TEMP_INPUT_DIR, os.path.basename(image_path))
        shutil.copy(image_path, temp_image_path)

        # detect.py コマンド作成
        detect_cmd = [
            sys.executable, "detect.py",
            "--weights", "yolov7.pt",
            "--conf", "0.25",
            "--img-size", "640",
            "--source", f"temp_input/{os.path.basename(image_path)}",  # 相対パス
            "--class", "0",
            "--save-txt",
            "--save-faces"
        ]

        print(f"🔍 detect.py 実行: {' '.join(detect_cmd)}")
        print(f"📁 作業ディレクトリ: {DETECT_DIR}")

        # detect.py を実行
        result = subprocess.run(
            detect_cmd,
            cwd=DETECT_DIR,
            capture_output=True,
            text=True,
            timeout=120  # 最大120秒のタイムアウト
        )

        if result.returncode != 0:
            print(f"⚠️ detect.py エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False, f"detect.py 実行エラー: {result.stderr}"

        print(f"✅ detect.py 実行成功")
        print(f"stdout: {result.stdout}")

        # 結果ファイルが生成されたか確認
        if not os.path.exists(CSV_PATH):
            return False, f"CSV出力ファイルが生成されませんでした"

        return True, "検出完了"

    except subprocess.TimeoutExpired:
        return False, "detect.py 実行がタイムアウトしました（120秒超過）"
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False, f"エラー: {str(e)}"


@app.route('/')
def index():
    image_data = load_results_from_csv()
    return render_template('index.html', images=image_data)


@app.route('/api/images')
def get_images():
    image_data = load_results_from_csv()
    return jsonify(image_data)


@app.route('/api/detect', methods=['POST'])
def detect_endpoint():
    """
    HTMLからアップロードされた画像に対して detect.py を実行
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'ファイルが見つかりません'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'ファイルが選択されていません'}), 400

        # 許可される拡張子
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS):
            return jsonify({'success': False, 'error': 'サポートされていない画像形式です'}), 400

        # テンポラリファイルに保存
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # detect.py 実行
        success, message = run_detect(temp_path)

        # テンポラリファイル削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if success:
            return jsonify({'success': True, 'message': message}), 200
        else:
            return jsonify({'success': False, 'error': message}), 500

    except Exception as e:
        print(f"❌ API エラー: {e}")
        return jsonify({'success': False, 'error': f'サーバーエラー: {str(e)}'}), 500


# ★修正: 動画フレーム処理エンドポイント
@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """
    動画の1フレームをBase64画像として受け取り、detect.pyで処理
    """
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        frame_number = data.get('frame_number', 0)  # フレーム番号を取得

        if not image_data:
            return jsonify({'success': False, 'error': 'image_dataが必要です'}), 400

        # 最初のフレーム（frame_number == 0）の場合、CSVを初期化
        if frame_number == 0:
            csv_header = ["メッシュID", "パス", "総合スコア", "Pitchスコア", "Yawスコア", "EARスコア", "Pitch角度(度)", "Yaw角度(度)", "EAR値", "ランドマーク画像"]
            with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
            print(f"✅ CSV初期化: {CSV_PATH}")

        # Base64 画像データをファイルに保存
        import base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        frame_bytes = base64.b64decode(image_data)
        frame_filename = f'frame_{frame_number}.jpg'
        frame_path = os.path.join(TEMP_INPUT_DIR, frame_filename)
        
        with open(frame_path, 'wb') as f:
            f.write(frame_bytes)

        # detect.py を実行（フレーム処理用）
        # --append-csv オプションを追加して追記モードで実行
        detect_cmd = [
            sys.executable, "detect.py",
            "--weights", "yolov7.pt",
            "--conf", "0.25",
            "--img-size", "640",
            "--source", f"temp_input/{frame_filename}",
            "--class", "0",
            "--save-txt",
            "--save-faces",
            "--append-csv"  # 追記モード
        ]

        result = subprocess.run(
            detect_cmd,
            cwd=DETECT_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"❌ detect.py エラー: {result.stderr}")
            return jsonify({'success': False, 'error': 'detect.py実行エラー'}), 500

        print(f"✅ Frame {frame_number} 処理完了")

        # CSVから検出結果を読み込む（最後の行を取得）
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                print(f"📊 CSV行数: {len(rows)}, フレーム番号: {frame_number}")
                
                if rows:
                    last_row = rows[-1]
                    print(f"  最終行データ: mesh_id={last_row.get('メッシュID')}, path={last_row.get('パス')}")
                    # CSVのカラムずれに備えて余剰カラム(Noneキー)を確認し、必要なら EAR 値を補正
                    ear_value = last_row.get('EAR値', '0')
                    extras = last_row.get(None)
                    if extras:
                        for ex in extras:
                            if ex is None:
                                continue
                            exs = str(ex).strip()
                            try:
                                float(exs)
                                if ear_value in ('', '0', '0.0'):
                                    ear_value = exs
                                    break
                            except Exception:
                                continue

                    # ランドマーク画像の有無やEAR値から検出判定
                    landmark_detected = bool(last_row.get('ランドマーク画像')) or (str(ear_value).strip() not in ('', '0', '0.0'))

                    return jsonify({
                        'success': True,
                        'pitch_score': last_row.get('Pitchスコア', '0'),
                        'yaw_score': last_row.get('Yawスコア', '0'),
                        'ear_score': last_row.get('EARスコア', '0'),
                        'pitch_angle': last_row.get('Pitch角度(度)', '0'),
                        'yaw_angle': last_row.get('Yaw角度(度)', '0'),
                        'ear_value': ear_value,
                        'landmark_detected': landmark_detected
                    })
                else:
                    print(f"⚠️ CSVに行がありません（フレーム {frame_number}）")

        return jsonify({
            'success': True,
            'pitch_score': '0',
            'yaw_score': '0',
            'ear_score': '0'
        })

    except Exception as e:
        print(f"❌ フレーム処理エラー: {e}")
        return jsonify({'success': False, 'error': f'サーバーエラー: {str(e)}'}), 500


# ★修正: 画像ファイルを提供するためのルート
@app.route('/images/faces/<path:filename>')
def serve_faces(filename):
    return send_from_directory(FACES_DIR, filename)


@app.route('/api/mesh_images', methods=['GET'])
def get_mesh_images():
    """
    メッシュ画像を取得するエンドポイント
    """
    try:
        # メッシュ画像ディレクトリを確認
        if not os.path.exists(FACES_DIR):
            return jsonify({"error": "メッシュ画像ディレクトリが存在しません。"}), 404

        # メッシュ画像を取得
        mesh_images = []
        for filename in os.listdir(FACES_DIR):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_url = f"/images/faces/{filename}"
                mesh_images.append({"filename": filename, "url": image_url})

        return jsonify({"mesh_images": mesh_images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 新規追加: 動画の各フレームのメッシュ画像と評価値を取得するエンドポイント
@app.route('/api/video_frames', methods=['GET'])
def get_video_frames():
    """
    動画の各フレームのメッシュ画像と評価値を取得するエンドポイント
    CSVから読み込んだデータを返す
    """
    try:
        # CSVから結果を読み込む
        results = load_results_from_csv()
        
        print(f"📊 /api/video_frames: CSVから読み込んだ結果数: {len(results)}")
        
        # フレームデータを構築
        frame_data = []
        for result in results:
            frame_data.append({
                "filename": result['filename'],
                "url": result['url'],
                "landmark_url": result.get('landmark_url', ''),  # ランドマーク画像URL
                "pitch_angle": result['pitch_angle'],      # 角度（度）
                "yaw_angle": result['yaw_angle'],          # 角度（度）
                "ear_value": result['ear_value'],          # EAR値
                "pitch_score": result['pitch_score'],      # スコア
                "yaw_score": result['yaw_score'],          # スコア
                "ear_score": result['ear_score']           # スコア
            })

        return jsonify({"frames": frame_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # ポート 5000 が使用中の場合は 5001 を使用
    app.run(host='127.0.0.1', port=5001, debug=True)