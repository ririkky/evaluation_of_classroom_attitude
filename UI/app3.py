import os
import csv
import subprocess
import sys
import tempfile
import shutil
from flask import Flask, render_template, jsonify, send_from_directory, request

app = Flask(__name__)

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # UIフォルダ
DETECT_DIR = os.path.join(BASE_DIR, '..', 'detect_yolo')  # detect_yoloフォルダ
BASE_OUTPUT_DIR = os.path.join(DETECT_DIR, 'images')  # detect_yolo/images

CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'results.csv')
FACES_DIR = os.path.join(BASE_OUTPUT_DIR, 'faces')

# 入力画像用のテンポラリディレクトリ
TEMP_INPUT_DIR = os.path.join(DETECT_DIR, 'temp_input')


def load_results_from_csv():
    image_map = []

    try:
        with open(CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)

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

                def is_float_str(s):
                    try:
                        float(str(s))
                        return True
                    except Exception:
                        return False

                if landmark_img and is_float_str(landmark_img):
                    next_idx = landmark_idx + 1
                    next_val = get_at(row, next_idx)
                    if next_val:
                        if ear_value in ('', '0', '0.0'):
                            ear_value = landmark_img
                        landmark_img = next_val

                filename = os.path.basename(path_value)
                image_path = os.path.join(FACES_DIR, filename)

                if not os.path.exists(image_path):
                    print(f"⚠️ Image file not found (face not detected): {image_path}")
                    print(f"  CSV row: mesh_id={mesh_id}, filename={filename}")
                    continue

                image_url = f"/images/faces/{filename}"

                landmark_url = ""
                if landmark_img:
                    landmark_path = os.path.join(FACES_DIR, landmark_img)
                    if os.path.exists(landmark_path):
                        landmark_url = f"/images/faces/{landmark_img}"
                    else:
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
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        if os.path.exists(FACES_DIR):
            shutil.rmtree(FACES_DIR)

        os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
        temp_image_path = os.path.join(TEMP_INPUT_DIR, os.path.basename(image_path))
        shutil.copy(image_path, temp_image_path)

        detect_cmd = [
            sys.executable, "detect.py",
            "--weights", "yolov7.pt",
            "--conf", "0.25",
            "--img-size", "640",
            "--source", f"temp_input/{os.path.basename(image_path)}",
            "--class", "0",
            "--save-txt",
            "--save-faces"
        ]

        print(f"🔍 detect.py 実行: {' '.join(detect_cmd)}")
        print(f"📁 作業ディレクトリ: {DETECT_DIR}")

        result = subprocess.run(
            detect_cmd,
            cwd=DETECT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"⚠️ detect.py エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False, f"detect.py 実行エラー: {result.stderr}"

        print(f"✅ detect.py 実行成功")
        print(f"stdout: {result.stdout}")

        if not os.path.exists(CSV_PATH):
            return False, "CSV出力ファイルが生成されませんでした"

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
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'ファイルが見つかりません'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'ファイルが選択されていません'}), 400

        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS):
            return jsonify({'success': False, 'error': 'サポートされていない画像形式です'}), 400

        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        success, message = run_detect(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if success:
            return jsonify({'success': True, 'message': message}), 200
        else:
            return jsonify({'success': False, 'error': message}), 500

    except Exception as e:
        print(f"❌ API エラー: {e}")
        return jsonify({'success': False, 'error': f'サーバーエラー: {str(e)}'}), 500


@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """動画の1フレームをBase64画像として受け取り、detect.pyで処理"""
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        frame_number = data.get('frame_number', 0)

        if not image_data:
            return jsonify({'success': False, 'error': 'image_dataが必要です'}), 400

        if frame_number == 0:
            csv_header = ["メッシュID", "パス", "総合スコア", "Pitchスコア", "Yawスコア", "EARスコア", "Pitch角度(度)", "Yaw角度(度)", "EAR値", "ランドマーク画像"]
            with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
            print(f"✅ CSV初期化: {CSV_PATH}")

        import base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        frame_bytes = base64.b64decode(image_data)
        frame_filename = f'frame_{frame_number}.jpg'
        frame_path = os.path.join(TEMP_INPUT_DIR, frame_filename)

        with open(frame_path, 'wb') as f:
            f.write(frame_bytes)

        detect_cmd = [
            sys.executable, "detect.py",
            "--weights", "yolov7.pt",
            "--conf", "0.25",
            "--img-size", "640",
            "--source", f"temp_input/{frame_filename}",
            "--class", "0",
            "--save-txt",
            "--save-faces",
            "--append-csv"
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

        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                print(f"📊 CSV行数: {len(rows)}, フレーム番号: {frame_number}")

                def parse_row(row_dict):
                    ear_value_local = row_dict.get('EAR値', '0')
                    extras_local = row_dict.get(None)
                    if extras_local:
                        for ex in extras_local:
                            if ex is None:
                                continue
                            exs = str(ex).strip()
                            try:
                                float(exs)
                                if ear_value_local in ('', '0', '0.0'):
                                    ear_value_local = exs
                                    break
                            except Exception:
                                continue

                    path_val = row_dict.get('パス', '')
                    filename_only = os.path.basename(path_val)
                    face_url = f"/images/faces/{filename_only}" if filename_only else ''

                    landmark_img_val = row_dict.get('ランドマーク画像', '')
                    landmark_url = ''
                    if landmark_img_val:
                        landmark_base = os.path.basename(landmark_img_val)
                        landmark_url = f"/images/faces/{landmark_base}"

                    return {
                        'mesh_id': row_dict.get('メッシュID', ''),
                        'path': row_dict.get('パス', ''),
                        'pitch_score': row_dict.get('Pitchスコア', '0'),
                        'yaw_score': row_dict.get('Yawスコア', '0'),
                        'ear_score': row_dict.get('EARスコア', '0'),
                        'pitch_angle': row_dict.get('Pitch角度(度)', '0'),
                        'yaw_angle': row_dict.get('Yaw角度(度)', '0'),
                        'ear_value': ear_value_local,
                        'landmark_img': row_dict.get('ランドマーク画像', ''),
                        'face_url': face_url,
                        'landmark_url': landmark_url,
                    }

                faces = []
                if rows:
                    suffix = f"_{frame_number}.jpg"
                    for row in rows:
                        info = parse_row(row)
                        filename = os.path.basename(info.get('path', ''))
                        if filename.endswith(suffix):
                            faces.append(info)

                    if not faces:
                        faces.append(parse_row(rows[-1]))

                    print(f"  フレーム {frame_number} の顔数: {len(faces)}")

                    first = faces[0]
                    landmark_detected = bool(first.get('landmark_img')) or (str(first.get('ear_value', '')).strip() not in ('', '0', '0.0'))

                    return jsonify({
                        'success': True,
                        'faces': [
                            {
                                'mesh_id': f.get('mesh_id', ''),
                                'pitch_score': f.get('pitch_score', '0'),
                                'yaw_score': f.get('yaw_score', '0'),
                                'ear_score': f.get('ear_score', '0'),
                                'pitch_angle': f.get('pitch_angle', '0'),
                                'yaw_angle': f.get('yaw_angle', '0'),
                                'ear_value': f.get('ear_value', '0'),
                                'face_url': f.get('face_url', ''),
                                'landmark_url': f.get('landmark_url', ''),
                            } for f in faces
                        ],
                        'pitch_score': first.get('pitch_score', '0'),
                        'yaw_score': first.get('yaw_score', '0'),
                        'ear_score': first.get('ear_score', '0'),
                        'pitch_angle': first.get('pitch_angle', '0'),
                        'yaw_angle': first.get('yaw_angle', '0'),
                        'ear_value': first.get('ear_value', '0'),
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


@app.route('/images/faces/<path:filename>')
def serve_faces(filename):
    return send_from_directory(FACES_DIR, filename)


@app.route('/api/mesh_images', methods=['GET'])
def get_mesh_images():
    try:
        if not os.path.exists(FACES_DIR):
            return jsonify({"error": "メッシュ画像ディレクトリが存在しません。"}), 404

        mesh_images = []
        for filename in os.listdir(FACES_DIR):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_url = f"/images/faces/{filename}"
                mesh_images.append({"filename": filename, "url": image_url})

        return jsonify({"mesh_images": mesh_images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/video_frames', methods=['GET'])
def get_video_frames():
    try:
        results = load_results_from_csv()

        print(f"📊 /api/video_frames: CSVから読み込んだ結果数: {len(results)}")

        frame_data = []
        for result in results:
            frame_data.append({
                "filename": result['filename'],
                "url": result['url'],
                "landmark_url": result.get('landmark_url', ''),
                "pitch_angle": result['pitch_angle'],
                "yaw_angle": result['yaw_angle'],
                "ear_value": result['ear_value'],
                "pitch_score": result['pitch_score'],
                "yaw_score": result['yaw_score'],
                "ear_score": result['ear_score']
            })

        return jsonify({"frames": frame_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
